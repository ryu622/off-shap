"""カウンターアタック候補シーンの抽出。

hs-pinn(卒論定義の再実装)は「奪取イベント時刻 + 5秒」の固定窓だったが、
本研究は「奪取 → シュート/ロスト」の可変長シーケンスを扱うため作り替えている。

idsse-data ではイベント時刻とトラッキングの同期ずれが試合ごとに ±1〜2秒あり
(documents/data_exploration.md)、イベント時刻を始点・終点に直接使うと
シーン境界が不安定になる。そこで境界はトラッキング側の保持チームフラグ
(owning_team_id)・ボール状態(ball_alive)から決め、イベントは
「対応する奪取イベントがあるか」「区間内にシュートがあったか」の
ラベル付けにのみ使う。

- 始点: インプレー中に保持チームが切り替わり、新しい保持が MIN_POSSESSION_S 以上続くフレーム
- 終点: 相手への保持切り替わり(ロスト) / アウトオブプレー / MAX_DURATION_S 経過 の最初
- 結果: 区間内に攻撃側のシュートがあれば shot(ゴールなら goal)、なければ終点の種類

ここではカウンターらしさで絞り込まず、判断材料となる特徴量を付けた候補を
すべて返す。絞り込みは `is_counter` で行う(閾値は分布確認後に調整する)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from off_shap.loader import MatchData

MIN_POSSESSION_S = 1.0  # 保持フラグの一瞬の揺れ(デュエル中など)を除外する最短保持時間
PRE_ALIVE_S = 1.0  # 切り替わり直前もインプレーだったことを要求する時間(リスタート直後を除外)
MAX_DURATION_S = 20.0  # これを超えるとカウンターではなく遅攻に移行したとみなし打ち切る
RECOVERY_MATCH_WINDOW_S = 3.0  # 奪取イベントとの対応付けを許容する時間差
SHOT_GRACE_S = 2.0  # 終点後もシュートを区間に含める猶予(イベント時刻の遅れを吸収)

RECOVERY_GENERIC_NAMES = {"TacklingGame", "BallDeflection"}


@dataclass(frozen=True)
class CounterCriteria:
    """カウンターらしさの暫定条件。分布を見ながら調整する前提。"""

    # 奪取位置(攻撃方向基準, m)。敵陣深くでの奪取はショートカウンターとして別途検討
    max_start_x: float = 70.0
    min_duration_s: float = 2.0
    min_progress_m: float = 30.0  # 最初の progress_window_s 秒でのボール前進距離
    progress_window_s: float = 10.0


def to_attack_frame(
    x: np.ndarray, y: np.ndarray, ground: str, pitch_length: float, pitch_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """STATIC_HOME_AWAY 座標を「攻撃側が常に +x 方向へ攻める」座標に変換する。
    away の場合は 180°回転(x, y とも反転)なので左右の対応も保たれる。"""
    if ground == "home":
        return x, y
    return pitch_length - x, pitch_width - y


def recovery_events(events: pd.DataFrame) -> pd.DataFrame:
    """ボール支配が実際に相手へ移った奪取イベント(hs-pinnと同じ判定)。

    RECOVERY(BallClaiming由来)は無条件。TacklingGame/BallDeflection は
    デュエルの勝敗を表すだけなので PossessionChange=="true" を要求し、
    奪取側は WinnerTeam から取る。
    """
    is_rec = events.event_type == "RECOVERY"
    is_duel = events.event_name.isin(RECOVERY_GENERIC_NAMES) & (events.possession_change == "true")
    rec = events[is_rec | is_duel].copy()
    rec["recovering_team_id"] = rec.team_id.where(rec.event_type == "RECOVERY", rec.winner_team_id)
    return rec


def _runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同じ値が続く区間の (開始インデックス, 長さ)。"""
    if len(values) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    change = np.ones(len(values), dtype=bool)
    change[1:] = values[1:] != values[:-1]
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, len(values)))
    return starts, lengths


def smooth_owner(owner: np.ndarray, min_run: int) -> np.ndarray:
    """min_run フレーム未満の保持区間(デュエル中の一瞬の揺れなど)を直前の保持に吸収する。

    これをしないと A→B(数フレーム)→A のような揺れで、A の保持継続が
    「A による再奪取」として別シーンに数えられてしまう。
    """
    smoothed = owner.copy()
    starts, lengths = _runs(owner)
    for i, (s, n) in enumerate(zip(starts, lengths)):
        is_last = i == len(starts) - 1
        if i > 0 and n < min_run and not is_last:
            smoothed[s : s + n] = smoothed[s - 1]
    return smoothed


def _find_end(
    owner: np.ndarray, alive: np.ndarray, start: int, team_id: str, max_len: int
) -> tuple[int, str]:
    """start 以降で最初の終了フレーム(そのフレーム自体は含まない)と終了理由。
    owner は smooth_owner 済みであること。"""
    stop = min(start + max_len, len(owner))
    for t in range(start, stop):
        if not alive[t]:
            return t, "dead"
        if owner[t] != team_id:
            return t, "lost"
    return stop, "timeout" if stop == start + max_len else "period_end"


def _progress_within(ball_x: np.ndarray, sec: float, fps: float) -> float:
    """始点から sec 秒以内に到達したボールの最大前進距離(攻撃方向基準)。"""
    k = min(round(sec * fps), len(ball_x) - 1)
    return float(np.nanmax(ball_x[: k + 1]) - ball_x[0])


def extract_candidates(match: MatchData) -> pd.DataFrame:
    """1試合分のカウンター候補(=オープンプレー中のターンオーバー起点のポゼッション)。"""
    meta = match.meta
    fps = meta.frame_rate
    min_run = round(MIN_POSSESSION_S * fps)
    pre_alive = round(PRE_ALIVE_S * fps)
    max_len = round(MAX_DURATION_S * fps)

    rec = recovery_events(match.events)
    shots = match.events[match.events.event_type == "SHOT"]

    rows = []
    for period_id, fp in match.frames.groupby("period_id", sort=True):
        fp = fp.sort_values("frame_id").reset_index(drop=True)
        owner = smooth_owner(fp.owning_team_id.to_numpy(), min_run)
        alive = fp.ball_alive.to_numpy()
        t = fp.t.to_numpy()
        starts, lengths = _runs(owner)

        for s, n in zip(starts[1:], lengths[1:]):
            team_id = owner[s]
            if team_id is None or n < min_run:
                continue
            if s < pre_alive or not alive[s - pre_alive : s + 1].all():
                continue

            end, end_reason = _find_end(owner, alive, s, team_id, max_len)
            ground = meta.ground_of(team_id)
            bx, _ = to_attack_frame(
                fp.ball_x.to_numpy()[s:end],
                fp.ball_y.to_numpy()[s:end],
                ground,
                meta.pitch_length,
                meta.pitch_width,
            )
            if len(bx) == 0 or np.isnan(bx[0]):
                continue

            t0, t1 = t[s], t[end - 1]
            rec_near = rec[
                (rec.period_id == period_id)
                & (rec.recovering_team_id == team_id)
                & ((rec.t - t0).abs() <= RECOVERY_MATCH_WINDOW_S)
            ]
            seq_shots = shots[
                (shots.period_id == period_id)
                & (shots.team_id == team_id)
                & (shots.t >= t0)
                & (shots.t <= t1 + SHOT_GRACE_S)
            ]

            if len(seq_shots):
                first_shot = seq_shots.iloc[0]
                outcome = "goal" if first_shot.result == "GOAL" else "shot"
                xg = float(seq_shots.xg.max())
                dfl_counter = first_shot.counter_attack_flag == "true"
            else:
                outcome, xg, dfl_counter = end_reason, 0.0, False

            rows.append(
                {
                    "match_id": meta.match_id,
                    "period_id": int(period_id),
                    "team_id": team_id,
                    "ground": ground,
                    "start_frame_id": int(fp.frame_id[s]),
                    "end_frame_id": int(fp.frame_id[end - 1]),
                    "start_t": float(t0),
                    "duration_s": (end - s) / fps,
                    "end_reason": end_reason,
                    "outcome": outcome,
                    "xg": xg,
                    "dfl_counter_flag": dfl_counter,
                    "start_x": float(bx[0]),
                    "max_x": float(np.nanmax(bx)),
                    "end_x": float(bx[-1]),
                    "progress_5s": _progress_within(bx, 5.0, fps),
                    "progress_10s": _progress_within(bx, 10.0, fps),
                    "has_recovery_event": len(rec_near) > 0,
                    "recovery_type": (rec_near.iloc[0].sub_type if len(rec_near) else None),
                }
            )

    return pd.DataFrame(rows)


def is_counter(candidates: pd.DataFrame, criteria: CounterCriteria | None = None) -> pd.Series:
    criteria = criteria or CounterCriteria()
    progress_col = f"progress_{int(criteria.progress_window_s)}s"
    return (
        (candidates.start_x <= criteria.max_start_x)
        & (candidates.duration_s >= criteria.min_duration_s)
        & (candidates[progress_col] >= criteria.min_progress_m)
    )
