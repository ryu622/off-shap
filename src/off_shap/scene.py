"""カウンターシーンの前処理済み軌道(以降のパイプライン共通の中間形式)。

Shapley計算(Stage 1)・軌道補完モデル(Stage 2)はこの `Scene` だけを入力に取る。
研究室データに差し替える場合も、`Scene` を作る変換層を書けばよい。

- 座標は攻撃側が常に +x 方向へ攻める向き(counter.to_attack_frame)
- 連合 S の対象は攻撃側フィールドプレイヤー(attack_*)。GK は連合から除外するが、
  ピッチコントロール等の空間指標の計算には必要なので *_gk_* として別に保持する
- 速度は Savitzky-Golay フィルタで平滑化した位置の微分
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from off_shap.counter import to_attack_frame
from off_shap.loader import MatchData

SAVGOL_WINDOW = 7  # 0.28秒 @ 25Hz
SAVGOL_POLYORDER = 2

# 試合単位の分割(同一試合のフレームが複数の split にまたがらないようにする)。
# Stage 1 は学習を伴わないため全試合を評価に使い、この分割は Stage 2 で使う。
SPLITS = {
    "J03WPY": "train",
    "J03WMX": "train",
    "J03WN1": "train",
    "J03WOH": "train",
    "J03WOY": "train",
    "J03WQQ": "valid",
    "J03WR9": "test",
}


@dataclass
class Scene:
    scene_id: str
    match_id: str
    period_id: int
    team_id: str
    ground: str
    fps: float
    start_frame_id: int
    end_frame_id: int
    outcome: str
    xg: float
    attack_ids: np.ndarray  # (n_a,)
    defend_ids: np.ndarray  # (n_d,)
    attack_pos: np.ndarray  # (T, n_a, 2) m
    attack_vel: np.ndarray  # (T, n_a, 2) m/s
    defend_pos: np.ndarray  # (T, n_d, 2)
    defend_vel: np.ndarray  # (T, n_d, 2)
    attack_gk_pos: np.ndarray  # (T, 2)。不在なら NaN
    defend_gk_pos: np.ndarray  # (T, 2)
    ball_pos: np.ndarray  # (T, 2)
    max_raw_speed: float  # 平滑化前の攻守フィールドプレイヤーの最大速度 [m/s]。座標の飛びの検出用

    @property
    def n_frames(self) -> int:
        return self.attack_pos.shape[0]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)}
        )

    @classmethod
    def load(cls, path: Path) -> Scene:
        with np.load(path, allow_pickle=False) as z:
            kwargs = {k: z[k] for k in z.files}
        for f in fields(cls):
            if f.type in ("str", "int", "float"):
                kwargs[f.name] = {"str": str, "int": int, "float": float}[f.type](kwargs[f.name])
        return cls(**kwargs)


def scene_id_of(row: pd.Series) -> str:
    return f"{row.match_id}_{row.period_id}_{row.start_frame_id}"


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """時間方向の欠損を線形補間し、端は最近傍で埋める。全欠損の系列は NaN のまま。"""
    out = a.copy()
    t = np.arange(a.shape[0])
    flat = out.reshape(a.shape[0], -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = ~np.isnan(col)
        if ok.any() and not ok.all():
            flat[:, j] = np.interp(t, t[ok], col[ok])
    return out


def raw_max_speed(pos: np.ndarray, fps: float) -> float:
    """平滑化前の1フレーム移動量から求めた最大速度。平滑化すると座標の飛びが
    周辺フレームに薄まって見えにくくなるため、品質チェックはこちらで行う。"""
    if pos.shape[0] < 2:
        return 0.0
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    return float(np.nanmax(step) * fps) if np.isfinite(step).any() else 0.0


def smooth_and_velocity(pos: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """位置 (T, ..., 2) を平滑化し、速度 [m/s] を返す。短すぎる系列は有限差分で代替。"""
    pos = _fill_nan(pos)
    T = pos.shape[0]
    if T >= SAVGOL_WINDOW:
        smoothed = savgol_filter(pos, SAVGOL_WINDOW, SAVGOL_POLYORDER, axis=0)
        vel = savgol_filter(pos, SAVGOL_WINDOW, SAVGOL_POLYORDER, deriv=1, delta=1 / fps, axis=0)
    else:
        smoothed = pos
        vel = np.gradient(pos, 1 / fps, axis=0) if T > 1 else np.zeros_like(pos)
    return smoothed, vel


def build_scene(match: MatchData, row: pd.Series) -> Scene:
    """カウンター候補1件(counter.extract_candidates の1行)から Scene を作る。"""
    meta = match.meta
    L, W = meta.pitch_length, meta.pitch_width
    frame_ids = np.arange(row.start_frame_id, row.end_frame_id + 1)
    T = len(frame_ids)

    gk_ids = {p.player_id for p in meta.players if p.is_goalkeeper}
    pl = match.players[match.players.frame_id.between(row.start_frame_id, row.end_frame_id)]
    on_pitch_at_start = pl[pl.frame_id == row.start_frame_id]

    def ids_of(team_is_attack: bool, goalkeeper: bool) -> list[str]:
        sel = on_pitch_at_start[(on_pitch_at_start.team_id == row.team_id) == team_is_attack]
        return sorted(pid for pid in sel.player_id if (pid in gk_ids) == goalkeeper)

    attack_ids, defend_ids = ids_of(True, False), ids_of(False, False)

    # (frame, player) の long 形式を、行=フレーム・列=選手のワイド形式に変換
    wide_x = pl.pivot(index="frame_id", columns="player_id", values="x").reindex(frame_ids)
    wide_y = pl.pivot(index="frame_id", columns="player_id", values="y").reindex(frame_ids)

    def positions(ids: list[str]) -> np.ndarray:
        x = wide_x.reindex(columns=ids).to_numpy(dtype=float)
        y = wide_y.reindex(columns=ids).to_numpy(dtype=float)
        ax, ay = to_attack_frame(x, y, row.ground, L, W)
        return np.stack([ax, ay], axis=-1).reshape(T, len(ids), 2)

    raw_attack, raw_defend = positions(attack_ids), positions(defend_ids)
    max_raw_speed = max(
        raw_max_speed(raw_attack, meta.frame_rate), raw_max_speed(raw_defend, meta.frame_rate)
    )
    attack_pos, attack_vel = smooth_and_velocity(raw_attack, meta.frame_rate)
    defend_pos, defend_vel = smooth_and_velocity(raw_defend, meta.frame_rate)

    def gk_positions(team_is_attack: bool) -> np.ndarray:
        ids = ids_of(team_is_attack, True)
        if not ids:
            return np.full((T, 2), np.nan)
        return _fill_nan(positions(ids[:1])[:, 0])

    fr = match.frames.set_index("frame_id").loc[frame_ids]
    bx, by = to_attack_frame(fr.ball_x.to_numpy(), fr.ball_y.to_numpy(), row.ground, L, W)

    return Scene(
        scene_id=scene_id_of(row),
        match_id=row.match_id,
        period_id=int(row.period_id),
        team_id=row.team_id,
        ground=row.ground,
        fps=meta.frame_rate,
        start_frame_id=int(row.start_frame_id),
        end_frame_id=int(row.end_frame_id),
        outcome=row.outcome,
        xg=float(row.xg),
        attack_ids=np.array(attack_ids),
        defend_ids=np.array(defend_ids),
        attack_pos=attack_pos,
        attack_vel=attack_vel,
        defend_pos=defend_pos,
        defend_vel=defend_vel,
        attack_gk_pos=gk_positions(True),
        defend_gk_pos=gk_positions(False),
        ball_pos=_fill_nan(np.stack([bx, by], axis=-1)),
        max_raw_speed=max_raw_speed,
    )
