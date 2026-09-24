"""Expected Threat (xT; Singh, 2019) の価値面。

ピッチを nx × ny のセルに分け、各セルでボールを持ったときに
この先ゴールにつながる確率を、パス・シュートのイベントから学習する:

    xT(c) = P_shot(c) · P_goal(c) + P_move(c) · ∑[c'] T(c → c') · xT(c')

公開されている xT の数値グリッドは再利用しにくいため(hs-pinn の
scripts/build_xt_grid.py と同じ判断)、idsse-data 7試合から自前で学習する。
シュートが少ないセルはゴール確率が不安定になるので、全体のゴール率に
向けて縮小推定する。

ピッチは左右対称と仮定し、学習したグリッドを y 方向に反転したものと平均する
(7試合ではシュートが171本しかなく、左右の差はほぼノイズのため)。
価値面として使うときは、セル中心の値を双線形補間して滑らかにする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from off_shap.counter import to_attack_frame
from off_shap.loader import MatchData

GOAL_PRIOR_SHOTS = 10.0  # ゴール確率の縮小推定で、全体平均を何本分のシュートとみなすか
N_ITER = 30


@dataclass
class XTGrid:
    values: np.ndarray  # (nx, ny)。x は攻撃方向
    pitch_length: float
    pitch_width: float
    n_passes: int
    n_shots: int
    n_goals: int

    def interpolator(self) -> RegularGridInterpolator:
        nx, ny = self.values.shape
        xs = (np.arange(nx) + 0.5) * self.pitch_length / nx
        ys = (np.arange(ny) + 0.5) * self.pitch_width / ny
        # ピッチ端はセル中心の外側になるので、端のセル値を延長する
        xs = np.concatenate([[0.0], xs, [self.pitch_length]])
        ys = np.concatenate([[0.0], ys, [self.pitch_width]])
        padded = np.pad(self.values, 1, mode="edge")
        return RegularGridInterpolator((xs, ys), padded, bounds_error=False, fill_value=None)

    def at(self, points: np.ndarray) -> np.ndarray:
        """攻撃方向基準の座標 (G, 2) における xT(双線形補間)。"""
        clipped = np.clip(points, [0, 0], [self.pitch_length, self.pitch_width])
        return self.interpolator()(clipped)


def _actions_in_attack_frame(match: MatchData) -> pd.DataFrame:
    meta = match.meta
    ev = match.events
    ev = ev[ev.event_type.isin(["PASS", "SHOT"]) & ev.team_id.notna() & ev.x.notna()].copy()
    for ground in ("home", "away"):
        team_id = meta.home_team_id if ground == "home" else meta.away_team_id
        sel = ev.team_id == team_id
        for a, b in (("x", "y"), ("end_x", "end_y")):
            ev.loc[sel, a], ev.loc[sel, b] = to_attack_frame(
                ev.loc[sel, a].to_numpy(),
                ev.loc[sel, b].to_numpy(),
                ground,
                meta.pitch_length,
                meta.pitch_width,
            )
    return ev


def fit_xt(matches: list[MatchData], nx: int = 12, ny: int = 8, symmetrize: bool = True) -> XTGrid:
    L, W = matches[0].meta.pitch_length, matches[0].meta.pitch_width
    actions = pd.concat([_actions_in_attack_frame(m) for m in matches], ignore_index=True)

    def cell(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        cx = np.clip((x.to_numpy() / L * nx).astype(int), 0, nx - 1)
        cy = np.clip((y.to_numpy() / W * ny).astype(int), 0, ny - 1)
        return cx, cy

    total = np.zeros((nx, ny))
    shots = np.zeros((nx, ny))
    goals = np.zeros((nx, ny))
    moves = np.zeros((nx, ny, nx, ny))

    cx, cy = cell(actions.x, actions.y)
    np.add.at(total, (cx, cy), 1)
    is_shot = (actions.event_type == "SHOT").to_numpy()
    is_goal = is_shot & (actions.result == "GOAL").to_numpy()
    np.add.at(shots, (cx[is_shot], cy[is_shot]), 1)
    np.add.at(goals, (cx[is_goal], cy[is_goal]), 1)

    done = (
        (actions.event_type == "PASS") & (actions.result == "COMPLETE") & actions.end_x.notna()
    ).to_numpy()
    ex, ey = cell(actions.end_x[done], actions.end_y[done])
    np.add.at(moves, (cx[done], cy[done], ex, ey), 1)

    denom = np.where(total > 0, total, 1.0)
    shot_prob = shots / denom
    global_goal_rate = goals.sum() / max(shots.sum(), 1.0)
    goal_prob = (goals + GOAL_PRIOR_SHOTS * global_goal_rate) / (shots + GOAL_PRIOR_SHOTS)
    transition = moves / denom[:, :, None, None]  # 失敗パスは遷移先なし(価値0)として扱う

    xt = np.zeros((nx, ny))
    for _ in range(N_ITER):
        xt = shot_prob * goal_prob + np.einsum("ijkl,kl->ij", transition, xt)
    if symmetrize:
        xt = (xt + xt[:, ::-1]) / 2

    return XTGrid(
        values=xt,
        pitch_length=L,
        pitch_width=W,
        n_passes=int((actions.event_type == "PASS").sum()),
        n_shots=int(shots.sum()),
        n_goals=int(goals.sum()),
    )


def load_xt(path) -> XTGrid:
    with np.load(path) as z:
        return XTGrid(
            values=z["values"],
            pitch_length=float(z["pitch_length"]),
            pitch_width=float(z["pitch_width"]),
            n_passes=int(z["n_passes"]),
            n_shots=int(z["n_shots"]),
            n_goals=int(z["n_goals"]),
        )
