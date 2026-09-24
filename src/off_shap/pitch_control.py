"""Spearman (2018) 型ピッチコントロールの、連合単位でベクトル化した実装。

定式化とパラメータは Laurie Shaw による公開実装(LaurieOnTracking,
Metrica_PitchControl.py)に従う。

- 選手 i が地点 r に到達するまでの時間(反応時間 τ_react の間は現在の速度で慣性移動し、
  その後は最高速度 v_max で直進する):
      τᵢ(r) = τ_react + ‖r − (xᵢ + vᵢ·τ_react)‖ / v_max
- 時刻 t までにボールをコントロールできる確率(ロジスティック分布):
      pᵢ(t) = 1 / (1 + exp(−π/(√3·σ) · (t − τᵢ(r))))
- 両チームのコントロール確率 P_att, P_def の時間発展(ボール到達時刻 t_ball から積分):
      dP_att/dt = (1 − P_att − P_def) · A(t),   A(t) = ∑[i ∈ 攻撃側] λᵢ pᵢ(t)
      dP_def/dt = (1 − P_att − P_def) · D(t),   D(t) = ∑[j ∈ 守備側] λⱼ pⱼ(t)

Shaw の実装は地点ごとに Euler 法でループするが、ここでは次の2点を使って
1,024 連合 × 数千地点を一度に計算する。

1. 連合 S のもとでの攻撃側レート A_S(t) は所属ベクトルに線形:
       A_S(t) = A_fixed(t) + ∑[i ∈ S] aᵢ(t)
   なので、全連合分を行列積 (連合 × 選手) @ (選手 × 地点 × 時刻) で得られる。
2. 各ステップ内でレートを一定とみなすと、未コントロール確率 U = 1 − P_att − P_def は
   厳密に指数減衰する(U ← U·exp(−(A+D)Δt))。その減少分を A : D の比で
   両チームに配分すれば、Euler 法より粗い Δt でも安定する。

Shaw の実装にある「一方が圧倒的に速い地点では 0/1 に打ち切る」近道とオフサイドの
除外は入れていない(近道は積分でも同じ値に収束する)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchControlParams:
    max_player_speed: float = 5.0  # m/s
    reaction_time: float = 0.7  # s
    tti_sigma: float = 0.45  # 到達時刻の不確実性 σ [s]
    lambda_att: float = 4.3  # コントロール率 [1/s]
    lambda_def: float = 4.3
    lambda_gk_def: float = 3 * 4.3  # 守備側GKは手を使えるのでコントロール率を上げる
    average_ball_speed: float = 15.0  # m/s
    int_dt: float = 0.04  # 積分ステップ [s]
    max_int_time: float = 10.0  # ボール到達後の積分時間 [s]


def make_grid(
    pitch_length: float, pitch_width: float, cell: float = 2.0
) -> tuple[np.ndarray, float]:
    """ピッチを cell [m] 四方のセルに分割し、セル中心 (G, 2) とセル面積を返す。"""
    nx = round(pitch_length / cell)
    ny = round(pitch_width / cell)
    xs = (np.arange(nx) + 0.5) * pitch_length / nx
    ys = (np.arange(ny) + 0.5) * pitch_width / ny
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1), (pitch_length / nx) * (pitch_width / ny)


def time_to_intercept(
    pos: np.ndarray, vel: np.ndarray, grid: np.ndarray, params: PitchControlParams
) -> np.ndarray:
    """選手 (n, 2) が各地点 (G, 2) に到達するまでの時間 (n, G)。"""
    reaction_pos = pos + vel * params.reaction_time
    dist = np.linalg.norm(grid[None, :, :] - reaction_pos[:, None, :], axis=-1)
    return params.reaction_time + dist / params.max_player_speed


def _rates(
    tti: np.ndarray, lam: np.ndarray, t: np.ndarray, params: PitchControlParams
) -> np.ndarray:
    """各選手のコントロール率 λᵢ·pᵢ(t)。tti (n, G), lam (n,), t (G, K) → (n, G, K)。"""
    k = np.pi / (np.sqrt(3.0) * params.tti_sigma)
    rates = lam[:, None, None] / (1.0 + np.exp(-k * (t[None, :, :] - tti[:, :, None])))
    return rates.astype(np.float32)


def pitch_control_coalitions(
    attack_pos: np.ndarray,
    attack_vel: np.ndarray,
    members: np.ndarray,
    defend_pos: np.ndarray,
    defend_vel: np.ndarray,
    ball_pos: np.ndarray,
    grid: np.ndarray,
    attack_fixed_pos: np.ndarray | None = None,
    attack_fixed_vel: np.ndarray | None = None,
    defend_gk_pos: np.ndarray | None = None,
    defend_gk_vel: np.ndarray | None = None,
    params: PitchControlParams | None = None,
    chunk_size: int = 32,
) -> np.ndarray:
    """連合ごとの攻撃側ピッチコントロール (C, G)。

    attack_pos/vel (n, 2): 連合の対象になる攻撃側選手
    members (C, n) bool: 各連合に含まれる選手
    attack_fixed_* (m, 2): 連合によらず常に存在する攻撃側(GK など)
    defend_* (k, 2): 守備側フィールドプレイヤー。defend_gk_* (1, 2) は λ_gk_def を使う
    """
    params = params or PitchControlParams()
    empty = np.zeros((0, 2))

    def valid(pos, vel):
        if pos is None:
            return empty, empty
        ok = ~np.isnan(pos).any(axis=-1)
        return pos[ok], vel[ok]

    fixed_pos, fixed_vel = valid(attack_fixed_pos, attack_fixed_vel)
    gk_pos, gk_vel = valid(defend_gk_pos, defend_gk_vel)
    def_pos = np.concatenate([defend_pos, gk_pos])
    def_vel = np.concatenate([defend_vel, gk_vel])
    def_lam = np.concatenate(
        [np.full(len(defend_pos), params.lambda_def), np.full(len(gk_pos), params.lambda_gk_def)]
    )
    # 1,024 連合 × 地点 × 時刻の中間配列が大きくなるため float32 で計算する
    member_f = members.astype(np.float32)

    n_steps = round(params.max_int_time / params.int_dt)
    out = np.empty((members.shape[0], grid.shape[0]))
    for g0 in range(0, grid.shape[0], chunk_size):
        g = grid[g0 : g0 + chunk_size]
        ball_time = np.linalg.norm(g - ball_pos, axis=-1) / params.average_ball_speed
        t = ball_time[:, None] + params.int_dt * np.arange(n_steps)[None, :]  # (Gc, K)
        t = t.astype(np.float32)

        att_rates = _rates(
            time_to_intercept(attack_pos, attack_vel, g, params),
            np.full(len(attack_pos), params.lambda_att),
            t,
            params,
        )  # (n, Gc, K)
        A = np.einsum("cn,ngk->cgk", member_f, att_rates)  # (C, Gc, K)
        if len(fixed_pos):
            A += _rates(
                time_to_intercept(fixed_pos, fixed_vel, g, params),
                np.full(len(fixed_pos), params.lambda_att),
                t,
                params,
            ).sum(axis=0)
        D = _rates(time_to_intercept(def_pos, def_vel, g, params), def_lam, t, params).sum(axis=0)

        total = A + D[None]
        # 各ステップ開始時点の未コントロール確率 U_k と、そのステップでの減少量
        log_u = -np.cumsum(total * params.int_dt, axis=-1)
        u_end = np.exp(log_u)
        u_start = np.concatenate([np.ones_like(u_end[..., :1]), u_end[..., :-1]], axis=-1)
        share = np.divide(A, total, out=np.zeros_like(A), where=total > 0)
        out[:, g0 : g0 + chunk_size] = (share * (u_start - u_end)).sum(axis=-1)
    return out


def pitch_control(
    attack_pos: np.ndarray,
    attack_vel: np.ndarray,
    defend_pos: np.ndarray,
    defend_vel: np.ndarray,
    ball_pos: np.ndarray,
    grid: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """全攻撃側選手がいる場合のピッチコントロール (G,)。"""
    members = np.ones((1, len(attack_pos)), dtype=bool)
    return pitch_control_coalitions(
        attack_pos, attack_vel, members, defend_pos, defend_vel, ball_pos, grid, **kwargs
    )[0]
