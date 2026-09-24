"""Stage 1(静的削除版)の worth 関数 v(S)。

    v_w(S) = (1 / |ピッチ|) · ∑[c] PC_S(c) · w(c) · ΔA

重み w は次の3種類を同時に計算する(ピッチコントロールの計算が支配的で、
重みを増やすコストは行列積1回分):

- xt          : w(c) = xT(c)
- xt_opp_half : w(c) = xT(c) · [c が相手陣内]
- xt_gain     : w(c) = max(xT(c) − xT(ボール位置), 0)

xt だけだと、自陣の広い面積(xT は小さいが面積が大きい)の支配が v(N) の
4〜5割を占め、最終ラインに残った選手にも脅威と無関係な φ が付く
(documents/stage1_worth.md)。残り2つはそれを除く代替案。

PC_S は連合 S に含まれない攻撃側フィールドプレイヤーを入力から除いて再計算した
ピッチコントロール。守備側と攻撃側GKは常に記録どおりの位置に置く。ピッチ面積で
割っているので、グリッド解像度によらず「ピッチ全体での、支配確率で重み付けした
xT の平均」として解釈できる。

1フレームごとに全 2ⁿ 連合の値を計算して保存し、シーン単位の集計(終端・平均など)は
後段で選べるようにする。Shapley 値は v に対して線形なので、
「フレーム平均した v の Shapley 値」=「フレームごとの Shapley 値の平均」。

既知の限界: 選手を除いても他の選手は記録どおりに動く前提なので、
デコイが守備を引きつけた効果(二次的反応)はデコイ本人の値には入らない。
これを捉えるのが Stage 2 の役割。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from off_shap.pitch_control import PitchControlParams, make_grid, pitch_control_coalitions
from off_shap.scene import Scene
from off_shap.shapley import all_coalitions
from off_shap.xt import XTGrid

# 刻みを Shaw の 0.04s → 0.1s に粗くしても最大誤差 0.013(平均 0.0005)で、計算は約2.6倍速い
# (documents/stage1_worth.md)。積分時間 10s は短くすると遠方の地点で大きく崩れるので維持する。
STAGE1_PC_PARAMS = PitchControlParams(int_dt=0.1)
GRID_CELL_M = 2.0
WEIGHT_NAMES = ("xt", "xt_opp_half", "xt_gain")
# 主指標(2026-09-23 確定、documents/stage1_worth.md 5.5節):
# 相手陣内の xT で重み付けし、シーン内の各フレームの φ を平均する
PRIMARY_WEIGHT = "xt_opp_half"
PRIMARY_AGGREGATION = "mean"


@dataclass
class Stage1Worth:
    xt: XTGrid
    params: PitchControlParams = STAGE1_PC_PARAMS
    cell: float = GRID_CELL_M

    def __post_init__(self) -> None:
        self.grid, cell_area = make_grid(self.xt.pitch_length, self.xt.pitch_width, self.cell)
        self._scale = cell_area / (self.xt.pitch_length * self.xt.pitch_width)
        self.xt_at_grid = self.xt.at(self.grid)
        self.weights = self.xt_at_grid * self._scale  # (G,)。重み xt

    def weight_matrix(self, scene: Scene, t: int) -> np.ndarray:
        """フレーム t の重み (G, K)。列の順は WEIGHT_NAMES。"""
        opp_half = self.grid[:, 0] >= self.xt.pitch_length / 2
        xt_ball = self.xt.at(scene.ball_pos[t][None])[0]
        w = np.stack(
            [
                self.xt_at_grid,
                self.xt_at_grid * opp_half,
                np.maximum(self.xt_at_grid - xt_ball, 0.0),
            ],
            axis=-1,
        )
        return w * self._scale

    def pitch_control(self, scene: Scene, t: int, members: np.ndarray) -> np.ndarray:
        """フレーム t における連合ごとのピッチコントロール (C, G)。"""
        return pitch_control_coalitions(
            scene.attack_pos[t],
            scene.attack_vel[t],
            members,
            scene.defend_pos[t],
            scene.defend_vel[t],
            scene.ball_pos[t],
            self.grid,
            attack_fixed_pos=scene.attack_gk_pos[t][None],
            attack_fixed_vel=_gk_velocity(scene.attack_gk_pos, t, scene.fps)[None],
            defend_gk_pos=scene.defend_gk_pos[t][None],
            defend_gk_vel=_gk_velocity(scene.defend_gk_pos, t, scene.fps)[None],
            params=self.params,
        )

    def coalition_values(self, scene: Scene, t: int) -> np.ndarray:
        """フレーム t における全連合の v(S) (2ⁿ, K)。行は連合のビットマスク、列は WEIGHT_NAMES。"""
        members = all_coalitions(len(scene.attack_ids))
        return self.pitch_control(scene, t, members) @ self.weight_matrix(scene, t)


def _gk_velocity(gk_pos: np.ndarray, t: int, fps: float) -> np.ndarray:
    """GK の速度(Scene には保存していないので中心差分で求める)。"""
    if np.isnan(gk_pos).all():
        return np.zeros(2)
    lo, hi = max(t - 1, 0), min(t + 1, len(gk_pos) - 1)
    if hi == lo:
        return np.zeros(2)
    return (gk_pos[hi] - gk_pos[lo]) * fps / (hi - lo)


def sample_frames(n_frames: int, fps: float, stride_s: float) -> np.ndarray:
    """シーンから stride_s 秒ごとにフレームを選ぶ。終端フレームは必ず含める。"""
    stride = max(round(stride_s * fps), 1)
    frames = np.arange(0, n_frames, stride)
    if frames[-1] != n_frames - 1:
        frames = np.append(frames, n_frames - 1)
    return frames
