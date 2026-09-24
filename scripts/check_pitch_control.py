"""ピッチコントロールと v(S) の目視確認用の図を作る。

1シーンの1フレームについて、
(1) 全選手がいる場合のピッチコントロール
(2) それに xT を掛けた「危険度で重み付けした支配」
(3) 1人を除いたときに失われる支配(PC_N − PC_{N∖{i}})
を描く。

実行: uv run python scripts/check_pitch_control.py [scene_id] [秒(終端からの遡り)]
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from off_shap.loader import PROJECT_ROOT
from off_shap.scene import Scene
from off_shap.worth import Stage1Worth
from off_shap.xt import load_xt

SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"
XT_PATH = PROJECT_ROOT / "data" / "processed" / "xt_grid.npz"
FIG_PATH = PROJECT_ROOT / "documents" / "images" / "pitch_control_example.png"

ATTACK = "#2a78d6"
DEFEND = "#eb6834"


def main() -> None:
    index = pd.read_parquet(INDEX_PATH)
    if len(sys.argv) > 1:
        scene_id = sys.argv[1]
    else:
        shots = index[index.outcome.isin(["shot", "goal"]) & ~index.tracking_glitch]
        scene_id = shots.sort_values("xg").scene_id.iloc[-1]
    back_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    scene = Scene.load(SCENE_DIR / f"{scene_id}.npz")
    worth = Stage1Worth(load_xt(XT_PATH))
    t = max(scene.n_frames - 1 - round(back_s * scene.fps), 0)
    n = len(scene.attack_ids)

    # 全員 + 1人ずつ除いた連合
    members = np.ones((n + 1, n), dtype=bool)
    for i in range(n):
        members[i + 1, i] = False
    pc = worth.pitch_control(scene, t, members)
    v = pc @ worth.weights  # 重み xt
    drop = v[0] - v[1:]
    key = int(np.argmax(drop))
    print(
        f"scene={scene_id} outcome={scene.outcome} xg={scene.xg:.3f} frame={t}/{scene.n_frames - 1}"
    )
    print(f"v(N)={v[0]:.5f}  1人除いたときの v の低下: {np.round(drop, 5)}")

    nx, ny = round(worth.xt.pitch_length / worth.cell), round(worth.xt.pitch_width / worth.cell)
    to_img = lambda a: a.reshape(nx, ny).T
    extent = (0, worth.xt.pitch_length, 0, worth.xt.pitch_width)

    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=extent[1],
        pitch_width=extent[3],
        line_color="#52514e",
        linewidth=0.8,
        line_zorder=2,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), facecolor="#fcfcfb")
    panels = [
        (to_img(pc[0]), "bwr_r", (0, 1), "ピッチコントロール(青=攻撃側)"),
        (to_img(pc[0] * worth.xt.at(worth.grid)), "Blues", None, "PC × xT"),
        (
            to_img(pc[0] - pc[key + 1]),
            "Blues",
            (0, None),
            f"選手 {key} を除くと失われる支配(Δv={drop[key]:.4f})",
        ),
    ]
    for ax, (img, cmap, lim, title) in zip(axes, panels):
        pitch.draw(ax=ax)
        vmin, vmax = lim if lim else (0, None)
        ax.imshow(
            img, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, zorder=0, alpha=0.9
        )
        ap, dp = scene.attack_pos[t], scene.defend_pos[t]
        ax.scatter(ap[:, 0], ap[:, 1], c=ATTACK, edgecolors="white", s=60, zorder=3)
        ax.quiver(
            ap[:, 0],
            ap[:, 1],
            scene.attack_vel[t, :, 0],
            scene.attack_vel[t, :, 1],
            color=ATTACK,
            scale=60,
            width=0.004,
            zorder=3,
        )
        ax.scatter(dp[:, 0], dp[:, 1], c=DEFEND, edgecolors="white", s=60, zorder=3)
        ax.scatter(
            *scene.defend_gk_pos[t], c=DEFEND, marker="s", edgecolors="white", s=60, zorder=3
        )
        ax.scatter(*scene.ball_pos[t], c="black", s=25, zorder=4)
        for i, (x, y) in enumerate(ap):
            ax.annotate(
                str(i), (x, y), color="white", ha="center", va="center", fontsize=6, zorder=5
            )
        ax.scatter(*ap[key], facecolors="none", edgecolors="black", s=160, lw=1.2, zorder=4)
        ax.set_title(title, loc="left")
    fig.suptitle(f"{scene_id}(→ 右方向へ攻撃、青=攻撃側、橙=守備側、■=守備GK)", x=0.01, ha="left")
    fig.savefig(FIG_PATH, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    print(f"図: {FIG_PATH}")


if __name__ == "__main__":
    main()
