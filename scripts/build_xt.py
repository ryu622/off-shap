"""idsse-data 7試合から xT グリッドを学習し、保存・可視化する。

出力:
- data/processed/xt_grid.npz
- documents/images/xt_grid.png(解像度 12×8 と 6×4 の比較)

実行: uv run python scripts/build_xt.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

from off_shap.loader import MATCH_IDS, PROJECT_ROOT, load_match
from off_shap.xt import fit_xt

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "xt_grid.npz"
FIG_PATH = PROJECT_ROOT / "documents" / "images" / "xt_grid.png"
NX, NY = 12, 8  # 採用する解像度(Singh の 16×12 は 7試合では疎になりすぎる)


def main() -> None:
    matches = [load_match(m) for m in MATCH_IDS]
    grids = {(nx, ny): fit_xt(matches, nx, ny) for nx, ny in [(12, 8), (6, 4)]}
    xt = grids[(NX, NY)]
    print(f"passes={xt.n_passes} shots={xt.n_shots} goals={xt.n_goals}")
    np.savez(
        OUT_PATH,
        values=xt.values,
        pitch_length=xt.pitch_length,
        pitch_width=xt.pitch_width,
        n_passes=xt.n_passes,
        n_shots=xt.n_shots,
        n_goals=xt.n_goals,
    )
    print(f"xT {NX}x{NY} (行=攻撃方向 x, 右端=相手ゴール側):")
    print(np.round(xt.values.T, 3))

    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=xt.pitch_length,
        pitch_width=xt.pitch_width,
        line_color="#52514e",
        linewidth=0.8,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    vmax = max(g.values.max() for g in grids.values())
    for ax, ((nx, ny), g) in zip(axes[:2], grids.items()):
        pitch.draw(ax=ax)
        ax.imshow(
            g.values.T,
            extent=(0, g.pitch_length, 0, g.pitch_width),
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            zorder=0,
        )
        ax.set_title(f"xT セル値 {nx}×{ny}", loc="left")
    gx, gy = np.meshgrid(np.linspace(0, xt.pitch_length, 211), np.linspace(0, xt.pitch_width, 137))
    smooth = xt.at(np.stack([gx.ravel(), gy.ravel()], -1)).reshape(gx.shape)
    pitch.draw(ax=axes[2])
    im = axes[2].imshow(
        smooth,
        extent=(0, xt.pitch_length, 0, xt.pitch_width),
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        zorder=0,
    )
    axes[2].set_title(f"採用: {NX}×{NY} を双線形補間(→ 右方向へ攻撃)", loc="left")
    fig.colorbar(im, ax=axes, shrink=0.8, label="xT")
    fig.savefig(FIG_PATH, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    print(f"図: {FIG_PATH}")


if __name__ == "__main__":
    main()
