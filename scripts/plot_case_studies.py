"""オフボール選手の φ が高いシーンの事例図。

各シーンについて
(左) シーン全体の軌道。各攻撃側選手に φ(× 1000、主指標)とボール関与の有無を表示
(右) 0.5 秒ごとの φ の推移(φ 上位3人とボール関与選手)
を描く。

実行: uv run python scripts/plot_case_studies.py [scene_id ...]
      引数なしなら自動で事例を選ぶ(下の select_cases)
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from off_shap.loader import PROJECT_ROOT
from off_shap.scene import Scene
from off_shap.worth import PRIMARY_WEIGHT

SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
RESULT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1"
TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "stage1_baseline_comparison.parquet"
IMG_DIR = PROJECT_ROOT / "documents" / "images"

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
ON_BALL, OFF_BALL, DEFEND, OTHER = "#2a78d6", "#eb6834", "#b8b6ae", "#8a8983"


def select_cases(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(scene_id, 注目選手ID) の一覧。"""
    cases = []
    for name in ["Serge Gnabry", "M. Saliakas"]:
        rows = df[(df.name == name) & ~df.touched_ball]
        best = rows.loc[rows.phi_full.idxmax()]
        cases.append((best.scene_id, best.player_id))
    shots = df[df.outcome.isin(["shot", "goal"])]
    top = shots.loc[shots.groupby("scene_id").phi_full.idxmax()]
    top = top[~top.touched_ball].sort_values("phi_full", ascending=False)
    for _, r in top.iterrows():
        if r.scene_id not in {c[0] for c in cases}:
            cases.append((r.scene_id, r.player_id))
            break
    return cases


def plot_case(df: pd.DataFrame, scene_id: str, focus_id: str | None, path) -> None:
    scene = Scene.load(SCENE_DIR / f"{scene_id}.npz")
    z = np.load(RESULT_DIR / f"{scene_id}.npz")
    k = [str(w) for w in z["weight_names"]].index(PRIMARY_WEIGHT)
    phi_t, frames = z["phi"][:, :, k] * 1e3, z["frames"]
    rows = df[df.scene_id == scene_id].set_index("player_id")
    ids = [str(p) for p in scene.attack_ids]
    touched = rows.loc[ids, "touched_ball"].to_numpy()
    phi = rows.loc[ids, "phi_full"].to_numpy() * 1e3
    names = rows.loc[ids, "name"].to_numpy()
    focus = ids.index(focus_id) if focus_id in ids else int(np.argmax(phi))

    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]}
    )
    pitch = Pitch(
        pitch_type="custom", pitch_length=105, pitch_width=68, line_color=TEXT_2, linewidth=0.8
    )
    pitch.draw(ax=ax)
    for j in range(len(scene.defend_ids)):
        ax.plot(*scene.defend_pos[:, j].T, color=DEFEND, lw=1, alpha=0.8)
        ax.scatter(*scene.defend_pos[-1, j], color=DEFEND, s=30, zorder=3)
    ax.plot(*scene.ball_pos.T, color=TEXT, lw=1, ls=(0, (2, 2)), zorder=4)
    ax.scatter(*scene.ball_pos[0], color=TEXT, s=15, marker="x", zorder=5)
    for i in range(len(ids)):
        color = OFF_BALL if i == focus else (ON_BALL if touched[i] else OTHER)
        lw = 2.2 if i == focus else 1.2
        ax.plot(*scene.attack_pos[:, i].T, color=color, lw=lw, zorder=3)
        ax.scatter(*scene.attack_pos[0, i], facecolors="none", edgecolors=color, s=25, zorder=3)
        ax.scatter(*scene.attack_pos[-1, i], color=color, s=45, edgecolors="white", zorder=4)
        label = f"{names[i]} {phi[i]:.2f}" + (" (関与)" if touched[i] else "")
        ax.annotate(
            label,
            scene.attack_pos[-1, i],
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
            color=TEXT if i == focus else TEXT_2,
            zorder=6,
        )
    ax.set_title(
        f"{scene_id}  結果: {scene.outcome}  {scene.n_frames / scene.fps:.1f}秒"
        "(→ 右へ攻撃、○=開始 ●=終端、点線=ボール、名前の後の数字=φ×1000、(関与)=ボール関与)",
        loc="left",
        fontsize=8,
        color=TEXT,
    )

    t = frames / scene.fps
    order = np.argsort(-phi)
    show = sorted(set(order[:3]) | set(np.flatnonzero(touched)) | {focus})
    for i in show:
        color = OFF_BALL if i == focus else (ON_BALL if touched[i] else OTHER)
        ax2.plot(t, phi_t[:, i], color=color, lw=2 if i == focus else 1.2, marker="o", ms=3)
        ax2.annotate(
            names[i],
            (t[-1], phi_t[-1, i]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=TEXT_2,
            va="center",
        )
    ax2.set_xlabel("シーン開始からの時間 [s]", color=TEXT)
    ax2.set_ylabel("φ × 1000(各時点)", color=TEXT)
    ax2.set_title("φ の推移(橙=注目選手、青=ボール関与、灰=その他の上位)", loc="left", color=TEXT)
    ax2.set_facecolor(SURFACE)
    ax2.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(GRID)
    ax2.tick_params(colors=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(TABLE_PATH)
    cases = [(s, None) for s in sys.argv[1:]] or select_cases(df)
    for n, (scene_id, focus) in enumerate(cases, 1):
        path = IMG_DIR / f"case_{n}.png"
        plot_case(df, scene_id, focus, path)
        r = df[(df.scene_id == scene_id)].sort_values("phi_full", ascending=False)
        print(f"case_{n}: {scene_id} 注目={focus}")
        print(
            r[["name", "position", "touched_ball", "phi_full", "phi_base"]]
            .assign(phi_full=r.phi_full * 1e3, phi_base=r.phi_base * 1e3)
            .round(2)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
