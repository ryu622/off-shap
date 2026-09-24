"""Stage 1 の Shapley 値をシーン × 選手の表に集計し、重み・集計方法の比較を行う。

入力: data/processed/stage1/*.npz(scripts/compute_stage1.py)
出力:
- data/processed/stage1_player_values.parquet
- documents/images/stage1_*.png

実行: uv run python scripts/analyze_stage1.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from off_shap.loader import PROJECT_ROOT, cache_paths
from off_shap.scene import Scene

SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
RESULT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "stage1_player_values.parquet"
IMG_DIR = PROJECT_ROOT / "documents" / "images"

TOUCH_RADIUS_M = 1.5  # ボールとの距離がこれ未満になったフレームがあれば「ボールに関与」とみなす

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _positions(match_id: str) -> dict[str, str]:
    meta = json.loads(cache_paths(match_id)["meta"].read_text())
    return {p["player_id"]: p["position"] for p in meta["players"]}


def build_table(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positions: dict[str, dict[str, str]] = {}
    for _, r in index.iterrows():
        path = RESULT_DIR / f"{r.scene_id}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        scene = Scene.load(SCENE_DIR / f"{r.scene_id}.npz")
        pos_of = positions.setdefault(r.match_id, _positions(r.match_id))
        phi, weights = z["phi"], [str(w) for w in z["weight_names"]]  # phi (F, n, K)
        dist_ball = np.linalg.norm(scene.attack_pos - scene.ball_pos[:, None], axis=-1)
        t_end = scene.n_frames - 1
        for i, pid in enumerate(scene.attack_ids):
            row = {
                "scene_id": r.scene_id,
                "match_id": r.match_id,
                "outcome": r.outcome,
                "player_id": str(pid),
                "position": pos_of.get(str(pid), "Unknown"),
                "touched_ball": bool((dist_ball[:, i] < TOUCH_RADIUS_M).any()),
                "x_end": float(scene.attack_pos[t_end, i, 0]),
                "x_minus_ball_end": float(scene.attack_pos[t_end, i, 0] - scene.ball_pos[t_end, 0]),
                "dist_ball_end": float(dist_ball[t_end, i]),
            }
            for k, w in enumerate(weights):
                row[f"phi_{w}_terminal"] = float(phi[-1, i, k])
                row[f"phi_{w}_mean"] = float(phi[:, i, k].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2)


def main() -> None:
    index = pd.read_parquet(INDEX_PATH)
    index = index[~index.tracking_glitch]
    df = build_table(index)
    df.to_parquet(OUT_PATH, index=False)
    weights = ["xt", "xt_opp_half", "xt_gain"]
    cols = [f"phi_{w}_{a}" for w in weights for a in ("terminal", "mean")]
    print(f"{df.scene_id.nunique()} シーン / {len(df)} 選手-シーン")

    # 1. シーン内の順位の一致度(Spearman、シーンごとに計算して中央値)
    print("\n[1] シーン内順位の一致度(Spearman ρ の中央値)")
    rho = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            rho.loc[a, b] = (
                df.groupby("scene_id")
                .apply(lambda g, a=a, b=b: spearmanr(g[a], g[b]).statistic)
                .median()
            )
    print(rho.round(2).to_string())

    # 2. 自陣に残る選手への配分: φ と終端での選手の x の関係
    print("\n[2] 終端で自陣(x < 52.5m)にいる選手が受け取る φ の割合(シーン平均)")
    for c in cols:
        share = df.groupby("scene_id").apply(
            lambda g, c=c: g.loc[g.x_end < 52.5, c].sum() / g[c].sum() if g[c].sum() > 0 else np.nan
        )
        print(f"  {c:28s} {share.mean():.2f}")

    # 3. オフボール選手への配分
    print("\n[3] ボールに関与しなかった選手(オフボール)が受け取る φ の割合(シーン平均)")
    print(f"  オフボール選手の人数割合: {1 - df.touched_ball.mean():.2f}")
    for c in cols:
        share = df.groupby("scene_id").apply(
            lambda g, c=c: (
                g.loc[~g.touched_ball, c].sum() / g[c].sum() if g[c].sum() > 0 else np.nan
            )
        )
        top_off = df.loc[df.groupby("scene_id")[c].idxmax()]
        print(
            f"  {c:28s} 割合 {share.mean():.2f} / シーン内1位がオフボール {(~top_off.touched_ball).mean():.2f}"
        )

    # 図: 終端での選手の x と φ(平均集計)の関係を重み別に
    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=SURFACE, sharey=False)
    for ax, w, color in zip(axes, weights, SERIES):
        c = f"phi_{w}_mean"
        scene_total = df.groupby("scene_id")[c].transform("sum")
        share = df[c] / scene_total
        ax.scatter(df.x_end, share, s=6, alpha=0.35, color=color, lw=0)
        ax.axvline(52.5, color=TEXT_2, lw=0.8, ls="--")
        ax.set_title(f"重み {w}(平均集計)", loc="left", color=TEXT)
        ax.set_xlabel("終端での選手の x [m](→ 相手ゴール)", color=TEXT)
        _style(ax)
    axes[0].set_ylabel("シーン内での φ の取り分", color=TEXT)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "stage1_phi_vs_x.png", dpi=150, facecolor=SURFACE)
    print(f"\n保存: {OUT_PATH}\n図: {IMG_DIR / 'stage1_phi_vs_x.png'}")


if __name__ == "__main__":
    main()
