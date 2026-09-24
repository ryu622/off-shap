"""主指標の Shapley 値(全員が対象)と、ボール関与選手だけを対象にしたベースラインを比較する。

ベースライン: 先行研究(A Model-Based Restricted Shapley Value)と同様に連合を
ボール関与選手に限定した条件。先行研究の手法の適用ではなく、worth 関数は提案手法と
共通にして連合の範囲だけを変える(documents/stage1_worth.md 6.1節)。
ボールに関与した選手だけで連合を作る。関与しなかった選手は常にピッチにいるとみなす
(shapley.restricted_shapley)。オフボール選手の値は定義上 0 になる。

出力:
- data/processed/stage1_baseline_comparison.parquet(選手-シーン単位)
- documents/images/stage1_baseline_*.png

実行: uv run python scripts/compare_baseline.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from off_shap.loader import PROJECT_ROOT, cache_paths
from off_shap.scene import Scene
from off_shap.shapley import restricted_shapley
from off_shap.worth import PRIMARY_WEIGHT

SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
RESULT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "stage1_baseline_comparison.parquet"
IMG_DIR = PROJECT_ROOT / "documents" / "images"

TOUCH_RADIUS_M = 1.5
MIN_SCENES = 8  # 選手単位の集計に含める最小シーン数

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
ON_BALL, OFF_BALL = "#2a78d6", "#eb6834"


def _players(match_id: str) -> dict[str, dict]:
    meta = json.loads(cache_paths(match_id)["meta"].read_text())
    names = meta["team_names"]
    return {
        p["player_id"]: {"name": p["name"], "position": p["position"], "team": names[p["team_id"]]}
        for p in meta["players"]
    }


def build_table(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    info: dict[str, dict] = {}
    for _, r in index.iterrows():
        z = np.load(RESULT_DIR / f"{r.scene_id}.npz")
        k = [str(w) for w in z["weight_names"]].index(PRIMARY_WEIGHT)
        values, phi = z["values"][:, :, k].astype(float), z["phi"][:, :, k]
        scene = Scene.load(SCENE_DIR / f"{r.scene_id}.npz")
        dist = np.linalg.norm(scene.attack_pos - scene.ball_pos[:, None], axis=-1)
        touched = (dist < TOUCH_RADIUS_M).any(axis=0)
        phi_full = phi.mean(axis=0)
        phi_base = np.mean([restricted_shapley(v, touched) for v in values], axis=0)
        players = info.setdefault(r.match_id, _players(r.match_id))
        for i, pid in enumerate(scene.attack_ids):
            p = players.get(str(pid), {"name": str(pid), "position": "Unknown", "team": "?"})
            rows.append(
                {
                    "scene_id": r.scene_id,
                    "match_id": r.match_id,
                    "outcome": r.outcome,
                    "player_id": str(pid),
                    "name": p["name"],
                    "team": p["team"],
                    "position": p["position"],
                    "touched_ball": bool(touched[i]),
                    "n_touched_in_scene": int(touched.sum()),
                    "phi_full": float(phi_full[i]),
                    "phi_base": float(phi_base[i]),
                }
            )
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
    df = build_table(index[~index.tracking_glitch])
    df.to_parquet(OUT_PATH, index=False)
    scale = 1e3  # 表示用(φ × 1000)

    print(f"{df.scene_id.nunique()} シーン / {len(df)} 選手-シーン")
    print("チーム別の選手-シーン数:", df.team.value_counts().to_dict())
    per_scene = df.groupby("scene_id").agg(
        n_touched=("n_touched_in_scene", "first"),
        total=("phi_full", "sum"),
        off=("phi_full", lambda s: s[~df.loc[s.index, "touched_ball"]].sum()),
    )
    per_scene = per_scene[per_scene.total > 0]
    print(
        f"ボール関与選手の人数(シーンあたり): {per_scene.n_touched.describe()[['mean', '50%', 'min', 'max']].round(1).to_dict()}"
    )
    print(f"関与選手がいないシーン: {(per_scene.n_touched == 0).sum()}")

    # 1. オフボール選手の取り分 = ボール関与だけの評価では見えない価値
    off_share = per_scene.off / per_scene.total
    print("\n[1] シーン価値のうちオフボール選手に配分される割合")
    print(
        f"  平均 {off_share.mean():.2f} / 中央値 {off_share.median():.2f} / 半分以上のシーン {(off_share >= 0.5).mean():.2f}"
    )

    # 2. ボール関与選手の評価の上乗せ
    on = df[df.touched_ball & (df.phi_full > 0)]
    ratio = on.phi_base / on.phi_full
    print("\n[2] ボール関与選手: ベースラインの φ / 全員対象の φ")
    print(
        f"  中央値 {ratio.median():.2f}(四分位 {ratio.quantile(0.25):.2f}〜{ratio.quantile(0.75):.2f})"
    )

    # 3. 選手単位の合計
    # ポジションは試合ごとに異なることがあるので、選手 ID で集計し最頻のものを使う
    agg = (
        df.groupby(["player_id", "name", "team"])
        .agg(
            position=("position", lambda s: s.mode().iloc[0]),
            scenes=("scene_id", "size"),
            touched_rate=("touched_ball", "mean"),
            phi_full=("phi_full", "sum"),
            phi_base=("phi_base", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg.scenes >= MIN_SCENES].copy()
    agg["rank_full"] = agg.phi_full.rank(ascending=False).astype(int)
    agg["rank_base"] = agg.phi_base.rank(ascending=False).astype(int)
    agg["rank_change"] = agg.rank_base - agg.rank_full  # 正 = 全員対象で順位が上がる
    rho = spearmanr(agg.phi_full, agg.phi_base).statistic
    print(
        f"\n[3] 選手単位の合計({len(agg)} 人、{MIN_SCENES} シーン以上に出場): 順位相関 ρ = {rho:.2f}"
    )
    cols = [
        "name",
        "team",
        "position",
        "scenes",
        "touched_rate",
        "phi_full",
        "phi_base",
        "rank_full",
        "rank_base",
        "rank_change",
    ]
    show = agg[cols].assign(phi_full=agg.phi_full * scale, phi_base=agg.phi_base * scale)
    print("\n  全員対象で順位が大きく上がる選手(ベースラインで過小評価):")
    print(show.sort_values("rank_change", ascending=False).head(8).round(2).to_string(index=False))
    print("\n  全員対象で順位が大きく下がる選手(ベースラインで過大評価):")
    print(show.sort_values("rank_change").head(8).round(2).to_string(index=False))

    # 4. ポジション別
    print("\n[4] ポジション別(選手-シーンあたりの平均 φ × 1000)")
    pos = df.groupby("position").agg(
        n=("scene_id", "size"),
        touched_rate=("touched_ball", "mean"),
        phi_full=("phi_full", "mean"),
        phi_base=("phi_base", "mean"),
    )
    pos[["phi_full", "phi_base"]] *= scale
    print(pos[pos.n >= 50].sort_values("phi_full", ascending=False).round(2).to_string())

    # 図: 選手単位の合計 φ(全員対象 vs ベースライン)
    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    ax = axes[0]
    ax.hist(off_share, bins=np.linspace(0, 1, 21), color=OFF_BALL, rwidth=0.9)
    ax.axvline(off_share.mean(), color=TEXT_2, lw=1, ls="--")
    ax.set_xlabel("シーン価値のうちオフボール選手に配分される割合", color=TEXT)
    ax.set_ylabel("シーン数", color=TEXT)
    ax.set_title("ボール関与だけの評価では見えない価値", loc="left", color=TEXT)
    _style(ax)
    ax = axes[1]
    ax.scatter(agg.phi_base * scale, agg.phi_full * scale, s=18, color=ON_BALL, alpha=0.7, lw=0)
    lim = max(agg.phi_base.max(), agg.phi_full.max()) * scale * 1.05
    ax.plot([0, lim], [0, lim], color=TEXT_2, lw=0.8, ls="--")
    for _, r in agg.sort_values("rank_change", ascending=False).head(4).iterrows():
        ax.annotate(
            r["name"],
            (r.phi_base * scale, r.phi_full * scale),
            fontsize=7,
            color=TEXT_2,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("ベースライン(ボール関与選手のみ)の合計 φ × 1000", color=TEXT)
    ax.set_ylabel("全員対象の合計 φ × 1000", color=TEXT)
    ax.set_title(f"選手単位の合計(ρ = {rho:.2f})", loc="left", color=TEXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "stage1_baseline_comparison.png", dpi=150, facecolor=SURFACE)
    print(f"\n保存: {OUT_PATH}\n図: {IMG_DIR / 'stage1_baseline_comparison.png'}")


if __name__ == "__main__":
    main()
