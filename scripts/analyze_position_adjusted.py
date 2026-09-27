"""位置で調整した φ: 「どれだけ前にいたか」では説明できない貢献を取り出す。

φ は選手の平均位置(ゴール方向の x)でほぼ決まる(選手単位で分散の約8割)。
そこで、平均 x から見込まれる φ を直線で予測し、実際の φ との差(残差)で選手を並べる。

    予測 φ = a + b · 平均 x          (118人の選手単位の値に最小二乗で当てはめる)
    残差   = 実際の φ − 予測 φ

残差がプラスの選手は、同じくらい前にいた選手より危険なスペースを押さえている。
残差が何と関係するかを見るため、サイド寄りの度合い(ピッチ中央線からの横方向の距離 |y − 34|)と
ゴール方向の速度との順位相関も出す。

- φ: 1シーンあたりの φ × 1000(主指標 `xt_opp_half`、シーン平均集計)
- 平均 x: φ を計算したフレーム(0.5 秒ごと + 終端)での x の平均を、シーンごとに出してから
  選手ごとに平均したもの(攻撃方向基準、m)
- 頑健性の確認: 選手-シーン単位で、同じ 5m 幅の x にいた選手-シーンの φ の平均を予測値とし、
  その残差を選手ごとに平均したもの。主の方法との順位相関を出す

入力: data/processed/stage1/*.npz、data/processed/scenes/*.npz、
      data/processed/offball_vs_onball.parquet(scripts/plot_offball_vs_onball.py の出力)
出力: data/processed/position_adjusted_phi.parquet(選手単位)、
      documents/images/position_adjusted_phi.png

実行: uv run python scripts/analyze_position_adjusted.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from off_shap.loader import PROJECT_ROOT
from off_shap.scene import Scene
from off_shap.worth import PRIMARY_WEIGHT

SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
RESULT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"
PLAYERS_PATH = PROJECT_ROOT / "data" / "processed" / "offball_vs_onball.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "position_adjusted_phi.parquet"
IMG_PATH = PROJECT_ROOT / "documents" / "images" / "position_adjusted_phi.png"

PITCH_WIDTH = 68.0
BIN_M = 5.0  # 頑健性の確認で使う x の幅
N_TOP, N_BOTTOM = 10, 5  # 右の図に出す人数

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
# plot_offball_vs_onball.py と同じ色
GROUP_COLOR = {
    "守備的MF": "#2a78d6",
    "FW・ウイング": "#eb6834",
    "SB": "#1baf7a",
    "攻撃的・サイドMF": "#eda100",
    "CB": "#e87ba4",
    "不明(交代出場など)": "#a3a29d",
}


def player_scene_table(scene_ids: pd.Series) -> pd.DataFrame:
    """選手-シーンごとの φ × 1000 と、φ を計算したフレームでの平均 x・横位置・前向きの速度。"""
    rows = []
    for sid in scene_ids:
        z = np.load(RESULT_DIR / f"{sid}.npz")
        k = [str(w) for w in z["weight_names"]].index(PRIMARY_WEIGHT)
        scene = Scene.load(SCENE_DIR / f"{sid}.npz")
        pos = scene.attack_pos[z["frames"]]
        x = pos[..., 0].mean(axis=0)
        wide = np.abs(pos[..., 1] - PITCH_WIDTH / 2).mean(axis=0)
        vx = scene.attack_vel[z["frames"], :, 0].mean(axis=0)
        phi = z["phi"][:, :, k].mean(axis=0) * 1e3
        for i, pid in enumerate(z["attack_ids"]):
            rows.append(
                {
                    "scene_id": sid,
                    "player_id": str(pid),
                    "x": x[i],
                    "wide": wide[i],
                    "vx": vx[i],
                    "phi": phi[i],
                }
            )
    return pd.DataFrame(rows)


def binned_residual(ps: pd.DataFrame) -> pd.Series:
    """選手-シーン単位で、同じ x の幅にいた選手-シーンの平均 φ との差。"""
    bins = np.floor(ps.x / BIN_M)
    return ps.phi - ps.groupby(bins).phi.transform("mean")


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
    ps = player_scene_table(index.scene_id[~index.tracking_glitch])
    ps["res_binned"] = binned_residual(ps)

    players = pd.read_parquet(PLAYERS_PATH)[["player_id", "name", "team", "position", "group"]]
    agg = (
        ps.groupby("player_id")
        .agg(
            scenes=("scene_id", "size"),
            x=("x", "mean"),
            wide=("wide", "mean"),
            vx=("vx", "mean"),
            phi=("phi", "mean"),
            res_binned=("res_binned", "mean"),
        )
        .reset_index()
        .merge(players, on="player_id")  # 8シーン以上に出場した選手だけが残る
    )
    b, a = np.polyfit(agg.x, agg.phi, 1)
    agg["pred"] = a + b * agg.x
    agg["residual"] = agg.phi - agg.pred
    r2 = 1 - (agg.residual**2).sum() / ((agg.phi - agg.phi.mean()) ** 2).sum()
    agg = agg.sort_values("residual", ascending=False).reset_index(drop=True)
    agg["rank_phi"] = agg.phi.rank(ascending=False).astype(int)
    agg["rank_residual"] = np.arange(1, len(agg) + 1)
    agg.to_parquet(OUT_PATH, index=False)

    print(f"{len(agg)} 人。予測 φ = {a:.3f} + {b:.4f} × 平均 x(R² = {r2:.2f})")
    rho = spearmanr(agg.residual, agg.res_binned).statistic
    print(f"頑健性: 選手-シーン単位で {BIN_M:.0f}m ごとに調整した残差との順位相関 ρ = {rho:.2f}")
    rho_phi = spearmanr(agg.residual, agg.phi).statistic
    print(f"残差と φ そのものの順位相関 ρ = {rho_phi:.2f}")
    for c, label in (("wide", "サイド寄りの度合い |y − 34|"), ("vx", "ゴール方向の速度")):
        print(f"残差と{label}の順位相関 ρ = {spearmanr(agg.residual, agg[c]).statistic:.2f}")
    cols = [
        "name",
        "team",
        "position",
        "scenes",
        "x",
        "wide",
        "vx",
        "phi",
        "pred",
        "residual",
        "rank_phi",
    ]
    print("\n残差が大きい(位置の割に φ が高い):")
    print(agg[cols].head(N_TOP).round(3).to_string(index=False))
    print("\n残差が小さい(位置の割に φ が低い):")
    print(agg[cols].tail(N_BOTTOM).round(3).to_string(index=False))
    print("\nポジション別の残差の平均:")
    print(
        agg.groupby("group")
        .agg(n=("residual", "size"), residual=("residual", "mean"), x=("x", "mean"))
        .round(3)
        .to_string()
    )

    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # 左: φ と平均 x。直線からの縦の差が残差
    ax = axes[0]
    size = 12 + agg.scenes * 1.6
    for group, color in GROUP_COLOR.items():
        g = agg[agg.group == group]
        ax.scatter(g.x, g.phi, s=size[g.index], color=color, alpha=0.8, lw=0.8, ec=SURFACE)
    xs = np.linspace(agg.x.min() - 2, agg.x.max() + 2, 50)
    ax.plot(xs, a + b * xs, color=TEXT_2, lw=1, ls="--")
    # 名前は右に置く。すぐ右に名前付きの点があるときだけ左に置き、重なりを避ける
    labels = pd.concat([agg.head(5), agg.tail(3)])
    for _, r in labels.iterrows():
        near = labels[(labels.x > r.x) & (labels.x - r.x < 8) & ((labels.phi - r.phi).abs() < 0.05)]
        left = not near.empty
        ax.annotate(
            r["name"],
            (r.x, r.phi),
            fontsize=7.5,
            color=TEXT,
            xytext=(-6 if left else 6, 3),
            textcoords="offset points",
            ha="right" if left else "left",
        )
    ax.set_xlabel("平均 x(ゴール方向の位置、m)", color=TEXT)
    ax.set_ylabel("1シーンあたりの φ × 1000", color=TEXT)
    ax.set_title(f"φ はほぼ「どれだけ前にいたか」で決まる(R² = {r2:.2f})", loc="left", color=TEXT)
    _style(ax)

    # 右: 残差の上位と下位
    ax = axes[1]
    show = pd.concat([agg.head(N_TOP), agg.tail(N_BOTTOM)]).iloc[::-1]
    ypos = np.arange(len(show))
    ypos[N_BOTTOM:] += 1  # 上位と下位の間を1行空ける
    ax.barh(ypos, show.residual, color=show.group.map(GROUP_COLOR), height=0.7)
    ax.set_yticks(ypos, [f"{n}({t})" for n, t in zip(show.name, show.team)], fontsize=7.5)
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_xlabel("残差 = 実際の φ − 位置から予測した φ(× 1000)", color=TEXT)
    ax.set_title(f"位置の割に φ が高い {N_TOP} 人・低い {N_BOTTOM} 人", loc="left", color=TEXT)
    _style(ax)
    ax.grid(axis="y", visible=False)

    handles = [
        Line2D([], [], ls="", marker="o", ms=7, color=c, label=g) for g, c in GROUP_COLOR.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.text(
        0.5,
        0.005,
        f"{len(agg)} 人(8 シーン以上)。左の点の大きさ = 出場シーン数、破線 = 平均 x による予測",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(IMG_PATH, dpi=150, facecolor=SURFACE)
    print(f"\n保存: {OUT_PATH}\n図: {IMG_PATH}")


if __name__ == "__main__":
    main()
