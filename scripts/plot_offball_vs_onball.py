"""提案手法の貢献度(φ)と、オンボールの実績の散布図。

「ボールにあまり関与しないのに、危険なスペースを押さえて攻撃に貢献している選手」を
右下(φ が高く、オンボールの実績は目立たない)に探す。

- 横軸: 1シーンあたりの φ(主指標 `xt_opp_half`、シーン平均集計)× 1000
- 縦軸(メイン): 1シーンあたりの xGChain
    = (その選手がボールに触ったシーンで打たれたシュートの xG の合計) ÷ 出場シーン数
  シーンの xG は scenes_index の `xg`(シーン内のシュートの xG の最大値。シュートがなければ 0。
  DFL 公式の xG で、終端後2秒の猶予を含めて対応付けたもの)
- 縦軸(サブ): ボールに触ってシュート/ゴールで終わったシーンの数 ÷ 出場シーン数
- 「ボールに触った」はベースラインと同じ(ボールとの距離 1.5 m 未満)
- 縦軸(xT 版): 1シーンあたりの xT 増加量(パスとドリブルで xT をどれだけ上げたか)
    パス: 成功したパスの xT(受けた位置) − xT(出した位置)。失敗したパスは数えない(Singh の定義)
    ドリブル: DFL のイベントにドリブルがないので、ボールを持ってから次のアクション(パス・
    シュート)までの移動を「運んだ」とみなし、xT(アクションの位置) − xT(持った位置) とする。
    持った位置は、直前の成功パスの受け手なら受けた位置、シーン最初のアクションなら
    シーン開始時(奪った瞬間)のボール位置。最後のアクションの後の移動は終点がないので数えない
    シーンとの対応: 同じピリオド・攻撃側のチームのパス・シュートで、時刻がシーンの前後
    EVENT_MARGIN_S 秒以内のもの(イベント時刻のずれを吸収)

注意: Stage 1 の φ は選手自身が押さえるスペースの分だけで、デコイの引きつけ効果は含まない
(documents/stage1_worth.md 7.6節)。

入力: data/processed/stage1_baseline_comparison.parquet(scripts/compare_baseline.py の出力)
出力: data/processed/offball_vs_onball.parquet(選手単位)、
      documents/images/offball_vs_onball.png(xGChain 版)、documents/images/offball_vs_onball_xt.png(xT 版)

実行: uv run python scripts/plot_offball_vs_onball.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from off_shap.counter import to_attack_frame
from off_shap.loader import PROJECT_ROOT, MatchData, load_match
from off_shap.scene import Scene, scene_id_of
from off_shap.xt import load_xt

COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "stage1_baseline_comparison.parquet"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "counter_candidates.parquet"
XT_PATH = PROJECT_ROOT / "data" / "processed" / "xt_grid.npz"
SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "offball_vs_onball.parquet"
IMG_PATH = PROJECT_ROOT / "documents" / "images" / "offball_vs_onball.png"
IMG_XT_PATH = PROJECT_ROOT / "documents" / "images" / "offball_vs_onball_xt.png"

MIN_SCENES = 8  # compare_baseline.py と同じ
N_LABELS = 6  # 右下の象限で名前を付ける人数
# 右上(オンボール・オフボールとも高い)の代表として名前を付ける選手
TOP_NAMES = ("T. Asano", "Kingsley Coman", "D. Ginczek", "E. Amenyido")
EVENT_MARGIN_S = 1.0  # シーンの前後に含めるパスの時間幅

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

# ポジションのまとめ方と色(dataviz の参照パレットの順。色覚多様性の検証済み)
POSITION_GROUP = {
    "Striker": "FW・ウイング",
    "Left Forward": "FW・ウイング",
    "Right Forward": "FW・ウイング",
    "Left Wing": "FW・ウイング",
    "Right Wing": "FW・ウイング",
    "Center Attacking Midfield": "攻撃的・サイドMF",
    "Left Midfield": "攻撃的・サイドMF",
    "Right Midfield": "攻撃的・サイドMF",
    "Left Defensive Midfield": "守備的MF",
    "Right Defensive Midfield": "守備的MF",
    "Left Back": "SB",
    "Right Back": "SB",
    "Left Center Back": "CB",
    "Right Center Back": "CB",
}
GROUP_COLOR = {
    "守備的MF": "#2a78d6",
    "FW・ウイング": "#eb6834",
    "SB": "#1baf7a",
    "攻撃的・サイドMF": "#eda100",
    "CB": "#e87ba4",
    "不明(交代出場など)": "#a3a29d",
}


def scene_actions(match: MatchData, scenes: pd.DataFrame) -> pd.DataFrame:
    """1試合分のシーンについて、攻撃側のパス・シュートを攻撃方向基準の座標で取り出す。"""
    events, meta = match.events, match.meta
    actions = events[events.event_type.isin(["PASS", "SHOT"]) & events.x.notna()]
    out = []
    for s in scenes.itertuples():
        t0, t1 = s.start_t - EVENT_MARGIN_S, s.start_t + s.duration_s + EVENT_MARGIN_S
        a = actions[
            (actions.period_id == s.period_id)
            & (actions.team_id == s.team_id)
            & (actions.t >= t0)
            & (actions.t <= t1)
        ].sort_values("t")
        if a.empty:
            continue
        a = a.assign(scene_id=s.scene_id)
        for c, d in (("x", "y"), ("end_x", "end_y")):
            a[c], a[d] = to_attack_frame(
                a[c].to_numpy(), a[d].to_numpy(), s.ground, meta.pitch_length, meta.pitch_width
            )
        out.append(a)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def xt_added(scene_ids: pd.Series) -> pd.DataFrame:
    """選手-シーンごとの xT 増加量(パス + ドリブル)。"""
    grid = load_xt(XT_PATH)
    cand = pd.read_parquet(CANDIDATES_PATH)
    cand = cand[cand.is_counter].copy()
    cand["scene_id"] = cand.apply(scene_id_of, axis=1)
    cand = cand[cand.scene_id.isin(set(scene_ids))]
    a = pd.concat(
        [scene_actions(load_match(m), scenes) for m, scenes in cand.groupby("match_id")],
        ignore_index=True,
    )

    is_pass = a.event_type == "PASS"
    ok = is_pass & (a.result == "COMPLETE") & a.end_x.notna()
    start_xt = grid.at(a[["x", "y"]].to_numpy())
    end_xt = np.full(len(a), np.nan)
    end_xt[ok.to_numpy()] = grid.at(a.loc[ok, ["end_x", "end_y"]].to_numpy())
    a["pass_xt"] = np.where(ok, end_xt - start_xt, 0.0)

    # ドリブル: ボールを持った位置からアクションの位置までの移動
    prev = a.groupby("scene_id").shift(1)
    received = (prev.result == "COMPLETE") & (prev.recipient_id == a.player_id) & prev.end_x.notna()
    first = a.scene_id != a.scene_id.shift(1)
    held = np.full((len(a), 2), np.nan)
    held[received.to_numpy()] = prev.loc[received, ["end_x", "end_y"]].to_numpy()
    start_ball = {
        sid: Scene.load(SCENE_DIR / f"{sid}.npz").ball_pos[0] for sid in a.scene_id[first]
    }
    held[first.to_numpy()] = np.stack([start_ball[sid] for sid in a.scene_id[first]])
    carry = received | first
    held_xt = np.full(len(a), np.nan)
    held_xt[carry.to_numpy()] = grid.at(held[carry.to_numpy()])
    a["carry_xt"] = np.where(carry, start_xt - held_xt, 0.0)
    a["is_pass"] = is_pass.astype(int)

    print(
        f"シーンに対応付いたパス: {is_pass.sum()} 本(成功 {ok.sum() / is_pass.sum():.0%})、"
        f"シュート: {(~is_pass).sum()} 本、アクションが1つ以上あるシーン: "
        f"{a.scene_id.nunique()} / {len(cand)}、ドリブル: {carry.sum()} 回"
        f"(うちシーン最初 {first.sum()})"
    )
    return (
        a.groupby(["scene_id", "player_id"])
        .agg(pass_xt=("pass_xt", "sum"), carry_xt=("carry_xt", "sum"), n_pass=("is_pass", "sum"))
        .reset_index()
    )


def player_table() -> pd.DataFrame:
    df = pd.read_parquet(COMPARISON_PATH)
    index = pd.read_parquet(INDEX_PATH)[["scene_id", "xg"]]
    df = df.merge(index, on="scene_id", how="left")
    df = df.merge(xt_added(df.scene_id), on=["scene_id", "player_id"], how="left")
    df[["pass_xt", "carry_xt", "n_pass"]] = df[["pass_xt", "carry_xt", "n_pass"]].fillna(0)
    df["xt_added"] = df.pass_xt + df.carry_xt
    shot = df.outcome.isin(["shot", "goal"])
    df["xg_chain"] = df.xg.where(df.touched_ball, 0.0)
    df["shot_involved"] = (shot & df.touched_ball).astype(int)
    df["phi_off"] = df.phi_full.where(~df.touched_ball, 0.0)

    # ポジションは試合ごとに異なることがあるので、選手 ID で集計し最頻のものを使う
    agg = (
        df.groupby(["player_id", "name", "team"])
        .agg(
            position=("position", lambda s: s.mode().iloc[0]),
            scenes=("scene_id", "size"),
            touched_rate=("touched_ball", "mean"),
            phi=("phi_full", "mean"),
            phi_off=("phi_off", "mean"),
            xg_chain=("xg_chain", "mean"),
            shot_involved=("shot_involved", "sum"),
            shot_rate=("shot_involved", "mean"),
            xt_added=("xt_added", "mean"),
            pass_xt=("pass_xt", "mean"),
            carry_xt=("carry_xt", "mean"),
            n_pass=("n_pass", "mean"),
        )
        .reset_index()
    )
    agg["group"] = agg.position.map(POSITION_GROUP).fillna("不明(交代出場など)")
    agg["phi"] *= 1e3
    agg["phi_off"] *= 1e3
    return agg[agg.scenes >= MIN_SCENES].reset_index(drop=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2)


def lower_right(agg: pd.DataFrame, y: str) -> pd.DataFrame:
    """横軸が中央値より上、縦軸が中央値以下の選手(φ の大きい順)。"""
    q = agg[(agg.phi > agg.phi.median()) & (agg[y] <= agg[y].median())]
    return q.sort_values("phi", ascending=False)


def left_corner(agg: pd.DataFrame, y: str, upper: bool) -> pd.Series:
    """左上(upper=True)または左下の象限で、角の方向に最も離れた選手。

    横軸・縦軸の単位が違うので、それぞれの範囲で割ってから中央値との差を足す。
    """
    mx, my = agg.phi.median(), agg[y].median()
    rx, ry = np.ptp(agg.phi), np.ptp(agg[y])
    left = agg.phi <= mx
    if upper:
        q = agg[left & (agg[y] > my)]
        score = (mx - q.phi) / rx + (q[y] - my) / ry
    else:
        q = agg[left & (agg[y] <= my)]
        score = (mx - q.phi) / rx + (my - q[y]) / ry
    return q.loc[score.idxmax()]


def scatter(ax, agg: pd.DataFrame, y: str, ylabel: str, title: str) -> None:
    size = 12 + agg.scenes * 1.6
    for group, color in GROUP_COLOR.items():
        g = agg[agg.group == group]
        ax.scatter(g.phi, g[y], s=size[g.index], color=color, alpha=0.8, lw=0.8, ec=SURFACE)
    ax.axvline(agg.phi.median(), color=TEXT_2, lw=0.8, ls="--")
    ax.axhline(agg[y].median(), color=TEXT_2, lw=0.8, ls="--")
    # 右下の点は縦軸 0 付近に固まるので、名前は段をずらして引き出し線で結ぶ
    labels = lower_right(agg, y).head(N_LABELS).sort_values("phi")
    for k, (_, r) in enumerate(labels.iterrows()):
        ax.annotate(
            r["name"],
            (r.phi, r[y]),
            fontsize=7.5,
            color=TEXT,
            xytext=(-30 if k % 2 else 12, 22 + 13 * k),
            textcoords="offset points",
            ha="right" if k % 2 else "left",
            arrowprops={"arrowstyle": "-", "color": TEXT_2, "lw": 0.5},
        )
    # 右上の点はばらけているので、すぐ右横に名前を置く
    for _, r in agg[agg.name.isin(TOP_NAMES)].iterrows():
        ax.annotate(
            r["name"],
            (r.phi, r[y]),
            fontsize=7.5,
            color=TEXT,
            xytext=(6, 3),
            textcoords="offset points",
        )
    # 左上・左下はそれぞれ角の方向に最も離れた1人
    for upper in (True, False):
        r = left_corner(agg, y, upper)
        ax.annotate(
            r["name"],
            (r.phi, r[y]),
            fontsize=7.5,
            color=TEXT,
            xytext=(6, 3) if upper else (14, -26),
            textcoords="offset points",
            arrowprops=None if upper else {"arrowstyle": "-", "color": TEXT_2, "lw": 0.5},
        )
    ax.set_xlabel("1シーンあたりの φ × 1000(提案手法)", color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_title(title, loc="left", color=TEXT)
    _style(ax)


def main() -> None:
    agg = player_table()
    agg.to_parquet(OUT_PATH, index=False)
    print(f"{len(agg)} 人({MIN_SCENES} シーン以上に出場)")
    print(f"xGChain が 0 の選手: {(agg.xg_chain == 0).sum()} 人")
    print(f"xT 増加量が 0 の選手: {(agg.xt_added == 0).sum()} 人")
    for y in ("xg_chain", "shot_rate", "xt_added"):
        rho = spearmanr(agg.phi, agg[y]).statistic
        print(f"順位相関 φ vs {y}: ρ = {rho:.2f}")
    rho_touch = spearmanr(agg.phi, agg.touched_rate).statistic
    print(f"順位相関 φ vs ボール関与率: ρ = {rho_touch:.2f}")

    cols = [
        "name",
        "team",
        "position",
        "scenes",
        "touched_rate",
        "phi",
        "phi_off",
        "xg_chain",
        "shot_involved",
        "xt_added",
    ]
    print("\n右下(φ が中央値より上、xGChain が中央値以下):")
    print(lower_right(agg, "xg_chain")[cols].round(3).to_string(index=False))
    print("\n右上(φ・xGChain とも上位):")
    top = agg[(agg.phi > agg.phi.median()) & (agg.xg_chain > agg.xg_chain.median())]
    print(top.sort_values("phi", ascending=False)[cols].head(10).round(3).to_string(index=False))

    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    scatter(
        axes[0],
        agg,
        "xg_chain",
        "1シーンあたりの xGChain(ボールに触ったシーンの xG)",
        "オフボールの貢献度 × オンボールの実績(xGChain)",
    )
    scatter(
        axes[1],
        agg,
        "shot_rate",
        "ボールに触ってシュートで終わったシーンの割合",
        "縦軸をシュート関与率にした場合",
    )
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
        f"{len(agg)} 人({MIN_SCENES} シーン以上)。点の大きさ = 出場シーン数。破線 = 中央値",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(IMG_PATH, dpi=150, facecolor=SURFACE)

    print("\n[xT 版] 右下(φ が中央値より上、xT 増加量が中央値以下):")
    print(lower_right(agg, "xt_added")[cols].head(12).round(3).to_string(index=False))
    print("\n[xT 版] 右上(φ・xT 増加量とも上位):")
    top = agg[(agg.phi > agg.phi.median()) & (agg.xt_added > agg.xt_added.median())]
    print(top.sort_values("phi", ascending=False)[cols].head(10).round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURFACE)
    scatter(
        ax,
        agg,
        "xt_added",
        "1シーンあたりの xT 増加量(パス + ドリブル)",
        "オフボールの貢献度 × オンボールの実績(xT 増加量)",
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.text(
        0.5,
        0.005,
        f"{len(agg)} 人({MIN_SCENES} シーン以上)。点の大きさ = 出場シーン数。破線 = 中央値",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=TEXT_2,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(IMG_XT_PATH, dpi=150, facecolor=SURFACE)
    print(f"\n保存: {OUT_PATH}\n図: {IMG_PATH}, {IMG_XT_PATH}")


if __name__ == "__main__":
    main()
