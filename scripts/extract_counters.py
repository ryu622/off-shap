"""全試合からカウンター候補を抽出し、分布の要約と図を出力する。

出力:
- data/processed/counter_candidates.parquet  全候補(特徴量付き)+ is_counter 列
- documents/images/counter_candidates_dist.png

実行: uv run python scripts/extract_counters.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from off_shap.counter import CounterCriteria, extract_candidates, is_counter
from off_shap.loader import MATCH_IDS, PROJECT_ROOT, load_match

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "counter_candidates.parquet"
FIG_PATH = PROJECT_ROOT / "documents" / "images" / "counter_candidates_dist.png"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
SERIES = ["#2a78d6", "#eb6834"]  # 1: 全候補, 2: カウンター判定


def plot_distributions(df: pd.DataFrame, criteria: CounterCriteria, path: Path) -> None:
    plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor=SURFACE)
    counter = df[df.is_counter]
    panels = [
        ("start_x", "奪取位置 x [m](攻撃方向基準)", np.arange(0, 106, 5), criteria.max_start_x),
        ("duration_s", "継続時間 [s]", np.arange(0, 20.5, 1), criteria.min_duration_s),
        (
            "progress_10s",
            "10秒以内のボール前進 [m]",
            np.arange(-20, 101, 5),
            criteria.min_progress_m,
        ),
        ("max_x", "到達最大 x [m]", np.arange(0, 106, 5), None),
    ]
    for ax, (col, label, bins, thr) in zip(axes.flat, panels):
        ax.hist(df[col], bins=bins, color=SERIES[0], alpha=0.35, label="全候補", rwidth=0.9)
        ax.hist(counter[col], bins=bins, color=SERIES[1], label="カウンター判定", rwidth=0.9)
        if thr is not None:
            ax.axvline(thr, color=TEXT_2, lw=1, ls="--")
        ax.set_xlabel(label, color=TEXT)
    for ax, (subset, title) in zip(axes.flat[4:], [(df, "全候補"), (counter, "カウンター判定")]):
        order = ["lost", "dead", "shot", "goal", "timeout", "period_end"]
        counts = subset.outcome.value_counts().reindex(order, fill_value=0)
        ax.barh(order[::-1], counts.values[::-1], color=SERIES[0] if subset is df else SERIES[1])
        for i, v in enumerate(counts.values[::-1]):
            ax.text(v, i, f" {v}", va="center", color=TEXT_2)
        ax.set_title(f"結果内訳({title}, n={len(subset)})", color=TEXT, loc="left")
    axes.flat[0].legend(frameon=False)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT_2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    criteria = CounterCriteria()
    df = pd.concat([extract_candidates(load_match(m)) for m in MATCH_IDS], ignore_index=True)
    df["is_counter"] = is_counter(df, criteria)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    c = df[df.is_counter]
    print(f"候補: {len(df)}  カウンター判定: {len(c)}  ({criteria})")
    print("\n試合別:")
    print(
        df.groupby("match_id").agg(
            candidates=("is_counter", "size"), counters=("is_counter", "sum")
        )
    )
    print("\n結果内訳(カウンター判定):")
    print(c.outcome.value_counts())
    print(
        f"\n奪取イベント対応率: 全候補 {df.has_recovery_event.mean():.2f} / カウンター {c.has_recovery_event.mean():.2f}"
    )
    flagged = df[df.dfl_counter_flag]
    print(
        f"DFL CounterAttack=true のシュート: {len(flagged)} 件中 is_counter={flagged.is_counter.sum()}"
    )
    print("\n特徴量の分位点(カウンター判定):")
    print(
        c[["start_x", "duration_s", "progress_5s", "progress_10s", "max_x", "xg"]]
        .describe()
        .round(2)
    )
    plot_distributions(df, criteria, FIG_PATH)
    print(f"\n保存: {OUT_PATH}\n図: {FIG_PATH}")


if __name__ == "__main__":
    main()
