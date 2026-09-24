"""カウンター判定されたシーンを前処理し、Scene として保存する。

入力: data/processed/counter_candidates.parquet(scripts/extract_counters.py)
出力:
- data/processed/scenes/{scene_id}.npz
- data/processed/scenes_index.parquet  シーン一覧(split・人数・品質チェック列付き)

実行: uv run python scripts/build_scenes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from off_shap.loader import PROJECT_ROOT, load_match
from off_shap.scene import SPLITS, build_scene

CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "counter_candidates.parquet"
SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"

MAX_PLAUSIBLE_SPEED = (
    12.0  # m/s。平滑化前にこれを超える選手がいるシーンは座標の飛びとして除外対象にする
)


def main() -> None:
    cands = pd.read_parquet(CANDIDATES_PATH)
    counters = cands[cands.is_counter]
    SCENE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for match_id, group in counters.groupby("match_id"):
        match = load_match(match_id)
        for _, row in group.iterrows():
            scene = build_scene(match, row)
            scene.save(SCENE_DIR / f"{scene.scene_id}.npz")
            rows.append(
                {
                    "scene_id": scene.scene_id,
                    "match_id": match_id,
                    "split": SPLITS[match_id],
                    "n_frames": scene.n_frames,
                    "n_attack": len(scene.attack_ids),
                    "n_defend": len(scene.defend_ids),
                    "has_defend_gk": not np.isnan(scene.defend_gk_pos).all(),
                    "nan_frac": float(np.isnan(scene.attack_pos).mean()),
                    "max_raw_speed": scene.max_raw_speed,
                    "tracking_glitch": scene.max_raw_speed > MAX_PLAUSIBLE_SPEED,
                    "outcome": scene.outcome,
                    "xg": scene.xg,
                }
            )
        print(f"{match_id}: {len(group)} scenes", flush=True)

    index = pd.DataFrame(rows)
    index.to_parquet(INDEX_PATH, index=False)

    print(f"\n合計 {len(index)} シーン → {SCENE_DIR}")
    print("\nsplit 別:")
    print(index.groupby("split").agg(scenes=("scene_id", "size"), matches=("match_id", "nunique")))
    print("\n攻撃側フィールドプレイヤー数:", index.n_attack.value_counts().to_dict())
    print("守備側フィールドプレイヤー数:", index.n_defend.value_counts().to_dict())
    print("守備GKありの割合:", index.has_defend_gk.mean())
    print("欠損率 最大:", index.nan_frac.max())
    q = index.max_raw_speed.quantile([0.5, 0.95, 0.99]).round(2).tolist()
    print(f"平滑化前の最大速度 分位点(50/95/99%): {q}")
    print(
        f"座標の飛び({MAX_PLAUSIBLE_SPEED} m/s 超)で除外対象: {index.tracking_glitch.sum()} シーン"
    )
    print("除外後の split 別:", index[~index.tracking_glitch].groupby("split").size().to_dict())


if __name__ == "__main__":
    main()
