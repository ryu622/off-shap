"""idsse-data 全7試合を kloppy から読み込み、data/cache/ に parquet でキャッシュする。

実行: uv run python scripts/build_cache.py [--overwrite]
"""

from __future__ import annotations

import argparse
import time

from off_shap.loader import MATCH_IDS, build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for match_id in MATCH_IDS:
        t0 = time.time()
        build_cache(match_id, overwrite=args.overwrite)
        print(f"{match_id}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
