"""Stage 1(静的削除版)の全連合 v(S) と Shapley 値を、全シーンについて計算する。

各シーンで STRIDE_S 秒ごと(+終端)のフレームについて全 2ⁿ 連合の v(S) を計算し、
フレームごとの Shapley 値とともに保存する。シーン単位の集計は後段で行う。
途中で止めても、保存済みのシーンは飛ばして再開できる。

出力: data/processed/stage1/{scene_id}.npz
  frames (F,), values (F, 2ⁿ, K), phi (F, n, K), attack_ids (n,), weight_names (K,)
  K は worth.WEIGHT_NAMES の重みの種類

実行: uv run python scripts/compute_stage1.py [--workers 6] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from off_shap.loader import PROJECT_ROOT
from off_shap.scene import Scene
from off_shap.shapley import shapley_from_values
from off_shap.worth import WEIGHT_NAMES, Stage1Worth, sample_frames
from off_shap.xt import load_xt

SCENE_DIR = PROJECT_ROOT / "data" / "processed" / "scenes"
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "scenes_index.parquet"
XT_PATH = PROJECT_ROOT / "data" / "processed" / "xt_grid.npz"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1"
STRIDE_S = 0.5

_worth: Stage1Worth | None = None


def _init_worker() -> None:
    global _worth
    _worth = Stage1Worth(load_xt(XT_PATH))


def _compute(scene_id: str) -> tuple[str, float]:
    t0 = time.time()
    scene = Scene.load(SCENE_DIR / f"{scene_id}.npz")
    frames = sample_frames(scene.n_frames, scene.fps, STRIDE_S)
    values = np.stack([_worth.coalition_values(scene, t) for t in frames])
    phi = np.stack(
        [
            np.stack([shapley_from_values(v[:, k]) for k in range(v.shape[1])], axis=-1)
            for v in values
        ]
    )
    np.savez_compressed(
        OUT_DIR / f"{scene_id}.npz",
        frames=frames,
        values=values.astype(np.float32),
        phi=phi,
        attack_ids=scene.attack_ids,
        weight_names=np.array(WEIGHT_NAMES),
    )
    return scene_id, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(INDEX_PATH)
    todo = [
        sid
        for sid in index[~index.tracking_glitch].scene_id
        if not (OUT_DIR / f"{sid}.npz").exists()
    ][: args.limit]
    print(f"計算対象: {len(todo)} シーン(workers={args.workers})", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_compute, sid) for sid in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            sid, sec = fut.result()
            if k % 10 == 0 or k == len(todo):
                elapsed = time.time() - t0
                print(f"[{k}/{len(todo)}] {sid} {sec:.1f}s  経過 {elapsed / 60:.1f}分", flush=True)


if __name__ == "__main__":
    main()
