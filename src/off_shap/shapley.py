"""協力ゲームの Shapley 値。

連合はビットマスクで表す: プレイヤー i が連合に含まれる ⇔ mask の i ビット目が 1。
v(S) の評価(ピッチコントロール、生成モデルのロールアウトなど)と Shapley 値の
計算を分離し、評価側は `all_coalitions(n)` の全連合をまとめて(ベクトル化して)
計算できるようにしている。

    φᵢ = ∑[S ⊆ N∖{i}] |S|!(n−|S|−1)!/n! · (v(S∪{i}) − v(S))
"""

from __future__ import annotations

from collections.abc import Callable
from math import factorial

import numpy as np

MAX_EXACT_PLAYERS = 16  # 2¹⁶ = 65,536 連合。これを超える場合は permutation_shapley を使う


def all_coalitions(n: int) -> np.ndarray:
    """全 2ⁿ 連合の所属行列 (2ⁿ, n) bool。行インデックスがビットマスクに対応する。"""
    masks = np.arange(1 << n)
    return ((masks[:, None] >> np.arange(n)) & 1).astype(bool)


def coalition_values(n: int, value_fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """value_fn(members: (n,) bool) を全連合で評価した (2ⁿ,) 配列。"""
    return np.array([value_fn(members) for members in all_coalitions(n)], dtype=float)


def _weights(n: int) -> np.ndarray:
    """サイズ s の連合 S に対する重み s!(n−s−1)!/n!(s = 0..n−1)。"""
    return np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])


def shapley_from_values(values: np.ndarray) -> np.ndarray:
    """全連合の値 v (2ⁿ,) から厳密な Shapley 値 (n,) を計算する。"""
    size = values.shape[0]
    n = size.bit_length() - 1
    if size != 1 << n:
        raise ValueError(f"values の長さは 2ⁿ である必要がある(受け取った長さ: {size})")
    if n > MAX_EXACT_PLAYERS:
        raise ValueError(f"n={n} は厳密計算の上限 {MAX_EXACT_PLAYERS} を超えている")

    masks = np.arange(size)
    sizes = np.bitwise_count(masks)
    w = _weights(n)
    phi = np.empty(n)
    for i in range(n):
        without_i = masks[(masks >> i) & 1 == 0]
        marginal = values[without_i | (1 << i)] - values[without_i]
        phi[i] = np.dot(w[sizes[without_i]], marginal)
    return phi


def exact_shapley(n: int, value_fn: Callable[[np.ndarray], float]) -> np.ndarray:
    return shapley_from_values(coalition_values(n, value_fn))


def permutation_shapley(
    n: int,
    value_fn: Callable[[np.ndarray], float],
    n_permutations: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """順列サンプリングによる Shapley 値の近似(Castro et al., 2009)。

    v(S) の評価が重い場合(Stage 2 の軌道生成など)向け。同じ連合は再評価しない。
    戻り値は (推定値 (n,), 標準誤差 (n,))。
    """
    rng = rng or np.random.default_rng()
    cache: dict[int, float] = {}

    def v(mask: int) -> float:
        if mask not in cache:
            cache[mask] = float(value_fn(((mask >> np.arange(n)) & 1).astype(bool)))
        return cache[mask]

    samples = np.empty((n_permutations, n))
    for k in range(n_permutations):
        mask = 0
        prev = v(0)
        for i in rng.permutation(n):
            mask |= 1 << i
            cur = v(mask)
            samples[k, i] = cur - prev
            prev = cur
    if n_permutations > 1:
        se = samples.std(axis=0, ddof=1) / np.sqrt(n_permutations)
    else:
        se = np.full(n, np.nan)
    return samples.mean(axis=0), se


def restricted_shapley(values: np.ndarray, players: np.ndarray) -> np.ndarray:
    """players に含まれる選手だけでゲームを行い、それ以外は常に存在するとみなした Shapley 値。

    先行研究(A Model-Based Restricted Shapley Value)のように評価対象を
    ボール関与選手に限るベースライン用。全連合の値 values (2ⁿ,) から、
        v'(T) = v(T ∪ 固定メンバー)   (T ⊆ players)
    の部分ゲームを作って計算する。対象外の選手の値は 0 を返す。

    名前は似ているが、先行研究の Restricted Shapley Value(和を取る連合を
    観測された組み合わせに制限し、重みを正規化し直すもの)とは別物。
    """
    n = values.shape[0].bit_length() - 1
    idx = np.flatnonzero(players)
    fixed = sum(1 << int(i) for i in range(n) if not players[i])
    m = len(idx)
    sub_values = np.empty(1 << m)
    for t in range(1 << m):
        mask = fixed
        for j in range(m):
            if (t >> j) & 1:
                mask |= 1 << int(idx[j])
        sub_values[t] = values[mask]
    phi = np.zeros(n)
    if m:
        phi[idx] = shapley_from_values(sub_values)
    return phi
