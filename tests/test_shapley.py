"""Shapley 値の公理と既知のゲームでの値の検証。"""

import numpy as np
import pytest

from off_shap.shapley import (
    all_coalitions,
    coalition_values,
    exact_shapley,
    permutation_shapley,
    restricted_shapley,
    shapley_from_values,
)


def _random_game(n: int, seed: int = 0) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=1 << n)
    values[0] = 0.0
    return values


def test_all_coalitions_rows_match_bitmask():
    members = all_coalitions(3)
    assert members.shape == (8, 3)
    assert members[0b101].tolist() == [True, False, True]


@pytest.mark.parametrize("n", [1, 2, 5, 9, 10])
def test_efficiency(n):
    # ∑φᵢ = v(N) − v(∅)
    values = _random_game(n)
    values[0] = 0.7
    assert shapley_from_values(values).sum() == pytest.approx(values[-1] - values[0])


def test_symmetry():
    # 0 と 1 を入れ替えても値が変わらないゲームでは φ₀ = φ₁
    def v(m):
        return float(m[0] or m[1]) + 2.0 * m[2]

    phi = exact_shapley(3, v)
    assert phi[0] == pytest.approx(phi[1])


def test_null_player():
    # どの連合に加わっても値を変えない選手の φ は 0
    base = _random_game(3)

    def v(m):
        return base[int(m[0]) | int(m[1]) << 1 | int(m[2]) << 2]

    phi = exact_shapley(4, v)  # プレイヤー3 は v に現れない
    assert phi[3] == pytest.approx(0.0)


def test_additivity():
    a, b = _random_game(4, 1), _random_game(4, 2)
    np.testing.assert_allclose(
        shapley_from_values(a + b), shapley_from_values(a) + shapley_from_values(b)
    )


def test_additive_game_returns_weights():
    weights = np.array([1.0, -2.0, 0.5, 3.0])
    phi = exact_shapley(4, lambda m: float(weights @ m))
    np.testing.assert_allclose(phi, weights)


def test_glove_game():
    # 左手袋1つ(0)・右手袋2つ(1, 2)。左右そろえば価値1 → φ = (2/3, 1/6, 1/6)
    phi = exact_shapley(3, lambda m: float(m[0] and (m[1] or m[2])))
    np.testing.assert_allclose(phi, [2 / 3, 1 / 6, 1 / 6])


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        shapley_from_values(np.zeros(6))


def test_permutation_estimate_converges_to_exact():
    values = _random_game(6, 3)
    exact = shapley_from_values(values)
    lookup = {tuple(m): values[i] for i, m in enumerate(all_coalitions(6))}
    est, se = permutation_shapley(6, lambda m: lookup[tuple(m)], 3000, np.random.default_rng(0))
    assert np.all(np.abs(est - exact) < 4 * se + 1e-9)
    assert est.sum() == pytest.approx(values[-1] - values[0])  # 順列ごとに効率性が成り立つ


def test_coalition_values_passes_member_mask():
    values = coalition_values(3, lambda m: float(m.sum()))
    assert values.tolist() == [0, 1, 1, 2, 1, 2, 2, 3]


def test_restricted_shapley_with_all_players_equals_full():
    values = _random_game(5, 7)
    np.testing.assert_allclose(
        restricted_shapley(values, np.ones(5, bool)), shapley_from_values(values)
    )


def test_restricted_shapley_fixes_others_as_present():
    values = _random_game(4, 8)
    players = np.array([True, False, True, False])
    phi = restricted_shapley(values, players)
    assert phi[1] == 0 and phi[3] == 0
    # 効率性: 対象選手の合計 = v(N) − v(対象外だけ)
    assert phi.sum() == pytest.approx(values[-1] - values[0b1010])
    # 加法ゲームなら対象選手の重みがそのまま返る
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    additive = coalition_values(4, lambda m: float(weights @ m))
    np.testing.assert_allclose(restricted_shapley(additive, players), [1, 0, 3, 0])
