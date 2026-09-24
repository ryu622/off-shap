import numpy as np
import pytest

from off_shap.pitch_control import (
    PitchControlParams,
    make_grid,
    pitch_control,
    pitch_control_coalitions,
    time_to_intercept,
)
from off_shap.shapley import all_coalitions

PARAMS = PitchControlParams()


def _shaw_reference(point, att_pos, att_vel, def_pos, def_vel, ball, params=PARAMS):
    """Shaw (Metrica_PitchControl.py) と同じ Euler 積分を1地点ずつ行う参照実装。"""
    ball_time = np.linalg.norm(point - ball) / params.average_ball_speed
    tti_a = time_to_intercept(att_pos, att_vel, point[None], params)[:, 0]
    tti_d = time_to_intercept(def_pos, def_vel, point[None], params)[:, 0]
    k = np.pi / (np.sqrt(3.0) * params.tti_sigma)
    pa = np.zeros(len(att_pos))
    pd_ = np.zeros(len(def_pos))
    for step in range(round(params.max_int_time / params.int_dt)):
        t = ball_time + step * params.int_dt
        rem = 1 - pa.sum() - pd_.sum()
        pa += rem / (1 + np.exp(-k * (t - tti_a))) * params.lambda_att * params.int_dt
        pd_ += rem / (1 + np.exp(-k * (t - tti_d))) * params.lambda_def * params.int_dt
    return pa.sum()


def _random_setup(seed, n_att=4, n_def=4):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform([0, 0], [105, 68], (n_att, 2)),
        rng.normal(0, 3, (n_att, 2)),
        rng.uniform([0, 0], [105, 68], (n_def, 2)),
        rng.normal(0, 3, (n_def, 2)),
        rng.uniform([0, 0], [105, 68], 2),
    )


def test_make_grid_covers_pitch():
    grid, area = make_grid(105.0, 68.0, cell=2.0)
    assert grid.shape == (52 * 34, 2)
    assert area * len(grid) == pytest.approx(105 * 68)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_shaw_euler_reference(seed):
    ap, av, dp, dv, ball = _random_setup(seed)
    points = np.random.default_rng(seed + 10).uniform([0, 0], [105, 68], (15, 2))
    ours = pitch_control(ap, av, dp, dv, ball, points)
    ref = np.array([_shaw_reference(p, ap, av, dp, dv, ball) for p in points])
    np.testing.assert_allclose(ours, ref, atol=0.03)


def test_symmetric_setup_gives_half():
    att = np.array([[40.0, 34.0]])
    dfn = np.array([[60.0, 34.0]])
    zero = np.zeros((1, 2))
    pc = pitch_control(att, zero, dfn, zero, np.array([50.0, 0.0]), np.array([[50.0, 34.0]]))
    assert pc[0] == pytest.approx(0.5, abs=1e-4)


def test_nearby_player_dominates():
    att = np.array([[50.0, 34.0]])
    dfn = np.array([[90.0, 10.0]])
    zero = np.zeros((1, 2))
    pc = pitch_control(att, zero, dfn, zero, np.array([50.0, 34.0]), np.array([[51.0, 34.0]]))
    assert pc[0] > 0.99


def test_coalitions_match_individual_runs_and_are_monotone():
    ap, av, dp, dv, ball = _random_setup(5)
    grid, _ = make_grid(105.0, 68.0, cell=5.0)
    members = all_coalitions(len(ap))
    pcs = pitch_control_coalitions(ap, av, members, dp, dv, ball, grid)
    for c in [0, 5, 15]:
        m = members[c]
        single = pitch_control(ap[m], av[m], dp, dv, ball, grid) if m.any() else np.zeros(len(grid))
        np.testing.assert_allclose(pcs[c], single, atol=1e-4)
    # 選手を加えて攻撃側のコントロールが減ることはない
    full, empty = pcs[-1], pcs[0]
    assert np.all(full >= empty - 1e-6)
    assert np.all(pcs[0b0011] >= pcs[0b0001] - 1e-6)


def test_goalkeeper_rate_and_fixed_attackers_are_used():
    zero = np.zeros((1, 2))
    point = np.array([[50.0, 34.0]])
    att, dfn, ball = np.array([[45.0, 34.0]]), np.array([[55.0, 34.0]]), np.array([50.0, 0.0])
    base = pitch_control(
        att, zero, dfn[:0], zero[:0], ball, point, defend_gk_pos=dfn, defend_gk_vel=zero
    )
    assert base[0] < 0.5  # GK は λ が大きいので等距離でも守備側が優位
    with_fixed = pitch_control_coalitions(
        att,
        zero,
        np.zeros((1, 1), bool),
        dfn,
        zero,
        ball,
        point,
        attack_fixed_pos=att,
        attack_fixed_vel=zero,
    )
    assert with_fixed[0, 0] == pytest.approx(0.5, abs=1e-4)
