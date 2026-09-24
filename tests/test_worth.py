import numpy as np
import pytest

from off_shap.scene import Scene
from off_shap.shapley import shapley_from_values
from off_shap.worth import Stage1Worth, sample_frames
from off_shap.xt import XTGrid


def _xt() -> XTGrid:
    # 相手ゴール(x=105)に近いほど価値が高い単純な面
    values = np.tile(np.linspace(0.01, 0.2, 6)[:, None], (1, 4))
    return XTGrid(values, 105.0, 68.0, 0, 0, 0)


def _scene(attack_xy, defend_xy) -> Scene:
    T = 3
    a = np.tile(np.array(attack_xy, float)[None], (T, 1, 1))
    d = np.tile(np.array(defend_xy, float)[None], (T, 1, 1))
    return Scene(
        scene_id="S",
        match_id="M",
        period_id=1,
        team_id="H",
        ground="home",
        fps=25.0,
        start_frame_id=0,
        end_frame_id=T - 1,
        outcome="lost",
        xg=0.0,
        attack_ids=np.array([f"a{i}" for i in range(len(attack_xy))]),
        defend_ids=np.array([f"d{i}" for i in range(len(defend_xy))]),
        attack_pos=a,
        attack_vel=np.zeros_like(a),
        defend_pos=d,
        defend_vel=np.zeros_like(d),
        attack_gk_pos=np.tile([5.0, 34.0], (T, 1)),
        defend_gk_pos=np.tile([100.0, 34.0], (T, 1)),
        ball_pos=np.tile([50.0, 34.0], (T, 1)),
        max_raw_speed=0.0,
    )


@pytest.fixture(scope="module")
def worth():
    return Stage1Worth(_xt(), cell=5.0)


def test_weights_integrate_to_mean_xt(worth):
    # PC ≡ 1 なら v はピッチ全体の xT 平均になる
    assert worth.weights.sum() == pytest.approx(worth.xt.at(worth.grid).mean())


def test_values_are_monotone_and_shapley_nonnegative(worth):
    scene = _scene([[70, 20], [80, 50], [40, 34]], [[75, 30], [60, 40]])
    values = worth.coalition_values(scene, 0)
    assert values.shape == (8, 3)
    for v in values.T:
        for mask in range(8):
            for i in range(3):
                assert v[mask | 1 << i] >= v[mask] - 1e-9
        assert np.all(shapley_from_values(v) >= -1e-9)


def test_advanced_player_gets_more_credit(worth):
    # 同条件で、相手ゴールに近く守備から離れた選手ほど φ が大きい
    scene = _scene([[85, 15], [30, 15]], [[60, 55]])
    phi = shapley_from_values(worth.coalition_values(scene, 0)[:, 0])
    assert phi[0] > phi[1] > 0


def test_alternative_weights_ignore_own_half(worth):
    # 自陣にしかいない選手は、相手陣内限定・ボール位置比の重みでは φ ≈ 0
    scene = _scene([[85, 15], [20, 34]], [[60, 55]])
    values = worth.coalition_values(scene, 0)
    phi_xt, phi_opp, phi_gain = (shapley_from_values(v) for v in values.T)
    assert phi_xt[1] > 0
    assert phi_opp[1] < 0.1 * phi_opp[0]
    assert phi_gain[1] < 0.1 * phi_gain[0]


def test_sample_frames_includes_last():
    assert sample_frames(30, 25.0, 0.5).tolist() == [0, 12, 24, 29]
    assert sample_frames(25, 25.0, 0.5).tolist() == [0, 12, 24]
