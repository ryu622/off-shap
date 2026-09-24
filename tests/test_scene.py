import numpy as np

from off_shap.scene import Scene, _fill_nan, raw_max_speed, smooth_and_velocity

FPS = 25.0


def test_constant_velocity_is_recovered():
    t = np.arange(50) / FPS
    pos = np.stack([3.0 * t, -2.0 * t], axis=-1)[:, None, :]  # (T, 1, 2)
    smoothed, vel = smooth_and_velocity(pos, FPS)
    np.testing.assert_allclose(smoothed, pos, atol=1e-9)
    np.testing.assert_allclose(vel[:, 0], np.tile([3.0, -2.0], (50, 1)), atol=1e-9)


def test_fill_nan_interpolates_and_extends_edges():
    a = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])[:, None]
    np.testing.assert_allclose(_fill_nan(a)[:, 0], [1.0, 1.0, 2.0, 3.0, 3.0])


def test_raw_max_speed_detects_jump():
    pos = np.zeros((10, 1, 2))
    pos[5:, 0, 0] = 4.0  # 1フレームで4m = 100 m/s
    assert raw_max_speed(pos, FPS) == 100.0


def test_scene_roundtrip(tmp_path):
    T, n = 5, 2
    scene = Scene(
        scene_id="M_1_100",
        match_id="M",
        period_id=1,
        team_id="H",
        ground="home",
        fps=FPS,
        start_frame_id=100,
        end_frame_id=104,
        outcome="shot",
        xg=0.1,
        attack_ids=np.array(["a", "b"]),
        defend_ids=np.array(["c", "d"]),
        attack_pos=np.random.rand(T, n, 2),
        attack_vel=np.random.rand(T, n, 2),
        defend_pos=np.random.rand(T, n, 2),
        defend_vel=np.random.rand(T, n, 2),
        attack_gk_pos=np.random.rand(T, 2),
        defend_gk_pos=np.random.rand(T, 2),
        ball_pos=np.random.rand(T, 2),
        max_raw_speed=8.0,
    )
    scene.save(tmp_path / "s.npz")
    loaded = Scene.load(tmp_path / "s.npz")
    assert loaded.scene_id == "M_1_100" and loaded.period_id == 1 and loaded.xg == 0.1
    assert isinstance(loaded.period_id, int) and isinstance(loaded.max_raw_speed, float)
    assert list(loaded.attack_ids) == ["a", "b"]
    np.testing.assert_array_equal(loaded.attack_pos, scene.attack_pos)
