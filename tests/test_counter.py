import numpy as np
import pandas as pd

from off_shap.counter import _find_end, extract_candidates, smooth_owner, to_attack_frame
from off_shap.loader import MatchData, MatchMeta

FPS = 25


def _meta() -> MatchMeta:
    return MatchMeta(
        match_id="TEST",
        frame_rate=FPS,
        pitch_length=105.0,
        pitch_width=68.0,
        home_team_id="H",
        away_team_id="A",
        team_names={"H": "home", "A": "away"},
        players=[],
    )


def _match(owner: list[str], alive: list[bool], ball_x: np.ndarray, events=None) -> MatchData:
    n = len(owner)
    frames = pd.DataFrame(
        {
            "frame_id": np.arange(n),
            "period_id": 1,
            "t": np.arange(n) / FPS,
            "ball_x": ball_x,
            "ball_y": np.full(n, 34.0),
            "ball_z": 0.0,
            "ball_alive": alive,
            "owning_team_id": owner,
        }
    )
    if events is None:
        events = pd.DataFrame(
            columns=[
                "event_id",
                "period_id",
                "t",
                "event_type",
                "event_name",
                "team_id",
                "possession_change",
                "winner_team_id",
                "sub_type",
                "result",
                "xg",
                "counter_attack_flag",
            ]
        )
    return MatchData(meta=_meta(), frames=frames, players=pd.DataFrame(), events=events)


def test_smooth_owner_absorbs_short_flicker():
    owner = np.array(["A"] * 30 + ["B"] * 3 + ["A"] * 30, dtype=object)
    assert (smooth_owner(owner, min_run=25) == "A").all()


def test_smooth_owner_keeps_long_switch():
    owner = np.array(["A"] * 30 + ["B"] * 30, dtype=object)
    assert (smooth_owner(owner, min_run=25) == owner).all()


def test_find_end_reasons():
    owner = np.array(["H"] * 10 + ["A"] * 10, dtype=object)
    alive = np.ones(20, dtype=bool)
    assert _find_end(owner, alive, 0, "H", 100) == (10, "lost")
    alive[5] = False
    assert _find_end(owner, alive, 0, "H", 100) == (5, "dead")
    assert _find_end(owner, np.ones(20, bool), 0, "H", 4) == (4, "timeout")


def test_to_attack_frame_rotates_away():
    x, y = to_attack_frame(np.array([10.0]), np.array([5.0]), "away", 105.0, 68.0)
    assert x[0] == 95.0 and y[0] == 63.0


def test_extract_candidates_turnover_to_shot():
    # A が 3 秒保持 → H が奪取して 4 秒で 40m 前進 → デッド
    owner = ["A"] * 75 + ["H"] * 100 + ["H"] * 25
    alive = [True] * 175 + [False] * 25
    ball_x = np.concatenate([np.full(75, 40.0), np.linspace(30, 70, 100), np.full(25, 70.0)])
    events = pd.DataFrame(
        [
            {
                "event_id": "s1",
                "period_id": 1,
                "t": 6.5,
                "event_type": "SHOT",
                "event_name": "shot",
                "team_id": "H",
                "possession_change": None,
                "winner_team_id": None,
                "sub_type": None,
                "result": "SAVED",
                "xg": 0.2,
                "counter_attack_flag": "true",
            }
        ]
    )
    cands = extract_candidates(_match(owner, alive, ball_x, events))
    assert len(cands) == 1
    c = cands.iloc[0]
    assert c.team_id == "H" and c.end_reason == "dead" and c.outcome == "shot"
    assert c.duration_s == 4.0 and c.xg == 0.2 and c.dfl_counter_flag
    assert abs(c.progress_5s - 40.0) < 1e-9


def test_restart_is_not_a_turnover():
    # デッドボール中に保持が切り替わった場合(スローイン等)は候補にしない
    owner = ["A"] * 50 + ["H"] * 100
    alive = [True] * 40 + [False] * 20 + [True] * 90
    cands = extract_candidates(_match(owner, alive, np.full(150, 50.0)))
    assert len(cands) == 0
