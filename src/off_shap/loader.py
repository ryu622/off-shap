"""idsse-data(kloppy経由)の読み込みと、軽量なテーブル形式へのキャッシュ。

kloppyでのロードは1試合あたり約40秒かかるため、トラッキング・イベントを
一度だけ parquet に変換し、以降の処理(カウンター抽出・Shapley計算)は
すべてこのキャッシュから読む。キャッシュ以降は kloppy に依存しないので、
研究室データ(J.League等)に差し替える場合も同じスキーマのテーブルを
作る変換層を書けばよい。

座標系(キャッシュ時点):
- 単位はメートル、原点はピッチの角、x ∈ [0, pitch_length], y ∈ [0, pitch_width]
- 向きは STATIC_HOME_AWAY(両ピリオドを通じて home が +x 方向へ攻撃)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from kloppy import sportec
from kloppy.domain import EventDataset, Orientation, TrackingDataset

MATCH_IDS = [
    "J03WPY",
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WQQ",
    "J03WR9",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

GOALKEEPER_POSITION_NAME = "Goalkeeper"


@dataclass
class PlayerInfo:
    player_id: str
    team_id: str
    name: str
    jersey_no: int | None
    position: str
    starting: bool
    is_goalkeeper: bool


@dataclass
class MatchMeta:
    match_id: str
    frame_rate: float
    pitch_length: float
    pitch_width: float
    home_team_id: str
    away_team_id: str
    team_names: dict[str, str]
    players: list[PlayerInfo]

    def ground_of(self, team_id: str) -> str:
        return "home" if team_id == self.home_team_id else "away"


def load_match_raw(match_id: str) -> tuple[TrackingDataset, EventDataset]:
    tracking = sportec.load_open_tracking_data(match_id=match_id)
    events = sportec.load_open_event_data(match_id=match_id)
    tracking = tracking.transform(to_orientation=Orientation.STATIC_HOME_AWAY)
    events = events.transform(to_orientation=Orientation.STATIC_HOME_AWAY)
    return tracking, events


def _build_meta(match_id: str, tracking: TrackingDataset) -> MatchMeta:
    md = tracking.metadata
    home = next(t for t in md.teams if t.ground.value == "home")
    away = next(t for t in md.teams if t.ground.value == "away")
    players = [
        PlayerInfo(
            player_id=p.player_id,
            team_id=team.team_id,
            name=p.name,
            jersey_no=p.jersey_no,
            position=str(p.starting_position),
            starting=bool(p.starting),
            is_goalkeeper=str(p.starting_position) == GOALKEEPER_POSITION_NAME,
        )
        for team in md.teams
        for p in team.players
    ]
    return MatchMeta(
        match_id=match_id,
        frame_rate=float(md.frame_rate),
        pitch_length=float(md.pitch_dimensions.pitch_length),
        pitch_width=float(md.pitch_dimensions.pitch_width),
        home_team_id=home.team_id,
        away_team_id=away.team_id,
        team_names={t.team_id: t.name for t in md.teams},
        players=players,
    )


def _tracking_tables(
    tracking: TrackingDataset, meta: MatchMeta
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """フレーム単位テーブル(ボール・状態)と、選手位置のlong形式テーブルを作る。"""
    L, W = meta.pitch_length, meta.pitch_width
    team_of = {p.player_id: p.team_id for p in meta.players}

    frame_rows = []
    player_rows = []
    for frame in tracking.records:
        ball = frame.ball_coordinates
        frame_rows.append(
            {
                "frame_id": frame.frame_id,
                "period_id": frame.period.id,
                "t": frame.timestamp.total_seconds(),
                "ball_x": ball.x * L if ball is not None else float("nan"),
                "ball_y": ball.y * W if ball is not None else float("nan"),
                "ball_z": ball.z if ball is not None and ball.z is not None else float("nan"),
                "ball_alive": frame.ball_state is not None and frame.ball_state.value == "alive",
                "owning_team_id": (
                    frame.ball_owning_team.team_id if frame.ball_owning_team is not None else None
                ),
            }
        )
        for player, data in frame.players_data.items():
            if data.coordinates is None:
                continue
            player_rows.append(
                (
                    frame.frame_id,
                    player.player_id,
                    team_of.get(player.player_id),
                    data.coordinates.x * L,
                    data.coordinates.y * W,
                    data.speed,
                )
            )

    frames = pd.DataFrame(frame_rows)
    players = pd.DataFrame(
        player_rows, columns=["frame_id", "player_id", "team_id", "x", "y", "speed"]
    )
    return frames, players


def _event_table(events: EventDataset, meta: MatchMeta) -> pd.DataFrame:
    """Shapley計算・カウンター抽出で使う列だけを抜き出したイベント表。

    DFL生データの属性(PossessionChange, WinnerTeam, xG, CounterAttack等)は
    kloppyのドメインモデルに載らないため raw_event から直接拾う。
    """
    L, W = meta.pitch_length, meta.pitch_width
    rows = []
    for e in events.records:
        raw = e.raw_event or {}
        coords = e.coordinates
        end = getattr(e, "receiver_coordinates", None)
        rows.append(
            {
                "event_id": e.event_id,
                "period_id": e.period.id,
                "t": e.timestamp.total_seconds(),
                "event_type": e.event_type.name,
                "event_name": e.event_name,
                "team_id": e.team.team_id if e.team is not None else raw.get("Team"),
                "player_id": e.player.player_id if e.player is not None else raw.get("Player"),
                "x": coords.x * L if coords is not None else float("nan"),
                "y": coords.y * W if coords is not None else float("nan"),
                "end_x": end.x * L if end is not None else float("nan"),
                "end_y": end.y * W if end is not None else float("nan"),
                "result": e.result.name if getattr(e, "result", None) is not None else None,
                "recipient_id": raw.get("Recipient"),
                "possession_change": raw.get("PossessionChange"),
                "winner_team_id": raw.get("WinnerTeam"),
                "winner_player_id": raw.get("Winner"),
                "loser_player_id": raw.get("Loser"),
                "sub_type": raw.get("Type"),
                "xg": float(raw["xG"]) if raw.get("xG") not in (None, "") else float("nan"),
                "counter_attack_flag": raw.get("CounterAttack"),
                "possession_phase": raw.get("BallPossessionPhase"),
            }
        )
    return pd.DataFrame(rows)


def cache_paths(match_id: str, cache_dir: Path = CACHE_DIR) -> dict[str, Path]:
    return {
        "frames": cache_dir / f"{match_id}_frames.parquet",
        "players": cache_dir / f"{match_id}_players.parquet",
        "events": cache_dir / f"{match_id}_events.parquet",
        "meta": cache_dir / f"{match_id}_meta.json",
    }


def build_cache(match_id: str, cache_dir: Path = CACHE_DIR, overwrite: bool = False) -> None:
    paths = cache_paths(match_id, cache_dir)
    if not overwrite and all(p.exists() for p in paths.values()):
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    tracking, events = load_match_raw(match_id)
    meta = _build_meta(match_id, tracking)
    frames, players = _tracking_tables(tracking, meta)
    frames.to_parquet(paths["frames"], index=False)
    players.to_parquet(paths["players"], index=False)
    _event_table(events, meta).to_parquet(paths["events"], index=False)
    paths["meta"].write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2))


@dataclass
class MatchData:
    meta: MatchMeta
    frames: pd.DataFrame  # 1行=1フレーム
    players: pd.DataFrame  # 1行=(フレーム, 選手)
    events: pd.DataFrame


def load_match(match_id: str, cache_dir: Path = CACHE_DIR) -> MatchData:
    """キャッシュから1試合分を読む(未作成なら kloppy から作成する)。"""
    build_cache(match_id, cache_dir)
    paths = cache_paths(match_id, cache_dir)
    meta_dict = json.loads(paths["meta"].read_text())
    meta_dict["players"] = [PlayerInfo(**p) for p in meta_dict["players"]]
    return MatchData(
        meta=MatchMeta(**meta_dict),
        frames=pd.read_parquet(paths["frames"]),
        players=pd.read_parquet(paths["players"]),
        events=pd.read_parquet(paths["events"]),
    )
