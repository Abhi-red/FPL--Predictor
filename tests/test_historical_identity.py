import sqlite3

import pytest
import pandas as pd

from db import get_connection, init_db
from features import build_features
from ingest import backfill_historical


def test_schema_preserves_identity_for_each_season(tmp_path):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "player_season_identity" in tables


def test_historical_features_use_historical_club_and_position(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO players (player_id, element_id, web_name, position, team) "
                     "VALUES (42, 7, 'Example', 'MID', 'Arsenal')")
        conn.execute("INSERT INTO player_gameweek_stats (player_id, season, gameweek, total_points, minutes_played, opponent_team) "
                     "VALUES (42, '2021-22', 1, 5, 90, 'Arsenal')")
        conn.execute("CREATE TABLE IF NOT EXISTS player_season_identity "
                     "(player_id INTEGER, season TEXT, web_name TEXT, position TEXT, team TEXT, "
                     "PRIMARY KEY (player_id, season))")
        conn.execute("INSERT INTO player_season_identity VALUES (42, '2021-22', 'Example', 'DEF', 'Brighton')")
    monkeypatch.setattr(build_features, "get_connection", lambda: get_connection(path))
    frame = build_features.load_stats_frame()
    assert frame.loc[0, "team"] == "Brighton"
    assert frame.loc[0, "position"] == "DEF"


def test_missing_historical_identity_fails_instead_of_using_current_club(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO players (player_id, web_name, position, team) "
                     "VALUES (42, 'Example', 'MID', 'Arsenal')")
        conn.execute("INSERT INTO player_gameweek_stats (player_id, season, gameweek, total_points) "
                     "VALUES (42, '2021-22', 1, 5)")
    monkeypatch.setattr(build_features, "get_connection", lambda: get_connection(path))
    with pytest.raises(RuntimeError, match="identity"):
        build_features.load_stats_frame()


def test_backfill_writes_season_identity_even_for_current_player(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO players (player_id, element_id, web_name, position, team) "
                     "VALUES (42, 7, 'Example', 'MID', 'Arsenal')")
    monkeypatch.setattr(backfill_historical, "get_connection", lambda: get_connection(path))
    players = pd.DataFrame([{"id": 8, "code": 42, "element_type": 2, "team": 2,
                             "web_name": "Example", "first_name": "An", "second_name": "Example",
                             "now_cost": 50}])
    teams = pd.DataFrame([{"id": 2, "name": "Brighton"}, {"id": 3, "name": "Arsenal"}])
    monkeypatch.setattr(backfill_historical, "_fetch_csv",
                        lambda url: players if url.endswith("players_raw.csv") else teams)
    monkeypatch.setattr(backfill_historical, "_load_merged_gw",
                        lambda season: pd.DataFrame([{"element": 8, "round": 1,
                                                       "total_points": 6, "minutes": 90,
                                                       "value": 50, "opponent_team": 3}]))
    backfill_historical.backfill_season("2021-22")
    with get_connection(path) as conn:
        identity = conn.execute("SELECT position, team FROM player_season_identity "
                                "WHERE player_id=42 AND season='2021-22'").fetchone()
    assert tuple(identity) == ("DEF", "Brighton")
