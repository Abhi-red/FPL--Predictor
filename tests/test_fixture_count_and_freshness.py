from ingest.aggregate import aggregate_history
from models.predict import next_gameweek, upcoming_fixtures, assert_not_older_than_site
from db import get_connection, init_db
from models import predict
import pandas as pd


def test_aggregation_records_exact_number_of_fixtures():
    rows = [{"round": 7, "fixture": n, "opponent_team": 2, "was_home": True,
             "total_points": 2, "minutes": 90, "value": 50} for n in (1, 2, 3)]
    assert aggregate_history(rows, {2: "Arsenal"})[0]["fixture_count"] == 3


def test_upcoming_fixture_count_includes_both_club_matches(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"team_h": 1, "team_a": 2, "kickoff_time": "2026-10-01"},
                {"team_h": 3, "team_a": 1, "kickoff_time": "2026-10-04"},
            ]

    monkeypatch.setattr("models.predict.requests.get", lambda *args, **kwargs: Response())
    bootstrap = {"teams": [{"id": 1, "name": "Arsenal"},
                           {"id": 2, "name": "Chelsea"},
                           {"id": 3, "name": "Brighton"}]}
    assert upcoming_fixtures(7, bootstrap)["Arsenal"] == (1, "Chelsea", 2)


def test_double_fixture_count_reaches_model_feature_row(tmp_path, monkeypatch):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO players (player_id, element_id, web_name, position, team, now_cost) "
                     "VALUES (42, 1, 'Example', 'MID', 'Arsenal', 75)")
    monkeypatch.setattr(predict, "get_connection", lambda: get_connection(path))
    monkeypatch.setattr(predict, "upcoming_fixtures", lambda *args: {"Arsenal": (1, "Chelsea", 2)})
    monkeypatch.setattr(predict, "strengths_by_season", lambda *args: {})
    stats = pd.DataFrame([{"player_id": 42, "season": "2026-27", "gameweek": 6,
                           "position": "MID", "team": "Arsenal", "web_name": "Example",
                           "now_cost": 75, "total_points": 5, "minutes_played": 90,
                           "goals_scored": 1, "assists": 0, "bps": 25, "bonus": 1,
                           "expected_goals": 0.3, "expected_assists": 0.1,
                           "was_home": 0, "opponent_team": "Brighton", "fixture_count": 1}])
    monkeypatch.setattr(predict, "load_stats_frame", lambda: stats)
    result = predict.build_upcoming_matrix(7, {})
    assert result.iloc[0]["fixture_count"] == 2


def test_in_progress_gameweek_is_not_treated_as_unplayed():
    bootstrap = {"events": [{"id": 7, "finished": False, "is_current": True},
                            {"id": 8, "finished": False, "is_current": False}]}
    assert next_gameweek(bootstrap) == 8


def test_published_gameweek_cannot_be_replaced_by_older_local_data(tmp_path):
    site = tmp_path / "meta.json"
    site.write_text('{"season": "2026-27", "gameweek": 6}')
    try:
        assert_not_older_than_site("2026-27", 3, site)
    except RuntimeError as error:
        assert "GW6" in str(error)
    else:
        raise AssertionError("stale predictions were accepted")
    assert_not_older_than_site("2026-27", 7, site)
