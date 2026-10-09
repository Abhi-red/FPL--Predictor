from db import get_connection, init_db
from models.predict import UPSERT_PREDICTION


def test_new_raw_prediction_clears_previous_news_adjustment(tmp_path):
    path = tmp_path / "fpl.db"
    init_db(path)
    with get_connection(path) as conn:
        conn.execute("INSERT INTO players (player_id, web_name, position, team) "
                     "VALUES (42, 'Example', 'MID', 'Arsenal')")
        conn.execute("INSERT INTO predictions (player_id, season, gameweek, raw_points, "
                     "adjusted_points, adjustment_factor, adjustment_reason, news_url, generated_at) "
                     "VALUES (42, '2026-27', 7, 5, 3.5, 0.7, 'old news', 'https://example.com', 'old')")
        conn.execute(UPSERT_PREDICTION, {"player_id": 42, "season": "2026-27",
                                         "gameweek": 7, "raw_points": 6, "generated_at": "new"})
        row = conn.execute("SELECT raw_points, adjusted_points, adjustment_factor, "
                           "adjustment_reason, news_url FROM predictions WHERE player_id=42").fetchone()
    assert tuple(row) == (6, None, None, None, None)
