"""Double-gameweek aggregation is where a silent bug would be hardest to spot:
a wrong sum or a capped minutes value would quietly bias every downstream model.
These tests pin the rules in src/ingest/aggregate.py::aggregate_history.
"""

from ingest.aggregate import aggregate_history

TEAM_MAP = {1: "Arsenal", 2: "Spurs", 3: "Chelsea", 4: "Everton"}


def _fixture(**over):
    base = {
        "round": 7,
        "kickoff_time": "2026-10-01T14:00:00Z",
        "fixture": 100,
        "opponent_team": 2,
        "was_home": True,
        "minutes": 90,
        "total_points": 6,
        "goals_scored": 1,
        "assists": 0,
        "clean_sheets": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "bonus": 1,
        "bps": 25,
        "starts": 1,
        "value": 75,
        "expected_goals": "0.40",
        "expected_assists": "0.10",
        "expected_goal_involvements": "0.50",
        "expected_goals_conceded": "1.10",
    }
    base.update(over)
    return base


def test_single_fixture_gameweek_passes_through():
    rows = aggregate_history([_fixture()], TEAM_MAP)
    assert len(rows) == 1
    row = rows[0]
    assert row["gameweek"] == 7
    assert row["is_double_gameweek"] == 0
    assert row["total_points"] == 6
    assert row["minutes_played"] == 90
    assert row["opponent_team"] == "Spurs"
    assert row["was_home"] == 1
    assert row["expected_goals"] == 0.4


def test_double_gameweek_sums_counting_stats():
    rows = aggregate_history(
        [
            _fixture(fixture=100, kickoff_time="2026-10-01T14:00:00Z",
                     minutes=90, total_points=6, goals_scored=1, assists=1,
                     bonus=3, bps=30, starts=1,
                     expected_goals="0.40", expected_assists="0.20",
                     expected_goal_involvements="0.60", expected_goals_conceded="1.00"),
            _fixture(fixture=101, kickoff_time="2026-10-04T14:00:00Z",
                     opponent_team=3, was_home=False,
                     minutes=75, total_points=9, goals_scored=2, assists=0,
                     bonus=2, bps=28, starts=1,
                     expected_goals="0.55", expected_assists="0.05",
                     expected_goal_involvements="0.60", expected_goals_conceded="0.90"),
        ],
        TEAM_MAP,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["is_double_gameweek"] == 1
    assert row["total_points"] == 15
    assert row["minutes_played"] == 165          # summed, and > 90 (not capped)
    assert row["goals_scored"] == 3
    assert row["assists"] == 1
    assert row["bonus"] == 5
    assert row["bps"] == 58
    assert row["starts"] == 2
    assert round(row["expected_goals"], 2) == 0.95
    assert round(row["expected_goal_involvements"], 2) == 1.20
    # home/away + opponent come from the FIRST fixture (earlier kick-off)
    assert row["was_home"] == 1
    assert row["opponent_team"] == "Spurs"


def test_first_fixture_is_earliest_kickoff_regardless_of_input_order():
    rows = aggregate_history(
        [
            _fixture(fixture=101, kickoff_time="2026-10-04T14:00:00Z",
                     opponent_team=4, was_home=False),
            _fixture(fixture=100, kickoff_time="2026-10-01T14:00:00Z",
                     opponent_team=2, was_home=True),
        ],
        TEAM_MAP,
    )
    assert rows[0]["opponent_team"] == "Spurs"
    assert rows[0]["was_home"] == 1


def test_minutes_over_90_survive_a_triple_gameweek():
    rows = aggregate_history(
        [_fixture(fixture=f, kickoff_time=f"2026-10-0{i}T14:00:00Z", minutes=90)
         for i, f in enumerate((100, 101, 102), start=1)],
        TEAM_MAP,
    )
    assert len(rows) == 1
    assert rows[0]["minutes_played"] == 270
    assert rows[0]["is_double_gameweek"] == 1


def test_missing_expected_stats_yield_none_not_zero():
    row = aggregate_history(
        [{"round": 3, "minutes": 45, "total_points": 2, "value": 50,
          "opponent_team": 1, "was_home": False}],
        TEAM_MAP,
    )[0]
    assert row["expected_goals"] is None
    assert row["minutes_played"] == 45
    assert row["opponent_team"] == "Arsenal"


def test_rounds_are_split_and_ordered():
    rows = aggregate_history(
        [_fixture(round=9, fixture=1), _fixture(round=7, fixture=2), _fixture(round=7, fixture=3,
         kickoff_time="2026-10-05T14:00:00Z")],
        TEAM_MAP,
    )
    assert [r["gameweek"] for r in rows] == [7, 9]
    assert rows[0]["is_double_gameweek"] == 1
    assert rows[1]["is_double_gameweek"] == 0
