"""Season-agnostic gameweek aggregation, shared by the live and historical ingests.

A player can have more than one fixture in a single gameweek (a "double
gameweek"). Both the live element-summary feed and the vaastav historical CSVs
give one row per fixture; this module rolls those up into exactly one row per
gameweek so `player_gameweek_stats` keeps its (player_id, season, gameweek)
primary key.

Aggregation rules (see DECISIONS.md):
  * counting stats (points, minutes, goals, assists, bps, bonus, starts, cards,
    clean sheets, expected_*) are SUMMED across the gameweek's fixtures;
  * minutes are NOT capped — a double gameweek can legitimately exceed 90;
  * was_home and opponent_team are taken from the FIRST fixture of the gameweek
    (ordered by kick-off time);
  * now_cost is the price at the LAST fixture (most recent);
  * is_double_gameweek is 1 when the gameweek had more than one fixture.

`aggregate_history` is deliberately pure (no DB, no network) so it is cheap to
unit-test; callers attach player_id / season to each returned row.
"""

from collections import defaultdict

# Fields summed as integers across a gameweek's fixtures.
_INT_SUM_FIELDS: tuple[tuple[str, str], ...] = (
    ("total_points", "total_points"),
    ("minutes", "minutes_played"),
    ("goals_scored", "goals_scored"),
    ("assists", "assists"),
    ("clean_sheets", "clean_sheets"),
    ("yellow_cards", "yellow_cards"),
    ("red_cards", "red_cards"),
    ("bonus", "bonus"),
    ("bps", "bps"),
    ("starts", "starts"),
)

# Fields summed as floats; absent everywhere in the gameweek -> None.
_FLOAT_SUM_FIELDS: tuple[str, ...] = (
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)


def _is_missing(value: object) -> bool:
    """True for None, empty string, or a NaN float (pandas fills gaps with NaN)."""
    if value is None or value == "":
        return True
    return isinstance(value, float) and value != value


def _to_int(value: object) -> int:
    if _is_missing(value):
        return 0
    return int(round(float(value)))


def _to_float_or_none(value: object) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _fixture_sort_key(row: dict) -> tuple:
    """Order fixtures within a gameweek: by kick-off time, then fixture id."""
    return (row.get("kickoff_time") or "", _to_int(row.get("fixture")))


def aggregate_history(
    history: list[dict], team_map: dict[int, str]
) -> list[dict]:
    """Roll per-fixture rows up to one row per gameweek.

    `history` rows use the FPL element-summary field names (`round`, `minutes`,
    `value`, `opponent_team` as a team id, `was_home` as a bool, ...). The
    historical ingest normalises its CSV rows to the same shape before calling
    this. Returns rows ordered by gameweek, each carrying the
    `player_gameweek_stats` column names (minus player_id / season).
    """
    by_gameweek: dict[int, list[dict]] = defaultdict(list)
    for row in history:
        gameweek = _to_int(row.get("round"))
        if gameweek < 1 or gameweek > 38:
            continue  # guard against stray / preseason rows
        by_gameweek[gameweek].append(row)

    aggregated: list[dict] = []
    for gameweek in sorted(by_gameweek):
        fixtures = sorted(by_gameweek[gameweek], key=_fixture_sort_key)
        first, last = fixtures[0], fixtures[-1]

        out: dict = {"gameweek": gameweek}

        for src_field, dest_field in _INT_SUM_FIELDS:
            out[dest_field] = sum(_to_int(f.get(src_field)) for f in fixtures)

        for field in _FLOAT_SUM_FIELDS:
            values = [
                _to_float_or_none(f.get(field)) for f in fixtures
            ]
            present = [v for v in values if v is not None]
            out[field] = round(sum(present), 3) if present else None

        opponent_id = _to_int(first.get("opponent_team"))
        out["opponent_team"] = team_map.get(opponent_id, str(opponent_id))
        out["was_home"] = 1 if first.get("was_home") in (True, 1, "True", "true") else 0
        out["now_cost"] = _to_int(last.get("value")) or None
        out["is_double_gameweek"] = 1 if len(fixtures) > 1 else 0

        aggregated.append(out)

    return aggregated
