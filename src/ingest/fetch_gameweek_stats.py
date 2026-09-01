"""Ingest the current season's per-gameweek stats for every player.

For each player we pull `element-summary/{element_id}/`, roll double gameweeks
up to a single row (see src/ingest/aggregate.py), and upsert into
`player_gameweek_stats` tagged with the live SEASON.

Requests are sequential with a short delay so we don't hammer the API. Transient
failures are retried with backoff; a player that still fails is skipped and
logged rather than aborting the whole run.

Run:
    python src/ingest/fetch_gameweek_stats.py
"""

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.aggregate import aggregate_history  # noqa: E402
from constants import (  # noqa: E402
    FPL_BASE,
    FPL_HEADERS,
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT,
    SEASON,
)
from db import get_connection  # noqa: E402

BOOTSTRAP_URL = f"{FPL_BASE}/bootstrap-static/"
ELEMENT_SUMMARY_URL = f"{FPL_BASE}/element-summary/{{element_id}}/"

# Politeness delay between element-summary calls.
INTER_REQUEST_DELAY = 0.3

UPSERT_GAMEWEEK_STATS = """
INSERT INTO player_gameweek_stats (
    player_id, season, gameweek, total_points, minutes_played, goals_scored,
    assists, now_cost, clean_sheets, yellow_cards, red_cards, was_home,
    opponent_team, bonus, bps, starts, expected_goals, expected_assists,
    expected_goal_involvements, expected_goals_conceded, is_double_gameweek
)
VALUES (
    :player_id, :season, :gameweek, :total_points, :minutes_played, :goals_scored,
    :assists, :now_cost, :clean_sheets, :yellow_cards, :red_cards, :was_home,
    :opponent_team, :bonus, :bps, :starts, :expected_goals, :expected_assists,
    :expected_goal_involvements, :expected_goals_conceded, :is_double_gameweek
)
ON CONFLICT(player_id, season, gameweek) DO UPDATE SET
    total_points               = excluded.total_points,
    minutes_played             = excluded.minutes_played,
    goals_scored               = excluded.goals_scored,
    assists                    = excluded.assists,
    now_cost                   = excluded.now_cost,
    clean_sheets               = excluded.clean_sheets,
    yellow_cards               = excluded.yellow_cards,
    red_cards                  = excluded.red_cards,
    was_home                   = excluded.was_home,
    opponent_team              = excluded.opponent_team,
    bonus                      = excluded.bonus,
    bps                        = excluded.bps,
    starts                     = excluded.starts,
    expected_goals             = excluded.expected_goals,
    expected_assists           = excluded.expected_assists,
    expected_goal_involvements = excluded.expected_goal_involvements,
    expected_goals_conceded    = excluded.expected_goals_conceded,
    is_double_gameweek         = excluded.is_double_gameweek
"""


def _get_json(url: str) -> dict:
    """GET `url` with retry + exponential backoff; raise on persistent failure."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url, headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 404:
                raise RuntimeError(f"{url} -> 404")  # definitive, don't retry
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                backoff = 2 ** attempt
                print(
                    f"  attempt {attempt}/{MAX_ATTEMPTS} for {url} failed "
                    f"({error}); retrying in {backoff}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
    raise RuntimeError(f"Could not fetch {url} after {MAX_ATTEMPTS} attempts") from last_error


def build_team_map(bootstrap: dict) -> dict[int, str]:
    """team id -> team name for the current season."""
    return {team["id"]: team["name"] for team in bootstrap["teams"]}


def load_players() -> list[dict]:
    """(player_id, element_id, web_name) for players that have a current element id."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT player_id, element_id, web_name FROM players "
            "WHERE element_id IS NOT NULL ORDER BY player_id"
        ).fetchall()
    return [dict(row) for row in rows]


def gameweek_rows_for_player(
    player_id: int, element_id: int, team_map: dict[int, str]
) -> list[dict]:
    """Fetch and aggregate one player's current-season gameweek history."""
    data = _get_json(ELEMENT_SUMMARY_URL.format(element_id=element_id))
    aggregated = aggregate_history(data.get("history", []), team_map)
    for row in aggregated:
        row["player_id"] = player_id
        row["season"] = SEASON
    return aggregated


def main() -> None:
    print("Fetching bootstrap-static for the team map")
    bootstrap = _get_json(BOOTSTRAP_URL)
    team_map = build_team_map(bootstrap)

    players = load_players()
    print(f"Fetching gameweek history for {len(players)} players")

    all_rows: list[dict] = []
    failures: list[tuple[int, str, str]] = []

    for index, player in enumerate(players, start=1):
        try:
            rows = gameweek_rows_for_player(
                player["player_id"], player["element_id"], team_map
            )
            all_rows.extend(rows)
        except Exception as error:  # noqa: BLE001 - skip-and-log per player
            failures.append((player["player_id"], player["web_name"], str(error)))
            print(
                f"  [{index}/{len(players)}] skipped {player['web_name']} "
                f"(id {player['player_id']}): {error}",
                file=sys.stderr,
            )
        time.sleep(INTER_REQUEST_DELAY)

    with get_connection() as conn:
        conn.executemany(UPSERT_GAMEWEEK_STATS, all_rows)

    doubles = sum(r["is_double_gameweek"] for r in all_rows)
    print(
        f"Wrote {len(all_rows)} gameweek rows for {len(players) - len(failures)} "
        f"players ({doubles} double-gameweek rows); {len(failures)} players failed"
    )
    if failures:
        print("Failed players:", file=sys.stderr)
        for player_id, name, error in failures:
            print(f"  {name} ({player_id}): {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
