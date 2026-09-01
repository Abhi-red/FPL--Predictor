"""Ingest the FPL bootstrap-static feed and populate the `players` table.

Scope: this script ONLY writes `players` (identity: code, element id, name,
position, team, current price). Per-gameweek stats are a separate ingest step
(src/ingest/fetch_gameweek_stats.py).

Run:
    python src/ingest/fetch_fpl.py
"""

import sys
import time
from pathlib import Path

import requests

# The rest of the project runs scripts directly (e.g. `python src/db.py`), so
# put `src/` on the path and import its modules by bare name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    FPL_BASE,
    FPL_HEADERS,
    MAX_ATTEMPTS,
    POSITION_MAP,
    REQUEST_TIMEOUT,
)
from db import get_connection  # noqa: E402

BOOTSTRAP_URL = f"{FPL_BASE}/bootstrap-static/"

# (code, element_id, web_name, first_name, second_name, position, team, now_cost)
PlayerRow = tuple[int, int, str, str, str, str, str, int]


def fetch_bootstrap(url: str = BOOTSTRAP_URL) -> dict:
    """GET the bootstrap-static feed and return the parsed JSON body.

    Retries a few times on transient network / 5xx errors with exponential
    backoff, then gives up and raises. Any failure here should stop the
    pipeline loudly rather than let it publish stale or empty data.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()  # raises ValueError if body isn't JSON
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                backoff = 2 ** attempt
                print(
                    f"  fetch attempt {attempt}/{MAX_ATTEMPTS} failed ({error}); "
                    f"retrying in {backoff}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)

    raise RuntimeError(
        f"Could not fetch {url} after {MAX_ATTEMPTS} attempts"
    ) from last_error


def build_team_map(data: dict) -> dict[int, str]:
    """team id -> team name, from the bootstrap `teams` list.

    Built fresh every run because team ids and names change season to season.
    """
    return {team["id"]: team["name"] for team in data["teams"]}


def parse_players(data: dict) -> list[PlayerRow]:
    """Extract identity rows from `elements`.

    Player id is the FPL `code` (stable across seasons); the season-specific
    `element` id is carried separately as `element_id`. Rows are ordered/typed
    to feed straight into the upsert's executemany.
    """
    team_map = build_team_map(data)
    rows: list[PlayerRow] = []

    for element in data["elements"]:
        code = element["code"]
        element_id = element["id"]
        web_name = element["web_name"]

        position = POSITION_MAP.get(element["element_type"])
        if position is None:
            # FPL has experimented with non-player elements (e.g. "managers",
            # element_type 5, in 2024-25). Skip anything outside GK/DEF/MID/FWD.
            print(
                f"  skipping element {code} ({web_name}): "
                f"element_type {element['element_type']!r}",
                file=sys.stderr,
            )
            continue

        team = team_map.get(element["team"])
        if team is None:
            raise ValueError(
                f"Unknown team id {element['team']!r} "
                f"for player {code} ({web_name})"
            )

        rows.append(
            (
                code,
                element_id,
                web_name,
                element.get("first_name", ""),
                element.get("second_name", ""),
                position,
                team,
                element["now_cost"],
            )
        )

    return rows


UPSERT_PLAYER = """
INSERT INTO players (
    player_id, element_id, web_name, first_name, second_name, position, team, now_cost
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(player_id) DO UPDATE SET
    element_id   = excluded.element_id,
    web_name     = excluded.web_name,
    first_name   = excluded.first_name,
    second_name  = excluded.second_name,
    position     = excluded.position,
    team         = excluded.team,
    now_cost     = excluded.now_cost
"""


def upsert_players(rows: list[PlayerRow]) -> int:
    """Insert new players and update existing ones. Returns the row count."""
    with get_connection() as conn:
        conn.executemany(UPSERT_PLAYER, rows)
    return len(rows)


def main() -> None:
    print(f"Fetching {BOOTSTRAP_URL}")
    data = fetch_bootstrap()

    rows = parse_players(data)
    count = upsert_players(rows)

    print(f"Inserted/updated {count} players")


if __name__ == "__main__":
    main()
