"""One-time backfill of completed seasons from the vaastav/Fantasy-Premier-League repo.

For each season in HISTORICAL_SEASONS we pull:
  * players_raw.csv  -> element id <-> stable `code`, plus identity fields
  * teams.csv        -> team id -> team name (for that season)
  * gws/merged_gw.csv -> one row per player per fixture (fallback: gws/gw{n}.csv)

vaastav's column names mostly match the live element-summary feed already; the
handful that need interpreting are spelled out in COLUMN_MAP below rather than
mapped implicitly. Players are matched across seasons by `code`, never by the
season-specific `element` id (a player who changed clubs keeps one `code`).

Run once (slow, ~5 CSV downloads per season):
    python src/ingest/backfill_historical.py
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.aggregate import aggregate_history  # noqa: E402
from constants import (  # noqa: E402
    FPL_HEADERS,
    HISTORICAL_SEASONS,
    MAX_ATTEMPTS,
    POSITION_MAP,
    REQUEST_TIMEOUT,
    VAASTAV_BASE,
)
from db import get_connection  # noqa: E402

# vaastav merged_gw column -> how we use it. The value is the FPL-style key that
# aggregate_history() reads (so historical rows look like element-summary rows).
# Columns marked "(from <season>)" are absent in older seasons and become NULL.
COLUMN_MAP: dict[str, str] = {
    "round": "round",                    # gameweek number
    "total_points": "total_points",
    "minutes": "minutes",                # -> minutes_played, summed, uncapped
    "goals_scored": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "bonus": "bonus",
    "bps": "bps",
    "starts": "starts",                  # (from 2022-23)
    "value": "value",                    # -> now_cost (tenths of a million)
    "was_home": "was_home",              # CSV string "True"/"False"
    "opponent_team": "opponent_team",    # team id, resolved via teams.csv
    "kickoff_time": "kickoff_time",      # fixture ordering within a gameweek
    "fixture": "fixture",
    "element": "element",                # season-specific id, mapped to `code`
    "expected_goals": "expected_goals",                          # (from 2022-23)
    "expected_assists": "expected_assists",                      # (from 2022-23)
    "expected_goal_involvements": "expected_goal_involvements",  # (from 2022-23)
    "expected_goals_conceded": "expected_goals_conceded",        # (from 2022-23)
}

UPSERT_PLAYER_IDENTITY = """
INSERT INTO players (
    player_id, element_id, web_name, first_name, second_name, position, team, now_cost
)
VALUES (:code, NULL, :web_name, :first_name, :second_name, :position, :team, :now_cost)
ON CONFLICT(player_id) DO UPDATE SET
    web_name    = excluded.web_name,
    first_name  = excluded.first_name,
    second_name = excluded.second_name,
    position    = excluded.position,
    team        = excluded.team
WHERE players.element_id IS NULL  -- never overwrite a current-season player
"""

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


def _fetch_csv(url: str) -> pd.DataFrame | None:
    """Download a CSV with retry; return None on a persistent 404 (optional file)."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            try:
                text = response.content.decode("utf-8")
            except UnicodeDecodeError:
                text = response.content.decode("latin-1")
            return pd.read_csv(io.StringIO(text))
        except requests.RequestException as error:
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


def _load_merged_gw(season: str) -> pd.DataFrame:
    """gws/merged_gw.csv, or gw1..gw38 concatenated if the merged file is absent."""
    merged = _fetch_csv(f"{VAASTAV_BASE}/{season}/gws/merged_gw.csv")
    if merged is not None and len(merged) > 0:
        return merged

    print(f"  {season}: merged_gw.csv unavailable, falling back to per-gameweek files")
    frames: list[pd.DataFrame] = []
    for gameweek in range(1, 39):
        frame = _fetch_csv(f"{VAASTAV_BASE}/{season}/gws/gw{gameweek}.csv")
        if frame is not None and len(frame) > 0:
            if "round" not in frame.columns:
                frame["round"] = gameweek
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No gameweek data found for {season}")
    return pd.concat(frames, ignore_index=True)


def _identity_maps(players_raw: pd.DataFrame, teams: dict[int, str]) -> tuple[
    dict[int, int], dict[int, dict]
]:
    """Return (element id -> code) and (code -> identity dict) for a season."""
    elem_to_code: dict[int, int] = {}
    code_to_identity: dict[int, dict] = {}
    for row in players_raw.itertuples(index=False):
        element_type = int(row.element_type)
        if element_type not in POSITION_MAP:
            continue  # e.g. "manager" elements (type 5), only in 2024-25
        code = int(row.code)
        elem_to_code[int(row.id)] = code
        code_to_identity[code] = {
            "code": code,
            "web_name": getattr(row, "web_name", "") or "",
            "first_name": getattr(row, "first_name", "") or "",
            "second_name": getattr(row, "second_name", "") or "",
            "position": POSITION_MAP[element_type],
            "team": teams.get(int(row.team), str(row.team)),
            "now_cost": int(getattr(row, "now_cost", 0) or 0),
        }
    return elem_to_code, code_to_identity


def _normalise_history_row(csv_row: dict) -> dict:
    """Project a vaastav gameweek row onto the FPL-style keys aggregate_history reads."""
    return {dest: csv_row.get(src) for src, dest in COLUMN_MAP.items()}


def backfill_season(season: str) -> tuple[int, int]:
    """Load one season into players + player_gameweek_stats. Returns (players, rows)."""
    print(f"Season {season}: fetching source CSVs")
    players_raw = _fetch_csv(f"{VAASTAV_BASE}/{season}/players_raw.csv")
    if players_raw is None:
        raise RuntimeError(f"{season}: players_raw.csv not found")

    teams_csv = _fetch_csv(f"{VAASTAV_BASE}/{season}/teams.csv")
    if teams_csv is None:
        raise RuntimeError(f"{season}: teams.csv not found")
    teams = {int(r.id): r.name for r in teams_csv.itertuples(index=False)}

    elem_to_code, code_to_identity = _identity_maps(players_raw, teams)
    merged = _load_merged_gw(season)

    unused = sorted(set(merged.columns) - set(COLUMN_MAP) - {"GW", "name", "position", "team"})
    if unused:
        print(f"  {season}: ignoring unmapped columns: {', '.join(unused)}")

    # Group fixture rows by code, dropping rows whose element id is unknown.
    rows_by_code: dict[int, list[dict]] = {}
    dropped = 0
    for csv_row in merged.to_dict("records"):
        element = csv_row.get("element")
        if element is None or pd.isna(element) or int(element) not in elem_to_code:
            dropped += 1
            continue
        code = elem_to_code[int(element)]
        rows_by_code.setdefault(code, []).append(_normalise_history_row(csv_row))
    if dropped:
        print(f"  {season}: dropped {dropped} rows with an unknown element id")

    gameweek_rows: list[dict] = []
    identity_rows: list[dict] = []
    for code, history in rows_by_code.items():
        identity = code_to_identity.get(code)
        if identity is None:
            continue
        identity_rows.append(identity)
        for out in aggregate_history(history, teams):
            out["player_id"] = code
            out["season"] = season
            # aggregate_history omits keys the source lacked; fill for the upsert.
            for key in (
                "starts",
                "expected_goals",
                "expected_assists",
                "expected_goal_involvements",
                "expected_goals_conceded",
            ):
                out.setdefault(key, None)
            gameweek_rows.append(out)

    with get_connection() as conn:
        conn.executemany(UPSERT_PLAYER_IDENTITY, identity_rows)
        conn.executemany(UPSERT_GAMEWEEK_STATS, gameweek_rows)

    print(
        f"  {season}: {len(identity_rows)} players, {len(gameweek_rows)} gameweek rows"
    )
    return len(identity_rows), len(gameweek_rows)


def main() -> None:
    total_players = 0
    total_rows = 0
    for season in HISTORICAL_SEASONS:
        players, rows = backfill_season(season)
        total_players += players
        total_rows += rows
    print(
        f"Backfill complete: {total_rows} gameweek rows across "
        f"{len(HISTORICAL_SEASONS)} seasons ({total_players} player-seasons)"
    )


if __name__ == "__main__":
    main()
