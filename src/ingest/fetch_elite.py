"""Sample elite-manager ownership from the top of FPL's overall league.

For a gameweek we take the top ELITE_SAMPLE_SIZE entries of league
ELITE_LEAGUE_ID (the global "Overall" league), pull each entry's 15 picks, and
store per player:

  picked_pct           fraction of the sample that owns the player
  captained_pct        fraction that captained them
  elite_template_score the value the optimiser blends in (currently = picked_pct)

This only works going forward: the picks endpoint covers the *current* season's
finished (and in-progress) gameweeks, and there is no elite history for past
seasons. Each run also backfills any earlier finished gameweek of the current
season that has no elite row yet, so the tunable history grows week by week.

Run:
    python src/ingest/fetch_elite.py             # target gw + backfill gaps
    python src/ingest/fetch_elite.py --gw 5      # only gameweek 5
    python src/ingest/fetch_elite.py --backfill  # only fill gaps, skip target
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    ELITE_LEAGUE_ID,
    ELITE_SAMPLE_SIZE,
    FPL_BASE,
    FPL_HEADERS,
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT,
    SEASON,
)
from db import get_connection  # noqa: E402

STANDINGS_URL = f"{FPL_BASE}/leagues-classic/{{league}}/standings/"
PICKS_URL = f"{FPL_BASE}/entry/{{entry}}/event/{{gw}}/picks/"
BOOTSTRAP_URL = f"{FPL_BASE}/bootstrap-static/"
INTER_REQUEST_DELAY = 0.15

UPSERT_ELITE = """
INSERT INTO elite_squads (
    season, gameweek, player_id, picked_pct, captained_pct,
    elite_template_score, sample_size, scraped_at
)
VALUES (:season, :gameweek, :player_id, :picked_pct, :captained_pct,
        :elite_template_score, :sample_size, :scraped_at)
ON CONFLICT(season, gameweek, player_id) DO UPDATE SET
    picked_pct           = excluded.picked_pct,
    captained_pct        = excluded.captained_pct,
    elite_template_score = excluded.elite_template_score,
    sample_size          = excluded.sample_size,
    scraped_at           = excluded.scraped_at
"""


def _get_json(url: str, params: dict | None = None) -> dict | None:
    """GET with retry/backoff. Returns None on a definitive 404."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url, params=params, headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch {url} after {MAX_ATTEMPTS} attempts") from last_error


def top_entries(sample_size: int) -> list[int]:
    """Entry ids for the top `sample_size` managers of the overall league."""
    entries: list[int] = []
    page = 1
    while len(entries) < sample_size:
        data = _get_json(STANDINGS_URL.format(league=ELITE_LEAGUE_ID), {"page_standings": page})
        results = (data or {}).get("standings", {}).get("results", [])
        if not results:
            break
        entries.extend(row["entry"] for row in results)
        if not (data or {}).get("standings", {}).get("has_next"):
            break
        page += 1
        time.sleep(INTER_REQUEST_DELAY)
    return entries[:sample_size]


def element_to_code() -> dict[int, int]:
    with get_connection() as conn:
        return {
            row["element_id"]: row["player_id"]
            for row in conn.execute(
                "SELECT element_id, player_id FROM players WHERE element_id IS NOT NULL"
            )
        }


def aggregate_gameweek(gameweek: int, entries: list[int], code_of: dict[int, int]) -> list[dict]:
    """Tally picks across the sampled entries for one gameweek."""
    picked: dict[int, int] = {}
    captained: dict[int, int] = {}
    counted = 0

    for index, entry in enumerate(entries, start=1):
        data = _get_json(PICKS_URL.format(entry=entry, gw=gameweek))
        time.sleep(INTER_REQUEST_DELAY)
        if not data or "picks" not in data:
            continue
        counted += 1
        for pick in data["picks"]:
            code = code_of.get(pick["element"])
            if code is None:
                continue
            picked[code] = picked.get(code, 0) + 1
            if pick.get("is_captain"):
                captained[code] = captained.get(code, 0) + 1
        if index % 50 == 0:
            print(f"  gw{gameweek}: {index}/{len(entries)} entries", flush=True)

    if counted == 0:
        return []

    scraped_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for code, count in picked.items():
        picked_pct = count / counted
        rows.append(
            {
                "season": SEASON,
                "gameweek": gameweek,
                "player_id": code,
                "picked_pct": round(picked_pct, 4),
                "captained_pct": round(captained.get(code, 0) / counted, 4),
                "elite_template_score": round(picked_pct, 4),
                "sample_size": counted,
                "scraped_at": scraped_at,
            }
        )
    return rows


def existing_elite_gameweeks() -> set[int]:
    with get_connection() as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT gameweek FROM elite_squads WHERE season = ?", (SEASON,)
            )
        }


def finished_and_current_gameweeks(bootstrap: dict) -> tuple[list[int], int | None]:
    finished = [e["id"] for e in bootstrap["events"] if e["finished"]]
    upcoming = next((e["id"] for e in bootstrap["events"] if not e["finished"]), None)
    return finished, upcoming


def scrape_gameweek(gameweek: int, entries: list[int], code_of: dict[int, int]) -> int:
    rows = aggregate_gameweek(gameweek, entries, code_of)
    if not rows:
        print(f"  gw{gameweek}: no picks available yet, skipping", file=sys.stderr)
        return 0
    with get_connection() as conn:
        conn.executemany(UPSERT_ELITE, rows)
    print(f"  gw{gameweek}: stored {len(rows)} players from {rows[0]['sample_size']} elite managers")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gw", type=int, help="scrape only this gameweek")
    parser.add_argument("--backfill", action="store_true", help="only fill missing finished gameweeks")
    args = parser.parse_args()

    bootstrap = _get_json(BOOTSTRAP_URL)
    finished, upcoming = finished_and_current_gameweeks(bootstrap)
    code_of = element_to_code()

    if args.gw:
        targets = [args.gw]
    else:
        have = existing_elite_gameweeks()
        targets = [gw for gw in finished if gw not in have]
        if not args.backfill and upcoming is not None:
            targets.append(upcoming)

    if not targets:
        print("Elite ownership already up to date; nothing to scrape")
        return

    print(
        f"Sampling top {ELITE_SAMPLE_SIZE} of league {ELITE_LEAGUE_ID} "
        f"for {SEASON} gameweeks {targets}"
    )
    entries = top_entries(ELITE_SAMPLE_SIZE)
    print(f"Got {len(entries)} elite entry ids")

    total = 0
    for gameweek in targets:
        total += scrape_gameweek(gameweek, entries, code_of)
    print(f"Done: {total} elite ownership rows across {len(targets)} gameweeks")


if __name__ == "__main__":
    main()
