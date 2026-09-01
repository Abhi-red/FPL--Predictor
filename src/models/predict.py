"""Predict raw points for the next unplayed gameweek, for every current player.

Loads the per-position models from data/models/, builds a feature row for each
player as of the upcoming gameweek (their real rolling history + the upcoming
fixture's home/away and difficulty), and writes:

  * table ``predictions`` — raw_points filled, adjusted_* left for the RAG step
  * ``data/predictions_raw.json`` — for quick inspection / the site

Run (after `python src/models/train.py`):
    python src/models/predict.py
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import FPL_BASE, FPL_HEADERS, REQUEST_TIMEOUT, SEASON  # noqa: E402
from db import get_connection  # noqa: E402
from features.build_features import (  # noqa: E402
    add_features,
    feature_columns,
    load_stats_frame,
    strengths_by_season,
)

_META_COLS = ["player_id", "season", "gameweek", "position", "team", "web_name"]

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
PREDICTIONS_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "predictions_raw.json"
POSITIONS = ("GK", "DEF", "MID", "FWD")

UPSERT_PREDICTION = """
INSERT INTO predictions (player_id, season, gameweek, raw_points, generated_at)
VALUES (:player_id, :season, :gameweek, :raw_points, :generated_at)
ON CONFLICT(player_id, season, gameweek) DO UPDATE SET
    raw_points   = excluded.raw_points,
    generated_at = excluded.generated_at
"""


def _bootstrap() -> dict:
    resp = requests.get(
        f"{FPL_BASE}/bootstrap-static/", headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def next_gameweek(bootstrap: dict) -> int:
    """First event that isn't finished; fall back to the one flagged `is_next`."""
    for event in bootstrap["events"]:
        if not event["finished"]:
            return int(event["id"])
    for event in bootstrap["events"]:
        if event.get("is_next"):
            return int(event["id"])
    raise RuntimeError("Could not determine the next gameweek from bootstrap events")


def upcoming_fixtures(gameweek: int, bootstrap: dict) -> dict[str, tuple[int, str]]:
    """team name -> (was_home, opponent name) for `gameweek`; first fixture on a double."""
    team_name = {t["id"]: t["name"] for t in bootstrap["teams"]}
    resp = requests.get(
        f"{FPL_BASE}/fixtures/",
        params={"event": gameweek},
        headers=FPL_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    out: dict[str, tuple[int, str]] = {}
    for fixture in sorted(resp.json(), key=lambda f: f.get("kickoff_time") or ""):
        home, away = team_name.get(fixture["team_h"]), team_name.get(fixture["team_a"])
        if home and home not in out:
            out[home] = (1, away)
        if away and away not in out:
            out[away] = (0, home)
    return out


def build_upcoming_matrix(gameweek: int, bootstrap: dict) -> pd.DataFrame:
    """One feature row per current player for `gameweek` (SEASON)."""
    stats = load_stats_frame()
    fixtures = upcoming_fixtures(gameweek, bootstrap)

    with get_connection() as conn:
        players = pd.read_sql_query(
            "SELECT player_id, web_name, position, team, now_cost "
            "FROM players WHERE element_id IS NOT NULL",
            conn,
        )

    synthetic_rows: list[dict] = []
    for player in players.itertuples(index=False):
        fixture = fixtures.get(player.team)
        if fixture is None:
            continue  # blank gameweek for this player's club
        was_home, opponent = fixture
        row = {col: np.nan for col in stats.columns}
        row.update(
            player_id=player.player_id,
            season=SEASON,
            gameweek=gameweek,
            now_cost=player.now_cost,
            was_home=was_home,
            opponent_team=opponent,
            is_double_gameweek=0,
            position=player.position,
            team=player.team,
            web_name=player.web_name,
        )
        synthetic_rows.append(row)

    synthetic = pd.DataFrame(synthetic_rows, columns=list(stats.columns))
    combined = (
        pd.concat([stats, synthetic], ignore_index=True)
        .sort_values(["player_id", "season", "gameweek"])
        .reset_index(drop=True)
    )
    strengths = strengths_by_season(sorted(combined["season"].unique()))
    enriched = add_features(combined, strengths)
    return enriched[
        (enriched["season"] == SEASON) & (enriched["gameweek"] == gameweek)
    ].copy()


def load_models() -> dict:
    import joblib

    models = {}
    for position in POSITIONS:
        path = MODELS_DIR / f"{position}.joblib"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run `python src/models/train.py` first."
            )
        models[position] = joblib.load(path)
    return models


def main() -> None:
    bootstrap = _bootstrap()
    gameweek = next_gameweek(bootstrap)
    print(f"Predicting {SEASON} GW{gameweek}")

    models = load_models()
    cols = feature_columns()
    matrix = build_upcoming_matrix(gameweek, bootstrap)

    matrix["raw_points"] = np.nan
    for position in POSITIONS:
        mask = matrix["position"] == position
        if not mask.any():
            continue
        preds = models[position].predict(matrix.loc[mask, cols])
        matrix.loc[mask, "raw_points"] = np.clip(preds, 0, None)

    matrix["raw_points"] = matrix["raw_points"].round(3)
    generated_at = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "player_id": int(r.player_id),
            "season": SEASON,
            "gameweek": gameweek,
            "raw_points": float(r.raw_points),
            "generated_at": generated_at,
        }
        for r in matrix.itertuples(index=False)
        if pd.notna(r.raw_points)
    ]

    with get_connection() as conn:
        conn.executemany(UPSERT_PREDICTION, records)

    # Persist the upcoming-gameweek feature rows so the site can show the form
    # numbers behind each prediction. build_features rewrites this table wholesale
    # on its next run, so these rows never reach training.
    persist_cols = [c for c in _META_COLS + cols + ["target_points"] if c in matrix.columns]
    upcoming = matrix[persist_cols].copy()
    if "target_points" not in upcoming.columns:
        upcoming["target_points"] = np.nan
    with get_connection() as conn:
        try:
            conn.execute(
                "DELETE FROM player_features WHERE season = ? AND gameweek = ?",
                (SEASON, gameweek),
            )
        except sqlite3.OperationalError:
            pass  # table not created yet; to_sql will create it
        upcoming.to_sql("player_features", conn, if_exists="append", index=False)

    export = (
        matrix[["player_id", "web_name", "position", "team", "price", "raw_points"]]
        .rename(columns={"price": "now_cost"})
        .sort_values("raw_points", ascending=False)
    )
    PREDICTIONS_JSON.write_text(
        json.dumps(
            {
                "season": SEASON,
                "gameweek": gameweek,
                "generated_at": generated_at,
                "players": export.to_dict("records"),
            },
            indent=2,
            default=lambda v: None if pd.isna(v) else float(v),
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(records)} predictions; top pick "
        f"{export.iloc[0]['web_name']} ({export.iloc[0]['raw_points']:.2f})"
    )


if __name__ == "__main__":
    main()
