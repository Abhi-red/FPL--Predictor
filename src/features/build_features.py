"""Turn player_gameweek_stats into a model-ready feature matrix.

One row per player per gameweek. Every rolling / trend feature is computed from
gameweeks strictly BEFORE the row's gameweek (``.shift(1)`` before ``.rolling``)
so a row never sees its own or any future result — the same no-leakage rule the
walk-forward backtest relies on. The row's ``target_points`` is that gameweek's
actual ``total_points``.

Outputs (all under data/):
  * table ``player_features``      — rewritten wholesale each run
  * ``features.parquet``           — same frame, for train / predict to share
  * ``models/feature_columns.json``— the ordered model input columns

Run:
    python src/features/build_features.py
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    FPL_BASE,
    FPL_HEADERS,
    HISTORICAL_SEASONS,
    REQUEST_TIMEOUT,
    SEASON,
    VAASTAV_BASE,
)
from db import get_connection  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
FEATURE_COLUMNS_JSON = DATA_DIR / "models" / "feature_columns.json"

# Stats we take rolling means of, over each window.
ROLL_STATS: tuple[str, ...] = (
    "total_points",
    "minutes_played",
    "goals_scored",
    "assists",
    "bps",
    "bonus",
    "expected_goals",
    "expected_assists",
)
ROLL_WINDOWS: tuple[int, ...] = (3, 5, 10)

# Non-rolling engineered columns, plus the passthrough `was_home`.
_EXTRA_FEATURES: tuple[str, ...] = (
    "start_rate_5",
    "form_ewm",
    "price",
    "price_trend_3",
    "points_per_90_5",
    "gw_gap",
    "ownership",
    "ownership_trend_3",
    "fdr",
    "was_home",
)


def feature_columns() -> list[str]:
    """The ordered list of model input columns (matches feature_columns.json)."""
    cols = [f"roll{w}_{stat}" for stat in ROLL_STATS for w in ROLL_WINDOWS]
    cols.extend(_EXTRA_FEATURES)
    return cols


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_stats_frame() -> pd.DataFrame:
    """player_gameweek_stats joined to player position/team, ordered for rolling."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT s.*, p.position, p.team, p.web_name
            FROM player_gameweek_stats s
            JOIN players p ON p.player_id = s.player_id
            """,
            conn,
        )
    if df.empty:
        return df
    df = df.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Fixture difficulty
# --------------------------------------------------------------------------- #
def _scaled_strengths(rows: list[tuple[str, float, float]]) -> dict[str, dict[str, float]]:
    """Map team -> {'h','a'} difficulty on a 1..5 scale, from raw strength numbers."""
    if not rows:
        return {}
    values = [v for _, h, a in rows for v in (h, a) if v and not np.isnan(v)]
    if not values:
        return {}
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0

    def scale(v: float) -> float:
        if not v or np.isnan(v):
            return 3.0
        return round(1.0 + 4.0 * (v - lo) / span, 3)

    return {team: {"h": scale(h), "a": scale(a)} for team, h, a in rows}


def _current_season_strengths() -> dict[str, dict[str, float]]:
    resp = requests.get(
        f"{FPL_BASE}/bootstrap-static/", headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    teams = resp.json()["teams"]
    rows = [
        (t["name"], float(t["strength_overall_home"]), float(t["strength_overall_away"]))
        for t in teams
    ]
    return _scaled_strengths(rows)


def _historical_season_strengths(season: str) -> dict[str, dict[str, float]]:
    resp = requests.get(
        f"{VAASTAV_BASE}/{season}/teams.csv", headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    teams = pd.read_csv(io.StringIO(resp.content.decode("utf-8", "replace")))
    rows = [
        (
            r["name"],
            float(r.get("strength_overall_home", np.nan)),
            float(r.get("strength_overall_away", np.nan)),
        )
        for _, r in teams.iterrows()
    ]
    return _scaled_strengths(rows)


def strengths_by_season(seasons: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Team difficulty tables per season; empty dict for any season we can't fetch."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for season in seasons:
        try:
            out[season] = (
                _current_season_strengths()
                if season == SEASON
                else _historical_season_strengths(season)
            )
        except Exception as error:  # noqa: BLE001 - difficulty is best-effort
            print(
                f"  fixture difficulty unavailable for {season} ({error}); "
                f"using neutral 3.0",
                file=sys.stderr,
            )
            out[season] = {}
    return out


def _fdr_column(df: pd.DataFrame, strengths: dict) -> pd.Series:
    def lookup(row: pd.Series) -> float:
        table = strengths.get(row["season"]) or {}
        entry = table.get(row["opponent_team"])
        if not entry:
            return 3.0
        # Opponent plays away when our player is at home, and vice versa.
        return entry["a"] if row["was_home"] == 1 else entry["h"]

    return df.apply(lookup, axis=1)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _player_block(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values(["season", "gameweek"])
    out: dict[str, pd.Series] = {}

    for stat in ROLL_STATS:
        prior = grp[stat].shift(1)
        for window in ROLL_WINDOWS:
            out[f"roll{window}_{stat}"] = prior.rolling(window, min_periods=1).mean()

    started = (grp["minutes_played"].fillna(0) >= 60).astype(float)
    out["start_rate_5"] = started.shift(1).rolling(5, min_periods=1).mean()
    out["form_ewm"] = grp["total_points"].shift(1).ewm(span=4).mean()

    out["price"] = grp["now_cost"].astype("float64")
    out["price_trend_3"] = grp["now_cost"] - grp["now_cost"].shift(3)

    pts5 = grp["total_points"].shift(1).rolling(5, min_periods=1).sum()
    mins5 = grp["minutes_played"].shift(1).rolling(5, min_periods=1).sum()
    out["points_per_90_5"] = pts5 / (mins5 / 90.0).replace(0.0, np.nan)

    out["gw_gap"] = grp.groupby("season")["gameweek"].diff().fillna(1.0)

    if "selected_by_percent" in grp.columns:
        own = grp["selected_by_percent"].astype("float64")
        out["ownership"] = own.shift(1)
        out["ownership_trend_3"] = own.shift(1) - own.shift(4)
    else:
        out["ownership"] = pd.Series(np.nan, index=grp.index)
        out["ownership_trend_3"] = pd.Series(np.nan, index=grp.index)

    return pd.DataFrame(out, index=grp.index)


def add_features(df: pd.DataFrame, strengths: dict) -> pd.DataFrame:
    """Attach every engineered feature column plus target_points to `df`."""
    blocks = df.groupby("player_id", group_keys=False).apply(_player_block)
    df = df.join(blocks)
    df["fdr"] = _fdr_column(df, strengths)
    df["was_home"] = df["was_home"].fillna(0).astype("int64")
    df["target_points"] = df["total_points"].astype("float64")
    return df


# --------------------------------------------------------------------------- #
# Build / persist
# --------------------------------------------------------------------------- #
_KEEP_META = ["player_id", "season", "gameweek", "position", "team", "web_name"]


def build() -> pd.DataFrame:
    stats = load_stats_frame()
    if stats.empty:
        raise RuntimeError(
            "player_gameweek_stats is empty — run the ingest steps first"
        )

    seasons = sorted(stats["season"].unique())
    strengths = strengths_by_season(seasons)
    enriched = add_features(stats, strengths)

    cols = _KEEP_META + feature_columns() + ["target_points"]
    matrix = enriched[cols].copy()

    FEATURE_COLUMNS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_JSON.write_text(
        json.dumps(feature_columns(), indent=2), encoding="utf-8"
    )
    matrix.to_parquet(FEATURES_PARQUET, index=False)
    with get_connection() as conn:
        matrix.to_sql("player_features", conn, if_exists="replace", index=False)

    print(
        f"Built {len(matrix)} feature rows across {len(seasons)} seasons, "
        f"{len(feature_columns())} feature columns -> {FEATURES_PARQUET.name}, "
        f"table player_features"
    )
    return matrix


def get_feature_frame(rebuild: bool = False) -> pd.DataFrame:
    """Return the feature matrix, building it if the parquet is missing/stale."""
    if rebuild or not FEATURES_PARQUET.exists():
        return build()
    return pd.read_parquet(FEATURES_PARQUET)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
