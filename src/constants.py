"""Project-wide constants shared across ingest, features, models, and optimizer.

Keep this file dependency-free (stdlib only) so anything can import it cheaply.
"""

# FPL's `element_type` codes -> the position strings our schema's CHECK allows.
# The bootstrap-static response also carries this mapping in `element_types`,
# but it uses "GKP" rather than "GK"; these are the canonical values for the
# whole project (per-position models, ILP formation rules, tests).
POSITION_MAP: dict[int, str] = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

# The season currently being played, in the vaastav/Fantasy-Premier-League
# directory format ("2026-27"). Every row we write to player_gameweek_stats is
# tagged with a season string; this is the tag for the live FPL API data.
# (As of 2026-09-01 the live API is on 2026-27, GW2 complete.)
SEASON: str = "2026-27"

# Completed seasons pulled once by src/ingest/backfill_historical.py. Ordered
# oldest -> newest. vaastav keeps a directory per season under `data/`.
HISTORICAL_SEASONS: tuple[str, ...] = (
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

# Shared HTTP config. Some CDNs reject the default "python-requests/x.y" agent,
# so every ingester identifies itself with the same UA.
FPL_HEADERS: dict[str, str] = {
    "User-Agent": "fpl-predictor (personal learning project)"
}
REQUEST_TIMEOUT: tuple[int, int] = (5, 30)  # (connect, read) seconds
MAX_ATTEMPTS: int = 3

# Data source base URLs.
FPL_BASE: str = "https://fantasy.premierleague.com/api"
VAASTAV_BASE: str = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# RAG layer. A small, fast local sentence-transformers model; 384-dim vectors.
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
NEWS_MAX_AGE_DAYS: int = 21  # chunks older than this age out of the FAISS index

# Explanation layer. Model id is user-specified; the Anthropic SDK reads the key
# from the ANTHROPIC_API_KEY environment variable (never hardcode it).
EXPLANATION_MODEL: str = "claude-sonnet-4-6"

# RAG prediction adjustment is bounded: a news signal can move a raw prediction
# by at most this fraction, never replace it outright.
ADJUSTMENT_CAP: float = 0.30

# Elite-template blending. The optimiser scores a player as
#   final_score = predicted_points * (1 + ELITE_WEIGHT * elite_template_score)
# ELITE_WEIGHT is NOT hardcoded: src/optimize/tune_elite_weight.py sweeps these
# candidates by walk-forward backtest and writes the winner to
# data/elite_weight_config.json, which the optimiser reads at run time. 0.0 is
# the mandatory control (pure stats, no elite signal) that every other weight is
# measured against. See DECISIONS.md.
ELITE_WEIGHT_CANDIDATES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)

# The sweep needs gameweeks that have BOTH elite ownership data and realized
# results. Below this many, the tuner defers to ELITE_WEIGHT = 0.0 rather than
# pretend it is tuned.
ELITE_TUNE_MIN_GAMEWEEKS: int = 5

# A candidate only beats the 0.0 control if it is ahead by more than this many
# realized points per gameweek; within this band, prefer the simpler weight.
ELITE_TUNE_NOISE_PTS: float = 0.25

# Elite ownership is sampled from the top of FPL's overall league (id 314).
ELITE_LEAGUE_ID: int = 314
ELITE_SAMPLE_SIZE: int = 100

# FPL squad rules, used by the ILP optimizer and its tests.
BUDGET: int = 1000  # total squad cost ceiling, in tenths of a million (£100.0m)
SQUAD: dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}  # 15-man squad
MAX_PER_CLUB: int = 3
# Valid starting-XI counts per position (min, max); always exactly 11 total.
XI_BOUNDS: dict[str, tuple[int, int]] = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}

# Retrain the production models when the saved manifest is older than this.
MODEL_MAX_AGE_DAYS: int = 30
