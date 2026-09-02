"""SQLite schema and connection layer for the FPL predictor.

Every other part of the pipeline (ingest, features, models, optimizer,
rag, explain) reads and writes through this module so the schema lives in
exactly one place.

Player identity is keyed on the FPL **`code`** field, not the season-specific
`element` id: `code` is stable across seasons, whereas `element` ids are reused,
so keying on `element` would collide once player_gameweek_stats spans several
seasons. The current season's element id is kept alongside as `element_id`
(needed to call the element-summary endpoint). See DECISIONS.md.

The `player_features` table is not defined here: src/features/build_features.py
owns it and rewrites it wholesale each run (its columns track the feature set).
"""

import sqlite3
from pathlib import Path

# Resolve the DB path relative to the project root, not the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fpl.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database and return a connection.

    Callers are responsible for closing the connection, or using it as a
    context manager for transactions.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # SQLite does not enforce foreign key constraints by default, so we need to enable them explicitly.
    conn.execute("PRAGMA foreign_keys = ON;")
    # Rows come back as dicts instead of tuples, which is more convenient for our use case.
    conn.row_factory = sqlite3.Row
    return conn


CREATE_PLAYERS = """
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,  -- FPL `code`: stable across seasons
    element_id INTEGER,             -- current-season `element` id (NULL for past-only players)
    web_name TEXT NOT NULL,
    first_name TEXT,
    second_name TEXT,
    position TEXT NOT NULL CHECK (position IN ('GK', 'DEF', 'MID', 'FWD')),
    team TEXT NOT NULL,
    now_cost INTEGER  -- current price, tenths of a million: 75 == GBP 7.5m
);
"""

CREATE_PLAYER_GAMEWEEK_STATS = """
CREATE TABLE IF NOT EXISTS player_gameweek_stats (
    player_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL CHECK (gameweek >= 1 AND gameweek <= 38),
    total_points INTEGER,
    minutes_played INTEGER,  -- summed across a double gameweek, so may exceed 90
    goals_scored INTEGER,
    assists INTEGER,
    now_cost INTEGER,
    clean_sheets INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    was_home INTEGER,        -- first fixture of the gameweek, 0/1
    opponent_team TEXT,      -- first fixture of the gameweek
    bonus INTEGER,
    bps INTEGER,
    starts INTEGER,
    expected_goals REAL,
    expected_assists REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded REAL,
    is_double_gameweek INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (player_id, season, gameweek),
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);
"""

CREATE_NEWS_CHUNKS = """
CREATE TABLE IF NOT EXISTS news_chunks (
    chunk_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT,
    title TEXT,
    published_at TEXT,   -- ISO 8601; used for age-out
    text TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    player_id INTEGER NOT NULL,
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    raw_points REAL,
    adjusted_points REAL,
    adjustment_factor REAL,
    adjustment_reason TEXT,
    news_url TEXT,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (player_id, season, gameweek),
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);
"""

CREATE_SQUADS = """
CREATE TABLE IF NOT EXISTS squads (
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    in_squad INTEGER NOT NULL DEFAULT 0,
    in_xi INTEGER NOT NULL DEFAULT 0,
    is_captain INTEGER NOT NULL DEFAULT 0,
    is_vice INTEGER NOT NULL DEFAULT 0,
    predicted_points REAL,
    PRIMARY KEY (season, gameweek, player_id),
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);
"""

CREATE_EXPLANATIONS = """
CREATE TABLE IF NOT EXISTS explanations (
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    markdown TEXT,
    json TEXT,
    PRIMARY KEY (season, gameweek)
);
"""

CREATE_ELITE_SQUADS = """
CREATE TABLE IF NOT EXISTS elite_squads (
    season TEXT NOT NULL,
    gameweek INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    picked_pct REAL,             -- fraction of sampled elite managers who own the player
    captained_pct REAL,          -- fraction who captained them
    elite_template_score REAL,   -- the value the optimiser blends in (currently = picked_pct)
    sample_size INTEGER,
    scraped_at TEXT NOT NULL,
    PRIMARY KEY (season, gameweek, player_id),
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);
"""

ALL_CREATE_STATEMENTS = (
    CREATE_PLAYERS,
    CREATE_PLAYER_GAMEWEEK_STATS,
    CREATE_NEWS_CHUNKS,
    CREATE_PREDICTIONS,
    CREATE_SQUADS,
    CREATE_EXPLANATIONS,
    CREATE_ELITE_SQUADS,
)

# Additive columns for player_gameweek_stats, applied by migrate() to a DB that
# predates them. Keep in sync with CREATE_PLAYER_GAMEWEEK_STATS above.
_PGS_ADDITIVE_COLUMNS: dict[str, str] = {
    "was_home": "INTEGER",
    "opponent_team": "TEXT",
    "bonus": "INTEGER",
    "bps": "INTEGER",
    "starts": "INTEGER",
    "expected_goals": "REAL",
    "expected_assists": "REAL",
    "expected_goal_involvements": "REAL",
    "expected_goals_conceded": "REAL",
    "is_double_gameweek": "INTEGER NOT NULL DEFAULT 0",
}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate(db_path: Path = DB_PATH) -> None:
    """Bring an existing database up to the current schema.

    Two kinds of change are handled:
      * additive columns on player_gameweek_stats -> ALTER TABLE ADD COLUMN;
      * the players re-key (element id -> FPL code) -> the pre-code players table
        is dropped and recreated. player_gameweek_stats is empty whenever this
        applies, so no gameweek data is lost; players must be re-fetched with
        `python src/ingest/fetch_fpl.py` afterwards.
    """
    conn = get_connection(db_path)
    # PRAGMA foreign_keys is a no-op inside a transaction; set it before any DML.
    conn.execute("PRAGMA foreign_keys = OFF;")
    try:
        if _table_exists(conn, "players"):
            # The pre-re-key players table has no `element_id` column (player_id
            # was the season-specific element id, not the stable FPL `code`).
            if "element_id" not in _columns(conn, "players"):
                print(
                    "migrate: dropping pre-re-key players table; "
                    "re-run `python src/ingest/fetch_fpl.py` to repopulate it"
                )
                conn.execute("DROP TABLE players")

        if _table_exists(conn, "player_gameweek_stats"):
            existing = _columns(conn, "player_gameweek_stats")
            for column, decl in _PGS_ADDITIVE_COLUMNS.items():
                if column not in existing:
                    print(f"migrate: adding player_gameweek_stats.{column}")
                    conn.execute(
                        f"ALTER TABLE player_gameweek_stats ADD COLUMN {column} {decl}"
                    )

        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create every table if it does not exist, then apply pending migrations."""
    with get_connection(db_path) as conn:
        for statement in ALL_CREATE_STATEMENTS:
            conn.execute(statement)
    migrate(db_path)
    # migrate() may have dropped `players`; recreate it so the schema is complete.
    with get_connection(db_path) as conn:
        conn.execute(CREATE_PLAYERS)


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
