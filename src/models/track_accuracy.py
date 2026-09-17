"""Score past predictions against realized results, once those results exist.

Every prediction we ever made is already frozen in ``predictions`` (predict.py
upserts by (player_id, season, gameweek), and never re-predicts a gameweek once
it has passed). Once fetch_gameweek_stats pulls in that gameweek's actual
``total_points``, we can join the two and see how far off we were - separately
for the raw model output and the news-adjusted one, per position.

Writes:
  * table ``prediction_accuracy``       — one row per (season, gameweek, position),
                                          plus an 'ALL' row per gameweek
  * ``data/models/prediction_accuracy.json`` — the same data, plus an overall
                                          rollup across every scored gameweek

Run (after fetch_gameweek_stats has pulled the latest results):
    python src/models/track_accuracy.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_connection  # noqa: E402

ACCURACY_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "prediction_accuracy.json"
POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")

SELECT_SCORED = """
SELECT pr.season, pr.gameweek, p.position, pr.raw_points, pr.adjusted_points,
       s.total_points AS actual
FROM predictions pr
JOIN players p ON p.player_id = pr.player_id
JOIN player_gameweek_stats s
       ON s.player_id = pr.player_id AND s.season = pr.season AND s.gameweek = pr.gameweek
WHERE s.total_points IS NOT NULL AND pr.raw_points IS NOT NULL
"""

UPSERT_ACCURACY = """
INSERT INTO prediction_accuracy (
    season, gameweek, position, n, mae_raw, rmse_raw, mae_adjusted, rmse_adjusted, computed_at
) VALUES (
    :season, :gameweek, :position, :n, :mae_raw, :rmse_raw, :mae_adjusted, :rmse_adjusted, :computed_at
)
ON CONFLICT(season, gameweek, position) DO UPDATE SET
    n              = excluded.n,
    mae_raw        = excluded.mae_raw,
    rmse_raw       = excluded.rmse_raw,
    mae_adjusted   = excluded.mae_adjusted,
    rmse_adjusted  = excluded.rmse_adjusted,
    computed_at    = excluded.computed_at
"""


# --------------------------------------------------------------------------- #
# Pure computation (unit-tested)
# --------------------------------------------------------------------------- #
def _metrics(errors: np.ndarray) -> dict[str, float | int | None]:
    if errors.size == 0:
        return {"n": 0, "mae": None, "rmse": None}
    return {
        "n": int(errors.size),
        "mae": round(float(np.abs(errors).mean()), 4),
        "rmse": round(float(np.sqrt((errors**2).mean())), 4),
    }


def compute_accuracy(rows: list[dict]) -> list[dict]:
    """Per-position (+ 'ALL') MAE/RMSE, for both raw and adjusted predictions.

    Each row needs: position, raw_points, adjusted_points (may be None, falls
    back to raw_points), actual. Positions absent from `rows` are omitted.
    """
    out: list[dict] = []
    for position in (*POSITIONS, "ALL"):
        subset = rows if position == "ALL" else [r for r in rows if r["position"] == position]
        if not subset:
            continue
        actual = np.array([r["actual"] for r in subset], dtype=float)
        raw = np.array([r["raw_points"] for r in subset], dtype=float)
        adjusted = np.array(
            [r["adjusted_points"] if r["adjusted_points"] is not None else r["raw_points"] for r in subset],
            dtype=float,
        )
        raw_metrics = _metrics(raw - actual)
        adjusted_metrics = _metrics(adjusted - actual)
        out.append(
            {
                "position": position,
                "n": raw_metrics["n"],
                "mae_raw": raw_metrics["mae"],
                "rmse_raw": raw_metrics["rmse"],
                "mae_adjusted": adjusted_metrics["mae"],
                "rmse_adjusted": adjusted_metrics["rmse"],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# DB I/O
# --------------------------------------------------------------------------- #
def load_scored_rows() -> dict[tuple[str, int], list[dict]]:
    """Every prediction whose gameweek now has a realized result, grouped by (season, gameweek)."""
    with get_connection() as conn:
        rows = conn.execute(SELECT_SCORED).fetchall()
    grouped: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["season"], r["gameweek"]), []).append(
            {
                "position": r["position"],
                "raw_points": r["raw_points"],
                "adjusted_points": r["adjusted_points"],
                "actual": float(r["actual"]),
            }
        )
    return grouped


def main() -> None:
    grouped = load_scored_rows()
    if not grouped:
        print("No gameweeks with both predictions and realized results yet; nothing to score")
        return

    computed_at = datetime.now(timezone.utc).isoformat()
    per_gameweek_json: list[dict] = []
    db_rows: list[dict] = []
    all_rows: list[dict] = []

    for season, gameweek in sorted(grouped):
        rows = grouped[(season, gameweek)]
        all_rows.extend(rows)
        metrics = compute_accuracy(rows)
        per_gameweek_json.append({"season": season, "gameweek": gameweek, "positions": metrics})
        for m in metrics:
            db_rows.append({"season": season, "gameweek": gameweek, "computed_at": computed_at, **m})

    with get_connection() as conn:
        conn.executemany(UPSERT_ACCURACY, db_rows)

    overall = compute_accuracy(all_rows)
    ACCURACY_JSON.parent.mkdir(parents=True, exist_ok=True)
    ACCURACY_JSON.write_text(
        json.dumps(
            {
                "generated_at": computed_at,
                "n_scored_gameweeks": len(grouped),
                "overall": overall,
                "per_gameweek": per_gameweek_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    overall_all = next(m for m in overall if m["position"] == "ALL")
    print(
        f"Scored {len(grouped)} gameweek(s), {overall_all['n']} predictions | "
        f"MAE raw={overall_all['mae_raw']} adjusted={overall_all['mae_adjusted']} "
        f"-> {ACCURACY_JSON}"
    )


if __name__ == "__main__":
    main()
