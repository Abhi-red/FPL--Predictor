"""Run the weekly pipeline end to end: ingest -> features -> predict -> RAG ->
optimise -> explain -> export site JSON.

This is the incremental path (current-season gameweek fetch only); the one-time
historical backfill is separate (src/ingest/backfill_historical.py).

Stage failures in the news/RAG stages are non-fatal (predictions just stay
un-adjusted); any other stage failing aborts with exit code 1 so CI notices.

Run:
    python src/pipeline.py
"""

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import MODEL_MAX_AGE_DAYS, SEASON  # noqa: E402
from db import get_connection, init_db  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
SITE_DATA = Path(__file__).resolve().parent.parent / "site" / "data"
NON_FATAL = {"embed", "adjust"}


def run_stage(name: str, fn) -> bool:
    print(f"\n{'=' * 4} START {name}", flush=True)
    started = time.time()
    try:
        fn()
    except SystemExit as exit_error:
        if exit_error.code:
            _report_fail(name, started)
            return False
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        _report_fail(name, started)
        return False
    print(f"{'=' * 4} END {name} (ok, {time.time() - started:.1f}s)", flush=True)
    return True


def _report_fail(name: str, started: float) -> None:
    print(
        f"{'=' * 4} FAIL {name} ({time.time() - started:.1f}s)",
        file=sys.stderr,
        flush=True,
    )


def stage_fetch_gameweek_stats() -> None:
    from ingest.fetch_gameweek_stats import main as fetch_gw

    fetch_gw()


def stage_build_features() -> None:
    from features.build_features import build

    build()


def stage_ensure_models() -> None:
    from models.train import production_fit

    manifest = MODELS_DIR / "manifest.json"
    models_present = all((MODELS_DIR / f"{p}.joblib").exists() for p in ("GK", "DEF", "MID", "FWD"))
    fresh = False
    if manifest.exists():
        trained_at = datetime.fromisoformat(json.loads(manifest.read_text())["trained_at"])
        fresh = (datetime.now(timezone.utc) - trained_at).days < MODEL_MAX_AGE_DAYS
    if models_present and fresh:
        print("models present and fresh; skipping retrain")
        return
    production_fit()


def stage_predict() -> None:
    from models.predict import main as predict_main

    predict_main()


def stage_embed() -> None:
    from rag.embed import main as embed_main

    embed_main()


def stage_adjust() -> None:
    from rag.adjust import main as adjust_main

    adjust_main()


def stage_optimize() -> None:
    from optimize.squad_optimizer import main as optimize_main

    optimize_main()


def stage_explain() -> None:
    from explain.generate_explanation import main as explain_main

    explain_main()


def stage_export_site_json() -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        gw_row = conn.execute(
            "SELECT MAX(gameweek) AS gw FROM predictions WHERE season = ?", (SEASON,)
        ).fetchone()
        gameweek = gw_row["gw"]
        players = [
            dict(r)
            for r in conn.execute(
                """
                SELECT p.player_id, p.web_name, p.position, p.team,
                       ROUND(p.now_cost / 10.0, 1) AS price,
                       pr.raw_points, pr.adjusted_points,
                       pr.adjustment_factor, pr.adjustment_reason, pr.news_url,
                       f.roll5_total_points, f.roll5_minutes_played,
                       f.start_rate_5, f.form_ewm, f.fdr, f.was_home
                FROM players p
                LEFT JOIN predictions pr
                       ON pr.player_id = p.player_id AND pr.season = :season
                      AND pr.gameweek = :gw
                LEFT JOIN player_features f
                       ON f.player_id = p.player_id AND f.season = :season
                      AND f.gameweek = :gw
                WHERE p.element_id IS NOT NULL
                ORDER BY pr.adjusted_points DESC NULLS LAST, pr.raw_points DESC NULLS LAST
                """,
                {"season": SEASON, "gw": gameweek},
            )
        ]

    meta = {
        "season": SEASON,
        "gameweek": gameweek,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player_count": len(players),
    }
    (SITE_DATA / "players.json").write_text(
        json.dumps({"meta": meta, "players": players}, indent=2), encoding="utf-8"
    )
    (SITE_DATA / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {len(players)} players for GW{gameweek}")


STAGES = [
    ("fetch_gameweek_stats", stage_fetch_gameweek_stats),
    ("build_features", stage_build_features),
    ("ensure_models", stage_ensure_models),
    ("predict", stage_predict),
    ("embed", stage_embed),
    ("adjust", stage_adjust),
    ("optimize", stage_optimize),
    ("explain", stage_explain),
    ("export_site_json", stage_export_site_json),
]


def main() -> None:
    init_db()
    print(f"Pipeline start: {SEASON}  {datetime.now(timezone.utc).isoformat()}")
    failed: list[str] = []
    for name, fn in STAGES:
        if run_stage(name, fn):
            continue
        failed.append(name)
        if name not in NON_FATAL:
            print(f"\nAborting: fatal stage {name} failed", file=sys.stderr)
            sys.exit(1)
        print(f"(continuing; {name} is non-fatal)", file=sys.stderr)

    print(f"\nPipeline done. Failed non-fatal stages: {failed or 'none'}")


if __name__ == "__main__":
    main()
