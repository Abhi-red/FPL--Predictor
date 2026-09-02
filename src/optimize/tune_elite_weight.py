"""Choose ELITE_WEIGHT by walk-forward backtest instead of guessing it.

For every candidate in ELITE_WEIGHT_CANDIDATES we replay history. At each past
gameweek that has BOTH elite ownership data and realized results we:

  1. fit the per-position models on gameweeks strictly earlier (the same
     no-leakage walk-forward rule as src/models/train.py),
  2. predict that gameweek,
  3. build a squad with  final_score = predicted_points * (1 + weight *
     elite_template_score),
  4. score that squad on the REAL total_points those players went on to get
     (XI + captain doubled), straight from player_gameweek_stats.

The weight with the best average realized points per gameweek wins. If nothing
beats the 0.0 control (pure stats) by more than ELITE_TUNE_NOISE_PTS per
gameweek, we fall back to the lowest non-zero weight that at least matches 0.0 -
and to 0.0 if none do. Below ELITE_TUNE_MIN_GAMEWEEKS usable gameweeks we defer
to 0.0 outright and say so, rather than pretend the weight is tuned.

Outputs:
  * data/elite_weight_config.json   - read by the optimiser at run time
  * DECISIONS.md                    - the full sweep table + the choice, rewritten
                                      between the ELITE_WEIGHT_TUNING markers

Run (needs elite data - src/ingest/fetch_elite.py - and features built):
    python src/optimize/tune_elite_weight.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    ELITE_TUNE_MIN_GAMEWEEKS,
    ELITE_TUNE_NOISE_PTS,
    ELITE_WEIGHT_CANDIDATES,
)
from db import get_connection  # noqa: E402
from features.build_features import feature_columns, get_feature_frame  # noqa: E402
from models.train import _SEASON_ORDER, _order_key, train_all_quiet  # noqa: E402
from optimize.squad_optimizer import optimize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO / "data" / "elite_weight_config.json"
DECISIONS_MD = REPO / "DECISIONS.md"
MARK_START = "<!-- ELITE_WEIGHT_TUNING:START -->"
MARK_END = "<!-- ELITE_WEIGHT_TUNING:END -->"

MIN_POOL = 20  # a backtest gameweek needs at least this many priced+predicted players


# --------------------------------------------------------------------------- #
# Pure decision helpers (unit-tested)
# --------------------------------------------------------------------------- #
def should_defer(n_gameweeks: int) -> bool:
    """True when there isn't enough elite+realized history to sweep meaningfully."""
    return n_gameweeks < ELITE_TUNE_MIN_GAMEWEEKS


def choose_weight(
    avg_by_weight: dict[float, float], noise: float = ELITE_TUNE_NOISE_PTS
) -> tuple[float, str]:
    """Pick a weight from {weight: avg realized pts/GW}, preferring simplicity.

    A non-zero weight is only chosen outright if it beats the 0.0 control by more
    than `noise`. Otherwise the lowest non-zero weight that still matches the
    control is used; failing that, 0.0.
    """
    control = avg_by_weight.get(0.0, 0.0)
    best_w = max(avg_by_weight, key=lambda k: avg_by_weight[k])
    best = avg_by_weight[best_w]
    margin = best - control

    if best_w != 0.0 and margin > noise:
        return best_w, (
            f"weight {best_w} beat the 0.0 control by {margin:+.3f} realized "
            f"pts/GW, above the {noise} noise band"
        )
    for weight in sorted(w for w in avg_by_weight if w > 0.0):
        if avg_by_weight[weight] >= control:
            return weight, (
                f"no weight beat 0.0 by more than the {noise} noise band "
                f"(best margin {margin:+.3f}); kept the lowest non-zero weight "
                f"({weight}) that still matched pure stats"
            )
    return 0.0, (
        f"no non-zero weight matched the 0.0 control (best margin {margin:+.3f}); "
        f"kept pure stats"
    )


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def usable_gameweeks() -> list[tuple[str, int]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT e.season, e.gameweek
            FROM elite_squads e
            WHERE EXISTS (
                SELECT 1 FROM player_gameweek_stats s
                WHERE s.season = e.season AND s.gameweek = e.gameweek
                  AND s.total_points IS NOT NULL
            )
            ORDER BY e.season, e.gameweek
            """
        ).fetchall()
    return [(r["season"], r["gameweek"]) for r in rows]


def elite_scores_for(season: str, gameweek: int) -> dict[int, float]:
    with get_connection() as conn:
        return {
            r["player_id"]: r["elite_template_score"]
            for r in conn.execute(
                "SELECT player_id, elite_template_score FROM elite_squads "
                "WHERE season = ? AND gameweek = ?",
                (season, gameweek),
            )
        }


def now_cost_for(season: str, gameweek: int) -> dict[int, int]:
    with get_connection() as conn:
        return {
            r["player_id"]: r["now_cost"]
            for r in conn.execute(
                "SELECT player_id, now_cost FROM player_gameweek_stats "
                "WHERE season = ? AND gameweek = ? AND now_cost IS NOT NULL",
                (season, gameweek),
            )
        }


def realized_map() -> dict[tuple[str, int, int], float]:
    with get_connection() as conn:
        return {
            (r["season"], r["gameweek"], r["player_id"]): float(r["total_points"])
            for r in conn.execute(
                "SELECT season, gameweek, player_id, total_points "
                "FROM player_gameweek_stats WHERE total_points IS NOT NULL"
            )
        }


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def candidate_pool(
    feat, order: int, season: str, gameweek: int, cols: list[str],
    elite: dict[int, float], costs: dict[int, int],
    realized: dict[tuple[str, int, int], float],
) -> list[dict]:
    train = feat[feat["_order"] < order]
    models = train_all_quiet(train, cols)
    if not models:
        return []
    test = feat[feat["_order"] == order]
    pool: list[dict] = []
    for position, model in models.items():
        sub = test[test["position"] == position]
        if sub.empty:
            continue
        preds = np.clip(model.predict(sub[cols]), 0, None)
        for player_id, web_name, team, pred in zip(
            sub["player_id"], sub["web_name"], sub["team"], preds
        ):
            pid = int(player_id)
            price = costs.get(pid)
            if price is None or (season, gameweek, pid) not in realized:
                continue
            pool.append(
                {
                    "player_id": pid,
                    "web_name": web_name,
                    "position": position,
                    "team": team,
                    "price": int(price),
                    "predicted_points": float(pred),
                    "elite_template_score": float(elite.get(pid, 0.0)),
                }
            )
    return pool


def realized_for_squad(result: dict, realized, season: str, gameweek: int) -> float:
    xi_points = sum(
        realized.get((season, gameweek, p["player_id"]), 0.0) for p in result["xi"]
    )
    captain_bonus = realized.get(
        (season, gameweek, result["captain"]["player_id"]), 0.0
    )
    return xi_points + captain_bonus


def run_sweep(usable: list[tuple[str, int]]) -> tuple[dict, list[dict], list[str]]:
    feat = get_feature_frame().assign(_order=lambda d: _order_key(d))
    cols = feature_columns()
    realized = realized_map()

    per_weight: dict[float, list[float]] = {w: [] for w in ELITE_WEIGHT_CANDIDATES}
    per_gameweek: list[dict] = []
    skipped: list[str] = []

    for season, gameweek in usable:
        order = _SEASON_ORDER.get(season, -1) * 100 + gameweek
        pool = candidate_pool(
            feat, order, season, gameweek, cols,
            elite_scores_for(season, gameweek),
            now_cost_for(season, gameweek),
            realized,
        )
        if len(pool) < MIN_POOL:
            skipped.append(f"{season} GW{gameweek}: pool too thin ({len(pool)})")
            continue

        gw_row: dict[float, float] = {}
        for weight in ELITE_WEIGHT_CANDIDATES:
            scored = [
                {**c, "score": c["predicted_points"] * (1 + weight * c["elite_template_score"])}
                for c in pool
            ]
            try:
                result = optimize(scored)
            except Exception as error:  # noqa: BLE001
                skipped.append(f"{season} GW{gameweek}: weight {weight}: {error}")
                gw_row = {}
                break
            gw_row[weight] = realized_for_squad(result, realized, season, gameweek)

        if len(gw_row) == len(ELITE_WEIGHT_CANDIDATES):
            for weight, value in gw_row.items():
                per_weight[weight].append(value)
            per_gameweek.append(
                {"season": season, "gameweek": gameweek,
                 **{str(w): round(v, 1) for w, v in gw_row.items()}}
            )

    return per_weight, per_gameweek, skipped


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _window_label(usable: list[tuple[str, int]]) -> str:
    if not usable:
        return "none"
    return f"{usable[0][0]} GW{usable[0][1]} - {usable[-1][0]} GW{usable[-1][1]}"


def write_config(payload: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_decisions_section(lines: list[str]) -> None:
    body = "\n".join([MARK_START, "", *lines, "", MARK_END])
    text = DECISIONS_MD.read_text(encoding="utf-8") if DECISIONS_MD.exists() else ""
    if MARK_START in text and MARK_END in text:
        head, _, rest = text.partition(MARK_START)
        _, _, tail = rest.partition(MARK_END)
        text = head + body + tail
    else:
        text = text.rstrip() + "\n\n## Elite-weight tuning (auto-generated)\n\n" + body + "\n"
    DECISIONS_MD.write_text(text, encoding="utf-8")


def deferred_payload(n: int, usable: list[tuple[str, int]]) -> dict:
    return {
        "elite_weight": 0.0,
        "status": "deferred",
        "reason": (
            f"only {n} gameweek(s) have both elite ownership data and realized "
            f"results; need {ELITE_TUNE_MIN_GAMEWEEKS}. Deferring to pure stats "
            f"until enough elite data accumulates."
        ),
        "n_backtest_gameweeks": n,
        "min_required": ELITE_TUNE_MIN_GAMEWEEKS,
        "candidates": list(ELITE_WEIGHT_CANDIDATES),
        "window": _window_label(usable),
        "sweep": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_decisions(payload: dict, per_gameweek: list[dict], skipped: list[str]) -> list[str]:
    lines = [
        f"_Last run {payload['generated_at']} — "
        f"`python src/optimize/tune_elite_weight.py`._",
        "",
        f"**Chosen `ELITE_WEIGHT` = {payload['elite_weight']}** "
        f"(status: `{payload['status']}`).",
        "",
    ]
    if payload["status"] == "deferred":
        lines += [payload["reason"], ""]
        if per_gameweek:
            lines += ["Gameweeks with elite data so far: " + payload["window"] + "."]
        return lines

    lines += [
        payload.get("chosen_reason", ""),
        "",
        f"Swept {payload['n_backtest_gameweeks']} walk-forward gameweeks "
        f"({payload['window']}). Realized points = XI actual total_points + "
        f"captain doubled, from `player_gameweek_stats`.",
        "",
        "| weight | avg realized pts / GW | total realized | Δ vs 0.0 |",
        "|---|---|---|---|",
    ]
    control = next((s["avg_realized"] for s in payload["sweep"] if s["weight"] == 0.0), 0.0)
    for entry in payload["sweep"]:
        delta = entry["avg_realized"] - control
        lines.append(
            f"| {entry['weight']} | {entry['avg_realized']:.3f} | "
            f"{entry['total_realized']:.1f} | {delta:+.3f} |"
        )
    if skipped:
        lines += ["", f"Skipped {len(skipped)} gameweek/weight combos: "
                  + "; ".join(skipped[:8]) + ("; ..." if len(skipped) > 8 else "")]
    return lines


def main() -> None:
    usable = usable_gameweeks()
    print(f"Usable backtest gameweeks (elite data + realized results): {len(usable)}")

    if should_defer(len(usable)):
        payload = deferred_payload(len(usable), usable)
        write_config(payload)
        write_decisions_section(render_decisions(payload, [], []))
        print(
            f"DEFERRED: {payload['reason']}\n"
            f"  wrote ELITE_WEIGHT = 0.0 to {CONFIG_PATH.name}"
        )
        return

    per_weight, per_gameweek, skipped = run_sweep(usable)
    scored = len(per_gameweek)
    if should_defer(scored):
        payload = deferred_payload(scored, usable)
        payload["reason"] = (
            f"elite data covers {len(usable)} gameweeks but only {scored} could be "
            f"backtested (thin candidate pools / infeasible squads); need "
            f"{ELITE_TUNE_MIN_GAMEWEEKS}. Deferring to pure stats."
        )
        write_config(payload)
        write_decisions_section(render_decisions(payload, per_gameweek, skipped))
        print(f"DEFERRED: {payload['reason']}")
        return

    avg = {w: mean(per_weight[w]) for w in ELITE_WEIGHT_CANDIDATES}
    total = {w: sum(per_weight[w]) for w in ELITE_WEIGHT_CANDIDATES}
    weight, reason = choose_weight(avg)

    payload = {
        "elite_weight": weight,
        "status": "tuned",
        "chosen_reason": reason,
        "n_backtest_gameweeks": scored,
        "min_required": ELITE_TUNE_MIN_GAMEWEEKS,
        "candidates": list(ELITE_WEIGHT_CANDIDATES),
        "window": _window_label(usable),
        "sweep": [
            {
                "weight": w,
                "avg_realized": round(avg[w], 3),
                "total_realized": round(total[w], 2),
                "n": scored,
            }
            for w in ELITE_WEIGHT_CANDIDATES
        ],
        "per_gameweek": per_gameweek,
        "skipped": skipped,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_config(payload)
    write_decisions_section(render_decisions(payload, per_gameweek, skipped))

    print(f"\nweight   avg realized pts/GW   total")
    for w in ELITE_WEIGHT_CANDIDATES:
        print(f"  {w:<5}  {avg[w]:>18.3f}   {total[w]:>7.1f}")
    print(f"\nChosen ELITE_WEIGHT = {weight}\n  {reason}\n  -> {CONFIG_PATH}")


if __name__ == "__main__":
    main()
