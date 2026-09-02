"""Pick the optimal 15-man squad + starting XI + captaincy with an ILP.

Maximises a per-player *score* over the XI (captain counted twice), subject to
the real FPL rules: GBP 100.0m budget, 2/5/5/3 squad by position, at most 3
players from any one club, and a legal XI shape (1 GK, 3-5 DEF, 2-5 MID, 1-3
FWD, 11 total) chosen from the 15.

    score = predicted_points * (1 + ELITE_WEIGHT * elite_template_score)

ELITE_WEIGHT is read at run time from data/elite_weight_config.json, which
src/optimize/tune_elite_weight.py writes after a walk-forward sweep. If that
file is missing (or the tuner deferred), ELITE_WEIGHT is 0.0 and the score is
just predicted_points. Captain/vice are always the top-two XI players by
predicted_points - the elite signal shapes *selection*, not the captaincy call.

``optimize(players)`` is the pure entry point; ``main()`` wraps it in DB I/O.

Run (after predict.py, ideally after adjust.py + fetch_elite.py):
    python src/optimize/squad_optimizer.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import BUDGET, MAX_PER_CLUB, SEASON, SQUAD, XI_BOUNDS  # noqa: E402
from db import get_connection  # noqa: E402

SITE_DATA = Path(__file__).resolve().parent.parent.parent / "site" / "data"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ELITE_WEIGHT_CONFIG = DATA_DIR / "elite_weight_config.json"
BENCH_POS_ORDER = {"GK": 3, "DEF": 0, "MID": 1, "FWD": 2}


def load_elite_weight() -> tuple[float, str]:
    """(weight, status) from the tuner's config; (0.0, 'no-config') if absent."""
    if not ELITE_WEIGHT_CONFIG.exists():
        return 0.0, "no-config"
    try:
        cfg = json.loads(ELITE_WEIGHT_CONFIG.read_text(encoding="utf-8"))
        return float(cfg.get("elite_weight", 0.0)), str(cfg.get("status", "unknown"))
    except Exception:  # noqa: BLE001
        return 0.0, "unreadable-config"


def optimize(players: list[dict]) -> dict:
    """Solve the squad ILP. Each `players` item needs: player_id, web_name,
    position, team, price (tenths of a million), predicted_points. Optional
    `score` (defaults to predicted_points) is what the objective maximises;
    optional `elite_template_score` is carried through for display."""
    if len(players) < 15:
        raise ValueError(f"need at least 15 players, got {len(players)}")

    by_id = {p["player_id"]: p for p in players}
    ids = list(by_id)
    points = {i: float(by_id[i].get("predicted_points") or 0.0) for i in ids}
    score = {i: float(by_id[i].get("score", points[i]) or 0.0) for i in ids}

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    xi = pulp.LpVariable.dicts("xi", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")

    # XI selection follows the elite-blended score; the captain double-up is
    # valued at the pure model expectation.
    prob += pulp.lpSum(score[i] * xi[i] for i in ids) + pulp.lpSum(
        points[i] * captain[i] for i in ids
    )

    prob += pulp.lpSum(squad[i] for i in ids) == 15
    for position, count in SQUAD.items():
        prob += (
            pulp.lpSum(squad[i] for i in ids if by_id[i]["position"] == position)
            == count
        )
    prob += pulp.lpSum(by_id[i]["price"] * squad[i] for i in ids) <= BUDGET
    for club in {by_id[i]["team"] for i in ids}:
        prob += (
            pulp.lpSum(squad[i] for i in ids if by_id[i]["team"] == club)
            <= MAX_PER_CLUB
        )

    prob += pulp.lpSum(xi[i] for i in ids) == 11
    for position, (low, high) in XI_BOUNDS.items():
        in_pos = pulp.lpSum(xi[i] for i in ids if by_id[i]["position"] == position)
        prob += in_pos >= low
        prob += in_pos <= high
    for i in ids:
        prob += xi[i] <= squad[i]
        prob += captain[i] <= xi[i]
    prob += pulp.lpSum(captain[i] for i in ids) == 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver returned {pulp.LpStatus[status]!r}, not Optimal")

    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = {i for i in chosen if xi[i].value() > 0.5}
    # Captain/vice: top-two XI by pure predicted_points (not the elite-blended score).
    ranked_xi = sorted(starters, key=lambda i: points[i], reverse=True)
    captain_id, vice_id = ranked_xi[0], ranked_xi[1]

    def view(i: int) -> dict:
        p = by_id[i]
        return {
            "player_id": i,
            "web_name": p["web_name"],
            "position": p["position"],
            "team": p["team"],
            "price": round(p["price"] / 10.0, 1),
            "predicted_points": round(points[i], 2),
            "elite_template_score": round(float(p.get("elite_template_score") or 0.0), 4),
            "score": round(score[i], 2),
            "in_xi": i in starters,
            "is_captain": i == captain_id,
            "is_vice": i == vice_id,
        }

    xi_players = sorted(
        (view(i) for i in starters),
        key=lambda v: (BENCH_POS_ORDER[v["position"]], -v["predicted_points"]),
    )
    bench_players = sorted(
        (view(i) for i in chosen if i not in starters),
        key=lambda v: (BENCH_POS_ORDER[v["position"]], -v["predicted_points"]),
    )
    formation = "-".join(
        str(sum(1 for v in xi_players if v["position"] == pos))
        for pos in ("DEF", "MID", "FWD")
    )
    return {
        "formation": formation,
        "total_cost": round(sum(by_id[i]["price"] for i in chosen) / 10.0, 1),
        "predicted_points": round(
            sum(points[i] for i in starters) + points[captain_id], 2
        ),
        "captain": view(captain_id),
        "vice": view(vice_id),
        "xi": xi_players,
        "bench": bench_players,
        "squad": xi_players + bench_players,
    }


UPSERT_SQUAD_ROW = """
INSERT INTO squads (
    season, gameweek, generated_at, player_id,
    in_squad, in_xi, is_captain, is_vice, predicted_points
)
VALUES (:season, :gameweek, :generated_at, :player_id, 1, :in_xi, :is_captain, :is_vice, :predicted_points)
ON CONFLICT(season, gameweek, player_id) DO UPDATE SET
    generated_at     = excluded.generated_at,
    in_squad         = 1,
    in_xi            = excluded.in_xi,
    is_captain       = excluded.is_captain,
    is_vice          = excluded.is_vice,
    predicted_points = excluded.predicted_points
"""


def load_candidates() -> tuple[int, list[dict]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(gameweek) AS gw FROM predictions WHERE season = ?", (SEASON,)
        ).fetchone()
        gameweek = row["gw"]
        if gameweek is None:
            raise RuntimeError("no predictions found - run src/models/predict.py first")
        # Elite picks for the upcoming gameweek aren't published until managers
        # set their teams, so fall back to the most recent gameweek that does
        # have elite data (the current template changes slowly week to week).
        rows = conn.execute(
            """
            SELECT pr.player_id, p.web_name, p.position, p.team, p.now_cost AS price,
                   COALESCE(pr.adjusted_points, pr.raw_points) AS predicted_points,
                   COALESCE(e.elite_template_score, 0.0)       AS elite_template_score
            FROM predictions pr
            JOIN players p ON p.player_id = pr.player_id
            LEFT JOIN elite_squads e
                   ON e.player_id = pr.player_id AND e.season = pr.season
                  AND e.gameweek = (
                      SELECT MAX(e2.gameweek) FROM elite_squads e2
                      WHERE e2.season = pr.season AND e2.gameweek <= pr.gameweek
                  )
            WHERE pr.season = ? AND pr.gameweek = ?
              AND p.now_cost IS NOT NULL AND pr.raw_points IS NOT NULL
            """,
            (SEASON, gameweek),
        ).fetchall()
    return gameweek, [dict(r) for r in rows]


def main() -> None:
    gameweek, candidates = load_candidates()
    weight, status = load_elite_weight()
    covered = sum(1 for c in candidates if c["elite_template_score"] > 0)
    print(
        f"Optimising {SEASON} GW{gameweek} from {len(candidates)} candidates | "
        f"ELITE_WEIGHT={weight} ({status}); elite data for {covered} players"
    )

    for c in candidates:
        c["score"] = c["predicted_points"] * (1 + weight * c["elite_template_score"])

    result = optimize(candidates)
    generated_at = datetime.now(timezone.utc).isoformat()
    result.update(
        season=SEASON,
        gameweek=gameweek,
        generated_at=generated_at,
        elite_weight=weight,
        elite_weight_status=status,
        elite_players_covered=covered,
    )

    squad_rows = [
        {
            "season": SEASON,
            "gameweek": gameweek,
            "generated_at": generated_at,
            "player_id": p["player_id"],
            "in_xi": int(p["in_xi"]),
            "is_captain": int(p["is_captain"]),
            "is_vice": int(p["is_vice"]),
            "predicted_points": p["predicted_points"],
        }
        for p in result["squad"]
    ]
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM squads WHERE season = ? AND gameweek = ?", (SEASON, gameweek)
        )
        conn.executemany(UPSERT_SQUAD_ROW, squad_rows)

    for target in (DATA_DIR / "squad.json", SITE_DATA / "squad.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"  {result['formation']} | GBP{result['total_cost']}m | "
        f"proj {result['predicted_points']} pts | "
        f"(C) {result['captain']['web_name']} (V) {result['vice']['web_name']}"
    )


if __name__ == "__main__":
    main()
