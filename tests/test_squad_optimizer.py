"""The ILP is the other place a silent bug hides: an optimiser that quietly
violates the budget, the 2/5/5/3 shape, the 3-per-club cap or a legal XI would
still return a plausible-looking squad. These tests check every constraint on a
synthetic pool.
"""

import pytest

from constants import BUDGET, MAX_PER_CLUB, SQUAD, XI_BOUNDS
from optimize.squad_optimizer import optimize


def make_pool():
    """~48 players: enough depth per position, varied price / club / points."""
    clubs = [f"Club{c}" for c in range(10)]
    pool = []
    pid = 0
    per_pos = {"GK": 8, "DEF": 16, "MID": 16, "FWD": 8}
    for position, count in per_pos.items():
        for i in range(count):
            pid += 1
            pool.append(
                {
                    "player_id": pid,
                    "web_name": f"{position}{i}",
                    "position": position,
                    "team": clubs[pid % len(clubs)],
                    "price": 40 + (i % 8) * 5,          # 4.0m .. 7.5m
                    "predicted_points": 2.0 + (i % 9) * 0.75,
                }
            )
    return pool


@pytest.fixture(scope="module")
def result():
    return optimize(make_pool())


def test_squad_is_fifteen_with_right_positions(result):
    assert len(result["squad"]) == 15
    counts = {}
    for p in result["squad"]:
        counts[p["position"]] = counts.get(p["position"], 0) + 1
    assert counts == SQUAD


def test_within_budget(result):
    total_tenths = round(result["total_cost"] * 10)
    assert total_tenths <= BUDGET


def test_max_three_per_club(result):
    by_club = {}
    for p in result["squad"]:
        by_club[p["team"]] = by_club.get(p["team"], 0) + 1
    assert max(by_club.values()) <= MAX_PER_CLUB


def test_starting_eleven_is_legal(result):
    xi = result["xi"]
    assert len(xi) == 11
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in xi:
        counts[p["position"]] += 1
    assert counts["GK"] == 1
    for position, (low, high) in XI_BOUNDS.items():
        assert low <= counts[position] <= high
    assert sum(counts.values()) == 11


def test_bench_is_the_other_four(result):
    assert len(result["bench"]) == 4
    xi_ids = {p["player_id"] for p in result["xi"]}
    bench_ids = {p["player_id"] for p in result["bench"]}
    assert xi_ids.isdisjoint(bench_ids)
    assert xi_ids | bench_ids == {p["player_id"] for p in result["squad"]}


def test_captain_and_vice_are_top_two_starters(result):
    xi_sorted = sorted(result["xi"], key=lambda p: p["predicted_points"], reverse=True)
    captain = next(p for p in result["xi"] if p["is_captain"])
    vice = next(p for p in result["xi"] if p["is_vice"])
    assert captain["player_id"] != vice["player_id"]
    assert captain["predicted_points"] == xi_sorted[0]["predicted_points"]
    assert vice["predicted_points"] == xi_sorted[1]["predicted_points"]
    # captain must be a starter
    assert captain["in_xi"] and vice["in_xi"]


def test_infeasible_pool_raises():
    thin = [p for p in make_pool() if p["position"] != "GK"][:14]
    with pytest.raises((ValueError, RuntimeError)):
        optimize(thin)
