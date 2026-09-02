"""The elite-weight tuner's decision rules: defer when data is thin, and prefer
the simpler weight unless a non-zero one clearly beats the 0.0 control.
"""

from constants import ELITE_TUNE_MIN_GAMEWEEKS
from optimize.tune_elite_weight import choose_weight, should_defer


def test_defers_below_the_minimum_gameweeks():
    assert should_defer(0)
    assert should_defer(ELITE_TUNE_MIN_GAMEWEEKS - 1)
    assert not should_defer(ELITE_TUNE_MIN_GAMEWEEKS)
    assert not should_defer(ELITE_TUNE_MIN_GAMEWEEKS + 10)


def test_clear_winner_is_chosen():
    avg = {0.0: 50.0, 0.05: 50.1, 0.1: 52.0, 0.2: 51.0}
    weight, reason = choose_weight(avg, noise=0.25)
    assert weight == 0.1
    assert "beat the 0.0 control" in reason


def test_sub_noise_margin_falls_back_to_lowest_matching_weight():
    # best is 0.1 but only +0.15 over control (< 0.25 noise); 0.05 matches control
    avg = {0.0: 50.0, 0.05: 50.0, 0.1: 50.15, 0.2: 49.5}
    weight, reason = choose_weight(avg, noise=0.25)
    assert weight == 0.05
    assert "lowest non-zero weight" in reason


def test_all_nonzero_worse_than_control_keeps_zero():
    avg = {0.0: 50.0, 0.05: 49.0, 0.1: 48.0, 0.3: 40.0}
    weight, reason = choose_weight(avg, noise=0.25)
    assert weight == 0.0
    assert "kept pure stats" in reason


def test_exact_tie_with_control_prefers_smallest_nonzero():
    avg = {0.0: 50.0, 0.05: 50.0, 0.15: 50.0}
    weight, _ = choose_weight(avg, noise=0.25)
    assert weight == 0.05
