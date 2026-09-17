"""compute_accuracy: per-position (+ 'ALL') MAE/RMSE for raw vs. adjusted predictions."""

from models.track_accuracy import compute_accuracy


def _row(position, raw, adjusted, actual):
    return {"position": position, "raw_points": raw, "adjusted_points": adjusted, "actual": actual}


def test_perfect_predictions_score_zero_error():
    rows = [_row("MID", 5.0, 5.0, 5.0), _row("MID", 2.0, 2.0, 2.0)]
    metrics = compute_accuracy(rows)
    mid = next(m for m in metrics if m["position"] == "MID")
    assert mid["n"] == 2
    assert mid["mae_raw"] == 0.0
    assert mid["mae_adjusted"] == 0.0


def test_adjusted_falls_back_to_raw_when_none():
    # No news adjustment happened (adjusted_points is None) - adjusted metrics
    # should equal the raw ones, not blow up or get skipped.
    rows = [_row("FWD", 4.0, None, 6.0)]
    metrics = compute_accuracy(rows)
    fwd = next(m for m in metrics if m["position"] == "FWD")
    assert fwd["mae_raw"] == fwd["mae_adjusted"] == 2.0


def test_adjustment_can_change_the_error_independently():
    # Raw missed by 3; the news adjustment corrected it to a miss of 1.
    rows = [_row("DEF", 2.0, 4.0, 5.0)]
    metrics = compute_accuracy(rows)
    d = next(m for m in metrics if m["position"] == "DEF")
    assert d["mae_raw"] == 3.0
    assert d["mae_adjusted"] == 1.0


def test_all_aggregates_across_positions():
    rows = [_row("GK", 1.0, 1.0, 3.0), _row("FWD", 10.0, 10.0, 8.0)]
    metrics = compute_accuracy(rows)
    overall = next(m for m in metrics if m["position"] == "ALL")
    assert overall["n"] == 2
    assert overall["mae_raw"] == 2.0


def test_missing_position_is_omitted_not_zeroed():
    rows = [_row("MID", 1.0, 1.0, 1.0)]
    metrics = compute_accuracy(rows)
    positions = {m["position"] for m in metrics}
    assert "GK" not in positions
    assert "DEF" not in positions
    assert positions == {"MID", "ALL"}
