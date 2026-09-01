"""Train one XGBoost points regressor per position, and backtest them.

Two modes:

  python src/models/train.py            # production fit: train on ALL feature
                                        # rows, save data/models/{POS}.joblib

  python src/models/train.py --backtest # walk-forward validation: for a grid of
                                        # historical gameweeks, train only on
                                        # rows strictly earlier and score the
                                        # held-out gameweek; report MAE / RMSE
                                        # per position -> data/models/backtest_metrics.json

Separate models per position because a defender's and a forward's points come
from different things; the feature columns are shared (data/models/feature_columns.json).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import HISTORICAL_SEASONS, SEASON  # noqa: E402
from features.build_features import feature_columns, get_feature_frame  # noqa: E402

POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

# Hand-picked, not grid-searched: shallow trees + shrinkage + row/column
# subsampling is a safe default for noisy tabular sports data.
XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": 4,
    "random_state": 42,
}

_SEASON_ORDER = {s: i for i, s in enumerate([*HISTORICAL_SEASONS, SEASON])}


def _order_key(df: pd.DataFrame) -> pd.Series:
    """A single sortable integer per (season, gameweek) for 'strictly before' masks."""
    return df["season"].map(_SEASON_ORDER).fillna(-1).astype(int) * 100 + df["gameweek"]


def _fit_one(frame: pd.DataFrame, cols: list[str]) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(frame[cols], frame["target_points"])
    return model


def train_all(df: pd.DataFrame, cols: list[str]) -> dict[str, XGBRegressor]:
    models: dict[str, XGBRegressor] = {}
    for position in POSITIONS:
        subset = df[(df["position"] == position) & df["target_points"].notna()]
        if subset.empty:
            print(f"  {position}: no rows, skipping", file=sys.stderr)
            continue
        models[position] = _fit_one(subset, cols)
        print(f"  {position}: fit on {len(subset)} rows")
    return models


# --------------------------------------------------------------------------- #
# Production fit
# --------------------------------------------------------------------------- #
def production_fit() -> None:
    import joblib

    df = get_feature_frame()
    cols = feature_columns()
    print(f"Production fit on {len(df)} feature rows")
    models = train_all(df, cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for position, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{position}.joblib")

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": cols,
        "rows_per_position": {
            p: int(((df["position"] == p) & df["target_points"].notna()).sum())
            for p in POSITIONS
        },
        "latest_season": str(df["season"].max()),
        "latest_gameweek": int(df.loc[df["season"] == df["season"].max(), "gameweek"].max()),
    }
    (MODELS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(models)} models + manifest to {MODELS_DIR}")


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def _metrics(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan")}
    return {
        "n": int(errors.size),
        "mae": round(float(np.abs(errors).mean()), 4),
        "rmse": round(float(np.sqrt((errors**2).mean())), 4),
    }


def backtest(start_season: str, stride: int) -> dict:
    df = get_feature_frame()
    cols = feature_columns()
    df = df.assign(_order=_order_key(df))

    start = _SEASON_ORDER.get(start_season, 0) * 100
    test_points = sorted(
        {
            (int(o), s, int(g))
            for o, s, g in zip(df["_order"], df["season"], df["gameweek"])
            if o >= start
        }
    )
    print(
        f"Walk-forward backtest: {len(test_points)} candidate gameweeks from "
        f"{start_season}, refitting every {stride}"
    )

    collected: dict[str, list[np.ndarray]] = {p: [] for p in POSITIONS}
    per_gw: list[dict] = []
    models: dict[str, XGBRegressor] = {}

    for index, (order, season, gameweek) in enumerate(test_points):
        train_df = df[df["_order"] < order]
        test_df = df[(df["_order"] == order) & df["target_points"].notna()]
        if train_df.empty or test_df.empty:
            continue
        if index % stride == 0 or not models:
            models = train_all_quiet(train_df, cols)

        gw_row: dict = {"season": season, "gameweek": gameweek}
        for position in POSITIONS:
            model = models.get(position)
            pos_test = test_df[test_df["position"] == position]
            if model is None or pos_test.empty:
                continue
            pred = np.clip(model.predict(pos_test[cols]), 0, None)
            err = pred - pos_test["target_points"].to_numpy()
            collected[position].append(err)
            gw_row[f"{position}_mae"] = round(float(np.abs(err).mean()), 3)
        per_gw.append(gw_row)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_season": start_season,
        "stride": stride,
        "per_position": {
            p: _metrics(np.concatenate(collected[p]) if collected[p] else np.array([]))
            for p in POSITIONS
        },
    }
    all_err = np.concatenate([e for lst in collected.values() for e in lst]) if any(
        collected.values()
    ) else np.array([])
    summary["overall"] = _metrics(all_err)
    summary["per_gameweek"] = per_gw

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "backtest_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nposition   n        MAE     RMSE")
    for position in POSITIONS:
        m = summary["per_position"][position]
        print(f"  {position:<6} {m['n']:>7}  {m['mae']:>7}  {m['rmse']:>7}")
    o = summary["overall"]
    print(f"  {'ALL':<6} {o['n']:>7}  {o['mae']:>7}  {o['rmse']:>7}")
    return summary


def train_all_quiet(df: pd.DataFrame, cols: list[str]) -> dict[str, XGBRegressor]:
    models: dict[str, XGBRegressor] = {}
    for position in POSITIONS:
        subset = df[(df["position"] == position) & df["target_points"].notna()]
        if len(subset) >= 50:
            models[position] = _fit_one(subset, cols)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest", action="store_true", help="run walk-forward validation")
    parser.add_argument(
        "--start-season",
        default=HISTORICAL_SEASONS[1] if len(HISTORICAL_SEASONS) > 1 else HISTORICAL_SEASONS[0],
        help="first season to score in the backtest (needs a prior season of history)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="refit every N scored gameweeks (1 = refit every gameweek, slowest/most faithful)",
    )
    args = parser.parse_args()

    if args.backtest:
        backtest(args.start_season, args.stride)
    else:
        production_fit()


if __name__ == "__main__":
    main()
