# Data corrections and refresh completed

The corrected pipeline has regenerated the **2026-27 GW6**
snapshot for 667 players. Its export timestamp is
`2026-10-09T07:17:38.223287+00:00` (UTC).

## Corrections now included

- Historical features use season-specific club, position, and player identity.
  All 136,402 historical/current training rows have a matching season identity.
- Historical and upcoming fixture counts are propagated into the model.
- Player identities and prices refresh on each pipeline run. Published gameweeks
  cannot be replaced by older predictions; incompatible feature schemas force
  model retraining.
- New raw predictions clear old news adjustments. News matching requires an
  identifying name and a claim in the same clause, with negation checks and
  boundaries that prevent substring matches or another player's injury claim.

## Refreshed models and validation

All four position models were retrained on 136,402 rows through GW5 with
**35 inputs**, including `fixture_count`. The database, trained models, and
feature matrix remain cache artifacts under `data/`; the hosted weekly pipeline
will retrain any restored 34-input models before generating new predictions.

The walk-forward backtest starts in 2024-25, scores 81 gameweeks and 59,473
player-gameweek predictions, and refits every five scored gameweeks. Training
uses only gameweeks earlier than each evaluation gameweek.

| Position | Predictions | MAE (points) | RMSE (points) |
|---|---:|---:|---:|
| GK | 6,562 | 0.6987 | 1.5811 |
| DEF | 19,680 | 1.0882 | 2.0432 |
| MID | 26,631 | 1.0001 | 1.9661 |
| FWD | 6,600 | 1.1576 | 2.2141 |
| Overall | 59,473 | 1.0135 | 1.9829 |

[Full model manifest, quality checks, and backtest results](validation/model-backtest-2026-10-09.json)
are committed alongside this note. This evaluates the raw points model, not a
historical replay of news adjustments or live captain performance. The prior
saved benchmark used a different sample and retraining cadence, so these
numbers are a new baseline rather than proof of an accuracy improvement.
Historical price and team-strength snapshots have not been audited as exact
pre-deadline observations.

Verification: **44 Python tests and 9 browser tests pass**. Data audits verify
unique player rows, consistent season/gameweek across exports, all four
35-input models, exact fixture-count flags, squad budget/club/position rules,
and agreement between squad predictions and player exports.

## News and explanation runtime

The final news pass fetched 150 articles, indexed 214 surviving chunks, and
adjusted two predictions. This local run used the supported hashing-vectorizer
and NumPy retrieval fallback; the hosted workflow retains its declared
sentence-transformers and FAISS dependencies.

Claude rejected the configured local API key with HTTP 401. The current
explanation uses the existing deterministic template and reflects the refreshed
squad. This does not change point predictions. Set a valid `ANTHROPIC_API_KEY`
in the execution environment to restore generated prose on a later run.

The elite-weight sweep over GW1-GW5 retained weight **0.0**, because no tested
nonzero weight matched the model-only selection on realized points.

## Reproduce

```powershell
pip install -r requirements.txt
python src/pipeline.py
python src/models/train.py --backtest --start-season 2024-25 --stride 5
pytest
npm ci
npm run test:dashboard
```

Activate a Conda environment before invoking its Python executable so its
native-library paths are available. Keep the historical backtest output separate
from any future live-results accuracy report.
