# FPL Weekly Predictor

[Live dashboard](https://abhi-red.github.io/FPL--Predictor/) · [Latest published data](site/data/meta.json) · [Engineering decisions](DECISIONS.md)

A weekly Fantasy Premier League prediction pipeline and public dashboard. It combines position-specific XGBoost forecasts, recent football news, and a PuLP integer program to recommend a 15-player squad, starting XI, captain, and vice-captain. The dashboard shows player forecasts and an explanation of the suggested squad.

## What is implemented

- Ingest current FPL players, fixtures, prices, and gameweek results; backfill historical seasons for model training.
- Build rolling form and fixture features, fit four position-specific XGBoost models, and evaluate them with walk-forward backtesting (`python src/models/train.py --backtest`).
- Read BBC Sport, Sky Sports, and Guardian football RSS feeds; embed recent articles in a local FAISS index; apply bounded, keyword-gated news adjustments to forecasts.
- Sample top-ranked managers' published squads to calculate an elite-ownership score. A walk-forward tuner can select its weight when enough comparable gameweeks exist. If tuning is deferred or its config is missing, the optimizer uses **weight 0** and elite ownership does not affect the squad. The [published squad JSON](site/data/squad.json) records the weight and status for each run.
- Optimize a legal FPL squad and XI under budget, position, formation, and club limits. Generate a written explanation with Claude when `ANTHROPIC_API_KEY` is available, or a deterministic template otherwise.
- Publish static JSON for the dashboard through a scheduled GitHub Actions workflow. The site runs without a backend.

The model's predictions are estimates, not guaranteed points. News adjustments use retrieved text and simple signals; they are not a verified medical or lineup feed. The current dashboard data and generation time are visible in the [metadata file](site/data/meta.json).

## Architecture and persistence

```text
FPL API + historical results ──> SQLite ──> features ──> XGBoost forecasts
Football RSS feeds ──> embeddings / FAISS ──> bounded news adjustments
Elite-manager sample ──> optional tuned ownership weight
Forecasts + constraints ──> PuLP squad optimizer ──> explanation
                                      │
                                      └──> site/data/*.json ──> GitHub Pages
```

`data/fpl.db`, trained models, features, and the FAISS index are **gitignored**. The weekly workflow restores them with `actions/cache`; on a cache miss it backfills historical data. Only `site/data/*.json` and the rendered explanation are committed for the static site. The workflow runs Monday at 06:30 UTC and can be dispatched manually; its results depend on upstream data and a successful run. See [the workflow](.github/workflows/weekly-pipeline.yml) and [Pages deployment](.github/workflows/pages.yml).

## Run locally

Python 3.12 is used in CI. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/db.py
python src/ingest/fetch_fpl.py
python src/ingest/backfill_historical.py  # first run only; downloads historical data
python src/pipeline.py
python -m http.server 8000 -d site
```

Open `http://localhost:8000`. The first run downloads historical data and an embedding model and may take substantially longer than later runs. The pipeline needs network access to its data sources. Set `ANTHROPIC_API_KEY` in your environment to enable Claude explanations; without it, the template explanation is used. No key is required to browse the published site.

For individual stages, see [`src/pipeline.py`](src/pipeline.py). Run the test suite with:

```bash
python -m pytest -q
```

To deploy a fork, enable GitHub Pages with **GitHub Actions** as the source. Add an `ANTHROPIC_API_KEY` Actions secret only if you want Claude explanations; the workflow can use the template fallback without it.

## Next steps

- Per-user squads and transfer or chip strategy would need additional account state and product work.
- Broaden historical evaluation of the news and elite-ownership adjustments as more comparable weeks accumulate.

## License

See [LICENSE](LICENSE).
