# FPL Weekly Predictor

An end-to-end system that predicts the best Fantasy Premier League (FPL) team every gameweek, combining statistical modeling with retrieval-augmented reasoning over injury news and team updates — and explains every recommendation in plain English.

## What this is

A public, read-only website where anyone can:
- Browse every Premier League player with live stats and predicted points for the upcoming gameweek
- See a suggested 15-man squad, starting XI, and captain pick
- Click into any player to see *why* — the stats behind the prediction, any injury/rotation flags pulled from news sources, and a natural-language summary tying it together

Data refreshes automatically once a week, after each gameweek's results are final.

## Why this project

Most FPL tools are either pure stats (xG, form, fixture difficulty) or pure written opinion (pundit articles, "who to captain" blog posts) — rarely both, and rarely with visible sourcing for *why* a recommendation was made. This project combines:

1. **Structured prediction** — a gradient-boosted model trained on historical performance, fixtures, and underlying stats
2. **Unstructured reasoning (RAG)** — retrieval over injury reports, team news, and analyst write-ups that a stats-only model can't see
3. **Optimization** — an integer linear program that picks the mathematically best squad under FPL's budget and formation rules
4. **Explanation** — an LLM call that turns the model's output into a readable rationale, citing both the numbers and the news

## Architecture

```
GitHub Actions (weekly cron)
        │
        ▼
Pipeline script
├── Fetch FPL API (prices, points, fixtures, ownership)
├── Fetch Understat/FBref (xG, xA)
├── Scrape team news / injury reports
├── Embed new articles (local model, e.g. sentence-transformers)
        │
        ├──► SQLite (structured player/gameweek data, committed to repo)
        └──► FAISS index (embedded news chunks, committed to repo)
                        │
                        ▼
        Prediction model (XGBoost, per position)
                        │
                        ▼
        RAG adjustment (retrieve relevant news per player,
                         flag/adjust raw prediction)
                        │
                        ▼
        ILP optimizer (PuLP — picks optimal 15/XI under
                        £100m budget, formation, 3-per-club rules)
                        │
                        ▼
        Claude API call (1x/week — generates natural-language
                          explanation for the suggested team)
                        │
                        ▼
        Static JSON output → committed/pushed to repo
                        │
                        ▼
        Static site (Vercel / GitHub Pages) — public, free, read-only
```

The key design decision: **everything expensive runs once a week inside a free GitHub Action.** The public site itself is just static files — no always-on server, no per-visitor compute cost.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Weekly compute | GitHub Actions (scheduled workflow) | Free, no idle server cost |
| Structured data | SQLite (file in repo) | Small dataset (~700 players × 38 weeks), no hosted DB needed |
| Vector store | FAISS (local index, file in repo) | Free, no managed vector DB needed at this scale |
| Embeddings | `sentence-transformers` (local model) | Free, runs inside the Action |
| Prediction model | XGBoost (per position: GK/DEF/MID/FWD) | Handles tabular sports data well, fast to train/retrain |
| Optimization | PuLP (ILP solver) | Standard tool for constrained squad-selection problems |
| Explanation layer | Claude API | Natural-language reasoning over structured + retrieved context |
| Frontend hosting | Vercel or GitHub Pages | Free static hosting |
| Data sources | FPL API, Understat/FBref, public news sources | All free/public |

## Pipeline steps (detailed)

1. **Data ingestion** — pull latest gameweek results, fixtures, prices, ownership from the official FPL API; pull xG/xA from Understat; scrape a small set of trusted news sources for injury/team news.
2. **Embedding refresh** — chunk new articles, embed them locally, update the FAISS index. Old articles age out after a few weeks to keep retrieval relevant.
3. **Feature engineering** — compute rolling form, fixture difficulty, xG/xA trends, price/ownership trends per player.
4. **Prediction** — position-specific XGBoost models output raw predicted points for the upcoming gameweek. Retrained periodically as more real season data accumulates.
5. **RAG adjustment** — for players under consideration, retrieve relevant recent news and use it to flag or adjust the raw prediction (e.g. discount a nailed-on starter flagged as a doubt).
6. **Optimization** — feed adjusted predictions into an ILP solver to pick the optimal squad under FPL's real constraints (£100m budget, 2 GK/5 DEF/5 MID/3 FWD, max 3 per club, valid XI formation).
7. **Explanation generation** — one Claude API call produces a plain-English rationale for the suggested team and standout picks, citing both stats and retrieved news.
8. **Publish** — write results as static JSON, commit/push to the repo, static site rebuilds and serves the update to all visitors.

## Cost breakdown

Designed to run at effectively **$0/month**:

| Component | Cost |
|---|---|
| Hosting (Vercel / GitHub Pages) | $0 |
| Weekly compute (GitHub Actions) | $0 (free tier covers a once-weekly job easily) |
| Structured data (SQLite in repo) | $0 |
| Vector store (FAISS in repo) | $0 |
| Embeddings (local model) | $0 |
| FPL API / Understat / news scraping | $0 (public data) |
| Claude API (~50 short calls + 1 summary, once/week) | ~$0.10–1/month |

The only non-zero line item is LLM API usage, and it stays small because explanations are generated **once per week and cached as static output**, not recomputed per visitor.

## Validation approach

Before trusting the RAG/LLM layer, the prediction model is backtested against past gameweeks using **walk-forward validation** — never training on data from after the gameweek being predicted, to avoid leakage. The RAG-adjusted predictions are compared against the stats-only baseline over time to confirm the qualitative layer is actually improving accuracy, not just adding noise.

## Roadmap / stretch goals

- [ ] Per-user personalized squads (would require auth + more frequent compute — changes the cost profile)
- [ ] Elite-manager ownership data as an additional feature signal
- [ ] Transfer/chip strategy engine (wildcard, bench boost, triple captain timing)
- [ ] Historical accuracy dashboard (model vs. actual results over the season)

## Running it

Everything runs directly with `python` (no packaging, no `__init__.py`).

```bash
pip install -r requirements.txt

python src/db.py                        # create / migrate data/fpl.db
python src/ingest/fetch_fpl.py          # player identity + prices (live FPL API)
python src/ingest/fetch_gameweek_stats.py   # current-season gameweek stats
python src/ingest/backfill_historical.py     # one-time: 5 past seasons (vaastav)

python src/features/build_features.py
python src/models/train.py --backtest   # walk-forward MAE / RMSE per position
python src/models/train.py              # fit + save production models
python src/models/predict.py            # raw points for the next gameweek

python src/rag/embed.py                 # scrape + embed news -> FAISS
python src/rag/adjust.py                # bounded news adjustment of predictions
python src/optimize/squad_optimizer.py  # ILP: 15 + XI + captain
python src/explain/generate_explanation.py   # needs ANTHROPIC_API_KEY (falls back to a template)

python src/pipeline.py                  # all of the above except the historical backfill
pytest                                  # aggregation + optimizer constraint tests
```

The static site in `site/` reads `site/data/*.json`; serve it with any static
server (`python -m http.server -d site`).

### Deploying

The project isn't a git repo yet. To get the weekly automation and the public
site running:

1. `git init`, commit, push to a GitHub repo.
2. Add `ANTHROPIC_API_KEY` as an Actions secret.
3. Enable GitHub Pages (source: GitHub Actions).

`.github/workflows/weekly-pipeline.yml` then runs the pipeline every Monday
morning UTC (and on demand), caching `data/` between runs and committing fresh
`site/data/*.json`; `pages.yml` publishes the site on each change.

See `DECISIONS.md` for the non-obvious engineering choices.

## Status

Implemented end to end: ingestion (live + historical), feature engineering,
per-position XGBoost models with walk-forward backtesting, RAG news adjustment,
ILP squad optimisation, LLM explanation, static site, and the weekly GitHub
Actions pipeline.