# Engineering decisions

Short rationale for choices that aren't obvious from the code. Newest context at
the top of each section.

## Data model

**Players are keyed on the FPL `code`, not the season-specific `element` id.**
`element` ids are reassigned every season, so keying `player_gameweek_stats` on
`element` would collide the moment historical data spans more than one season. We
use the stable `code` as `players.player_id` and keep the current season's
`element` id in `players.element_id` (needed to hit `element-summary/{id}/`).
Historical rows map `element -> code` through each season's `players_raw.csv`.
`db.migrate()` drops a pre-`code` `players` table (the gameweek table is empty
whenever that applies) and `fetch_fpl.py` repopulates it.

**Season tags.** `SEASON = "2026-27"` (the live FPL API's current season as of
2026-09-01, GW2 complete). `HISTORICAL_SEASONS` is the five completed seasons
before it, `2021-22 .. 2025-26`, pulled once from
`vaastav/Fantasy-Premier-League`.

**Double gameweeks** are rolled up to one `player_gameweek_stats` row
(`src/ingest/aggregate.py`): counting stats (points, minutes, goals, assists,
bps, bonus, starts, cards, xG/xA/xGI/xGC) are **summed**; **minutes are not
capped** (a double can exceed 90); `was_home` and `opponent_team` come from the
**first** fixture by kick-off time; `now_cost` is the last fixture's price;
`is_double_gameweek = 1` when the gameweek had >1 fixture. One pure function,
shared by the live and historical ingests and unit-tested.

**`player_features` is owned by `build_features.py`**, not `db.py` — it is
rewritten wholesale (`to_sql(if_exists="replace")`) each run so its columns can
track the feature set. `data/models/feature_columns.json` is the authoritative
ordered input list for train/predict.

## Persistence & CI

**`data/*.db`, `data/models/`, `data/faiss/` are gitignored** (per the brief),
which conflicts with the README's "commit the DB/FAISS to the repo". Resolution:
the static site consumes **only `site/data/*.json`**, which *is* committed by the
weekly workflow. The DB, models and index are persisted between CI runs via
`actions/cache`; a cache miss makes the workflow run `backfill_historical.py`
before the weekly job. This keeps the repo small and the site self-contained.

**Git / GitHub is left to the user.** The folder isn't a git repo yet. The
weekly workflow and Pages deploy only do anything once the user runs `git init`,
pushes to a GitHub repo, and adds `ANTHROPIC_API_KEY` as an Actions secret.

## Modelling

**One XGBoost regressor per position** (GK/DEF/MID/FWD); shared feature columns.
Params are hand-picked (shallow depth 4, 400 trees, lr 0.05, row/col subsample
0.8) — a safe default for noisy tabular data, not grid-searched. XGBoost's native
NaN handling means early-career rows and the always-NaN ownership columns need no
imputation.

**No leakage:** every rolling/trend feature is `.shift(1)` before `.rolling(...)`,
so a row never sees its own or any later result. The walk-forward backtest
(`train.py --backtest`) trains only on `(season, gameweek)` tuples strictly
before the scored one; it refits every `--stride` gameweeks (default 3) rather
than every gameweek purely for runtime.

**Fixture difficulty** is derived from team `strength_overall_home/away`
(bootstrap for the live season, vaastav `teams.csv` for past seasons), scaled to
1–5 per season. If a season's strength table can't be fetched, `fdr` falls back
to a neutral 3.0 — difficulty is a helper signal, not load-bearing.

**Ownership features** (`ownership`, `ownership_trend_3`) are placeholders (always
NaN): `selected_by_percent` is only in the live bootstrap, not in
`player_gameweek_stats`, so there's no historical series to train on. Kept as
named columns so the feature set is stable if a snapshot source is added later.

## RAG

**News sources:** BBC Sport, Sky Sports and The Guardian football RSS feeds.
RSS over HTML scraping for stability; each feed is wrapped so one failure never
aborts the step. (Reddit's `r/FantasyPL` JSON was the original third source but
Reddit 403s datacenter IPs — including GitHub Actions — so it was swapped for
the Guardian feed.)

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, CPU),
FAISS `IndexFlatIP` over L2-normalised vectors (cosine). The index is rebuilt
from every surviving `news_chunks` row each run after chunks older than
`NEWS_MAX_AGE_DAYS` (21) are deleted.

**Adjustment is bounded and signal-gated.** A chunk only adjusts a prediction if
it contains the player's surname *and* a categorised keyword: `OUT` → ×0.70,
`DOUBT` → ×0.85, `BOOST` → ×1.15. The factor is clamped to
`[1 − ADJUSTMENT_CAP, 1 + ADJUSTMENT_CAP]` = `[0.70, 1.30]` — news can nudge a
prediction, never replace it. Factor + reason + source URL are stored on
`predictions`. If the FAISS index is missing, `adjust.py` sets adjusted = raw
and logs a warning.

## Explanation

**Model `claude-sonnet-4-6`** (as specified in the brief; a valid current model
id). The Anthropic SDK reads the key from `ANTHROPIC_API_KEY` — never hardcoded.
The call asks for a small JSON object and parses it defensively (no reliance on a
specific structured-output API shape). If `ANTHROPIC_API_KEY` is unset, or the
call fails, a deterministic template writes the same fields so `pipeline.py`
still completes (important for local runs and CI without the secret).

## Elite-template weight

The optimiser can nudge selection toward what elite managers own:

    final_score = predicted_points * (1 + ELITE_WEIGHT * elite_template_score)

`elite_template_score` is the fraction of a sampled top-`ELITE_SAMPLE_SIZE`
(100) slice of FPL's overall league (id 314) that owns the player that gameweek
(`src/ingest/fetch_elite.py`). Picks for the *upcoming* gameweek aren't
published until managers set their teams, so the optimiser (and the site export)
fall back to the most recent gameweek that has elite data — the template moves
slowly week to week. `fetch_elite.py` also backfills any finished gameweek of
the current season that has no elite row yet, so the tunable history grows on
its own.

**`ELITE_WEIGHT` is never hardcoded.** `src/optimize/tune_elite_weight.py`
sweeps `ELITE_WEIGHT_CANDIDATES` (`0.0, 0.05, 0.1, 0.15, 0.2, 0.3` — `0.0` is
the mandatory pure-stats control) by walk-forward backtest against *realized*
historical points and writes the winner to `data/elite_weight_config.json`,
which `squad_optimizer.py` reads at run time. Re-running the tuner as more elite
data accumulates updates the live weight with no code change. It runs from the
pipeline only when the prediction models are retrained (retuning needs a fresh
batch of realized results to be worth doing); the weekly steps just optimise
with the stored weight.

**Elite ownership only exists going forward.** The FPL picks endpoint covers the
current season's gameweeks only — there is no elite history for past seasons.
Until `ELITE_TUNE_MIN_GAMEWEEKS` (5) gameweeks have both elite data and realized
results, the tuner **defers**: it writes `ELITE_WEIGHT = 0.0`, status
`deferred`, and this fact is surfaced in `DECISIONS.md` (the auto-generated
section below) and in every weekly explanation, so a deferred weight is never
mistaken for a tuned one. Ties or sub-noise (`ELITE_TUNE_NOISE_PTS`) margins
also resolve to the lowest non-zero weight that matches the control, or 0.0.

## Frontend

Plain HTML/CSS/vanilla JS, no build step, fetches `./data/*.json` with relative
paths (works on a GitHub Pages project site and on Vercel). Placeholder JSON in
`site/data/` lets the page render before the first pipeline run.

## Elite-weight tuning (auto-generated)

<!-- ELITE_WEIGHT_TUNING:START -->

_Last run 2026-09-01T18:43:39.865613+00:00 — `python src/optimize/tune_elite_weight.py`._

**Chosen `ELITE_WEIGHT` = 0.0** (status: `deferred`).

only 2 gameweek(s) have both elite ownership data and realized results; need 5. Deferring to pure stats until enough elite data accumulates.


<!-- ELITE_WEIGHT_TUNING:END -->
