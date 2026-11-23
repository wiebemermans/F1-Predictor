# CLAUDE.md

F1 race-outcome predictor. Predicts the finishing order of a race **after qualifying, before the
race starts**, using only earlier races plus the qualifying result of that race. Data comes from
the Jolpica-F1 API (Ergast-compatible). Personal hobby project, built phase by phase.

## Required reading

- Working rules for LLMs (Dutch, binding): @.github/agent-instructions.md
- Phased plan with checkboxes (Dutch): `F1-voorspelmodel-stappenplan.md`. Check which phase is
  active before starting work, and tick boxes / update the phase's docs when a task is done.
- Data columns and which ones are pre-race vs. labels: `docs/data_schema.md`.
- Baseline and backtest rules (prediction moment, tie-breaks, metrics): `docs/phase5_baselines.md`.
- Feature definitions, leakage tests and the experiment log: `docs/features.md`.

## Language

Docs, docstrings, comments, error messages, and log messages are written in **Dutch**. Keep that
convention in new code and docs. Identifiers are English.

## Commands (run with `uv`, from the repo root; shell is PowerShell on Windows)

```powershell
uv sync --locked                 # install
uv run pytest                    # tests (offline, synthetic data)
uv run ruff check .              # lint
uv run ruff format --check .     # format check (CI runs all three)
uv run f1 fetch --start 2014 --end 2026   # fetch data -> data/raw/results_{season}.parquet
uv run f1 features                        # feature table -> data/processed/features.parquet
uv run python -m f1_predictor.evaluation.explore   # phase 4 report -> output/ + docs/phase4_exploration.md
uv run python -c "from f1_predictor.evaluation.backtest import run_baseline_backtest; run_baseline_backtest()"
uv run python -c "from f1_predictor.evaluation.backtest import run_feature_backtest; run_feature_backtest()"
```

Python >= 3.13, Ruff line length 100, rules E/F/I/UP/B.

## Layout

- `src/f1_predictor/data/` – `jolpica.py` (paged, cached, retrying client), `parse.py` (API JSON ->
  rows), `fetch.py` (per-season table, atomic parquet write).
- `src/f1_predictor/models/baselines.py` – `RacePredictor` protocol: `fit(train_df) -> self`,
  `predict(race_df) -> pd.Series` of unique ranks 1..n with the same index as `race_df`.
  Every future model must implement this interface. `models/linear.py` – `RidgeFeaturePredictor`,
  the phase 6 measuring tool for feature groups.
- `src/f1_predictor/evaluation/` – `metrics.py` (`score_race`), `backtest.py` (`run_backtest`,
  walk-forward over `(season, round)`; `run_feature_backtest` for cumulative feature groups),
  `explore.py` (phase 4 analysis + report generator).
- `src/f1_predictor/features/` – `groups.py` (one function per feature group, plus the shared
  `retirement_mask`), `build.py` (`FeatureConfig`, `build_features`, `feature_columns`,
  `write_feature_table`). Models read only `feature_columns(config)`; the leakage tests in
  `tests/test_features.py` run with all groups enabled, so new groups are covered.
- `src/f1_predictor/paths.py` – all paths (`RAW_DIR`, `CACHE_DIR`, `PROCESSED_DIR`, `OUTPUT_DIR`).
  Use these instead of relative paths.
- `config/` – empty; `config.yaml` arrives in phase 8 and will be loaded into `FeatureConfig`.
- `data/` is git-ignored; `output/` CSVs are committed on purpose.

## Domain rules that are easy to get wrong

- **No leakage.** Features for race N use only races before N plus N's qualifying. Use
  `shift(1)` before rolling/expanding per driver; aggregate per (team, race) before shifting
  team features. `finish_position`, `position_text`, `status`, `points`, `laps`,
  `sprint_points` of race N are labels only.
- `quali_position` (known after qualifying) ≠ `grid` (final grid, may include penalties;
  `0` = pit-lane start). Never mix them between a model and the baseline it is compared with.
- Raw data stays untouched. Special cases (grid 0, DNS, DSQ, missing qualifying) are handled and
  documented in the feature/metric that uses them.
- `status` values changed in the source around 2023-2024: older seasons use `"+1 Lap"`,
  `"Engine"`, etc.; newer ones use `"Lapped"`, `"Retired"`, `"Did not start"`, `"Disqualified"`.
  Any retirement/DNF logic must handle both vocabularies.
- Compare models only via `run_backtest` on identical races with the same `min_train_races`.

## Git

- Feature branches named `feat/...`, `docs/...`, `chore/...`, `fix/...`; merge via PR into `main`.
  Never push to `main` unless explicitly asked. Remote: `origin`
  (github.com/wiebemermans/F1-Predictor).
- The agent instructions mention a Copilot co-author line; that applies to Copilot's commits.
  Commits made by Claude use Claude's own co-author line.
