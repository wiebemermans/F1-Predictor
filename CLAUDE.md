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
- Model variants, tuning protocol (validation 2015–2021, holdout 2022+) and results:
  `docs/model.md`.
- CLI commands, `config/config.yaml` fields, output files and the JSON contract for a website:
  `docs/usage.md`.
- Test overview, coverage floor, the end-to-end CLI test, leakage checks and rules for new
  tests: `docs/testing.md`.
- GitHub Actions automation (schedule, `predict --next` decisions, bot commits to `main`,
  caching, troubleshooting): `docs/automation.md`.

## Language

Docs, docstrings, comments, error messages, and log messages are written in **Dutch**. Keep that
convention in new code and docs. Identifiers are English.

## Commands (run with `uv`, from the repo root; shell is PowerShell on Windows)

```powershell
uv sync --locked                 # install
uv run pytest                    # tests (offline, synthetic data)
uv run pytest --cov              # tests + branch coverage; fails below 90% (as in CI)
uv run ruff check .              # lint
uv run ruff format --check .     # format check (CI runs all three)
uv run f1 fetch                           # data -> data/raw/results_{season}.parquet (config: data)
uv run f1 features                        # feature table -> data/processed/features.parquet
uv run f1 backtest --suite baselines      # phase 5 -> output/backtest_baselines.csv
uv run f1 backtest --suite features       # phase 6 -> output/backtest_features.csv
uv run f1 tune                            # phase 7 grid on validation seasons -> output/model_tuning.csv (~45 min)
uv run f1 backtest                        # baselines + model.tuned -> backtest_models.csv, model_comparison.csv/.json
uv run f1 train                           # model.final -> output/models/ + output/model_feature_importance.csv
uv run f1 predict --season 2026 --round 22   # after qualifying -> output/predictions/2026-22.json + latest.json
uv run f1 predict --next                  # next race from the calendar; exit 0 with "niets te voorspellen" before qualifying
uv run f1 evaluate                        # pre-race predictions vs. results -> output/prediction_history.json
uv run f1 readme                          # top 3 of the latest pre-race prediction -> block in README.md
uv run python -m f1_predictor.evaluation.explore   # phase 4 report -> output/ + docs/phase4_exploration.md
```

Every `f1` command reads `config/config.yaml` (`--config PATH` for another file). Library functions
take explicit arguments; only the CLI reads the config.

Python >= 3.13, Ruff line length 100, rules E/F/I/UP/B.

## Layout

- `src/f1_predictor/data/` – `jolpica.py` (paged, cached, retrying client), `parse.py` (API JSON ->
  rows), `fetch.py` (per-season table, atomic parquet write; `fetch_race_entry` gets the
  qualifying entry list of one race, uncached, for `predict`; `fetch_schedule` gets a season's
  calendar with race/qualifying start in UTC, uncached).
- `src/f1_predictor/models/baselines.py` – `RacePredictor` protocol: `fit(train_df) -> self`,
  `predict(race_df) -> pd.Series` of unique ranks 1..n with the same index as `race_df`.
  Every future model must implement this interface. `models/linear.py` – `RidgeFeaturePredictor`
  (phase 6 measuring tool, phase 7 linear model). `models/boosting.py` – `XGBRegressorPredictor`,
  `XGBRankerPredictor` (`rank:pairwise`, one group per race), `XGBNDCGRankerPredictor`
  (`rank:ndcg`), `DEFAULT_XGB_PARAMS`.
  `models/common.py` – shared feature-input checks. `models/train.py` – `ModelSpec`
  (family + feature set + params, rejects unknown params and `subsample`), `train_final_model`
  (joblib + metadata JSON with git commit in git-ignored `output/models/`).
  `models/predict.py` – `predict_race` (retrains `model.final` on all races before the target
  race, so it equals that race's backtest fold; upcoming races get their field from live
  qualifying), `write_prediction` (JSON + `latest.json`; rewrites only on a content change and
  never replaces a pre-race prediction with a retrospective one), `next_race` /
  `race_to_predict` (calendar decision for `--next`, injectable `now`), `saved_predictions`.
- `src/f1_predictor/evaluation/` – `metrics.py` (`score_race`), `backtest.py` (`run_backtest`,
  walk-forward over `(season, round)`; `run_feature_backtest` for cumulative feature groups),
  `model_selection.py` (`summarize_backtest` with paired ΔMAE, tuning grid, `run_model_tuning`
  on validation seasons only, `run_model_backtest`, `write_comparison_json`), `history.py`
  (`f1 evaluate`: scores stored pre-race predictions against results on the drivers in both
  lists, vs. the qualifying baseline), `explore.py` (phase 4 analysis + report).
- `src/f1_predictor/features/` – `groups.py` (one function per feature group, plus the shared
  `retirement_mask`), `build.py` (`FeatureConfig`, `build_features`, `feature_columns`,
  `write_feature_table`). Models read only `feature_columns(config)`; the leakage tests in
  `tests/test_features.py` run with all groups enabled, so new groups are covered.
- `src/f1_predictor/config.py` – `load_config` -> `ProjectConfig` (data, features as
  `FeatureConfig`, model with `tuned` per family and `final`, backtest); unknown keys are
  rejected. `config/config.yaml` holds the phase 7 choice. `cli.py` – Typer commands.
- `src/f1_predictor/runinfo.py` (git commit/dirty, versions, config for the `run` block) and
  `output.py` (atomic writes, `write_json` with NaN -> null, `write_json_if_changed` which
  ignores `generated_at` and `run` so scheduled runs without new data commit nothing). Bump
  `schema_version` when a JSON field is removed or renamed (contract in `docs/usage.md`).
  `readme.py` – `update_readme` (`f1 readme`): writes the top 3 of the latest pre-race
  prediction between `<!-- voorspelling:start -->` / `<!-- voorspelling:end -->` in
  `README.md`, only when that block changes; keeps the file's line endings.
- `.github/workflows/` – `ci.yml` (lint + tests), `predict.yml` (every 1.5 h Fri–Sun, two cron
  lines at :17 and :47: `fetch` + `predict --next` + `readme`) and `post-race.yml` (Mon/Tue:
  `fetch`, three backtest suites, `evaluate`). The two automation workflows push only
  `output/` (and `predict.yml` the README block) to `main` as github-actions[bot];
  `tests/test_workflows.py` checks that every `uv run f1 ...` in them exists.
- `src/f1_predictor/paths.py` – all paths (`CONFIG_PATH`, `README_PATH`, `RAW_DIR`, `CACHE_DIR`,
  `PROCESSED_DIR`, `OUTPUT_DIR`, `PREDICTIONS_DIR`). Use these instead of relative paths.
  Library defaults like `data_dir=RAW_DIR` are bound at import, so tests pass explicit paths.
- `tests/` – one file per module area; `test_models.py` holds the contract test every
  `RacePredictor` (both baselines + each family in `MODEL_FAMILIES`) must pass.
  `test_pipeline.py` runs the whole CLI on `tests/fixtures/jolpica_mini.json` with paths
  redirected to a temp folder (`_redirects`), plus a race-weekend run with a fixed clock; a new
  command that writes files needs a redirect there, or the repo-snapshot guard (`data/`,
  `output/`, `README.md`) fails the test. Tests never read `data/` and never write to
  `output/` or `README.md`.
- `data/` is git-ignored; `output/` CSVs and JSON are committed on purpose.

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
- Hyperparameters and model choice are made on validation races (≤ 2021) only; the holdout
  (≥ 2022) is for reporting. Don't tune on holdout scores. Rules: hyperparameters per family by
  lowest validation MAE; final model by highest validation podium overlap (tie: winner hit
  rate, then MAE), see `select_final_model`. The choice lives in `config/config.yaml`;
  `test_configured_models_follow_from_the_selection_rules` checks it against
  `output/model_tuning.csv`. XGBoost models use no `subsample`: row sampling depends on the
  thread count and breaks reproducibility across machines.

## Git

- Feature branches named `feat/...`, `docs/...`, `chore/...`, `fix/...`; merge via PR into `main`.
  Never push to `main` unless explicitly asked. Remote: `origin`
  (github.com/wiebemermans/F1-Predictor). The automation bot pushing `output/` (and the README
  block) to `main` is the user's explicit choice (phase 10); this rule is for agents. Pull before
  working on `output/` or `README.md` locally: the bot commits there on race weekends.
- The agent instructions mention a Copilot co-author line; that applies to Copilot's commits.
  Commits made by Claude use Claude's own co-author line.
