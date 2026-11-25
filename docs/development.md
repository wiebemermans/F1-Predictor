# Ontwikkelen

## Omgeving installeren

Het project vereist Python 3.13 of nieuwer, zoals aangegeven in `pyproject.toml`.
Met [uv](https://docs.astral.sh/uv/) installeer je de vastgelegde runtime- en
ontwikkelafhankelijkheden vanuit de repository-root:

```powershell
uv sync --locked
```

Controleer de beschikbare tools en configureer Git indien dat nog niet gebeurd is:

```powershell
python --version
git --version
uv --version
git config --global user.name "Jouw Naam"
git config --global user.email "jij@voorbeeld.com"
```

De importnaam is `f1_predictor`; de installeerbare CLI heet `f1`.

## Projectindeling

- `config/config.yaml`: projectconfiguratie (seizoenen, features, gekozen modellen, backtest);
  velden en commando's staan in [usage.md](usage.md).
- `src/f1_predictor/cli.py`: de commando's `f1 fetch`, `features`, `backtest`, `tune`, `train`,
  `predict` en `evaluate`; `config.py` leest en valideert de configuratie.
- `src/f1_predictor/data/`: Jolpica-client, parsing, seizoensbestanden en de racekalender.
- `src/f1_predictor/features/`: featuregroepen (`groups.py`) en de featuretabel met
  `FeatureConfig` (`build.py`).
- `src/f1_predictor/models/`: voorspellers (baselines, Ridge, XGBoost-regressie en -ranker),
  modelspecificaties, het trainen van het eindmodel en het voorspellen van één race.
- `src/f1_predictor/evaluation/`: data-exploratie, metrics, backtests, modelselectie en de
  historiek van voorspellingen vooraf tegen de uitslag.
- `.github/workflows/`: CI (`ci.yml`) en de automatisering (`predict.yml`, `post-race.yml`, zie
  [automation.md](automation.md)).
- `tests/`: offline tests met synthetische data; `tests/fixtures/` bevat de mini-dataset in
  Jolpica-formaat voor de integratietest.
- `notebooks/`: API-verkenning en experimenten.
- `data/raw/` en `data/processed/`: lokale parquet-data (ruw en featuretabel); deze wordt
  niet in Git bijgehouden.
- `src/f1_predictor/runinfo.py` en `output.py`: herkomst van een run (git-commit, versies,
  configuratie) en het atomisch wegschrijven van CSV- en JSON-uitvoer.
- `output/`: reproduceerbare rapporten, backtestresultaten en voorspellingen
  (`output/predictions/`); `output/models/` (getrainde modellen) staat niet in git.
- `docs/`: dataschema, methode en faseverslagen.

## Kwaliteitscontroles

Voer vóór je wijzigingen commit de tests en Ruff uit:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`uv run pytest --cov` meet daarbij de coverage en faalt onder 90%. Wat elke test afdekt, de
integratietest en richtlijnen voor nieuwe tests staan in [testing.md](testing.md).

Installeer desgewenst de Git-hooks; die controleren Ruff en algemene
bestandsconventies vóór een commit:

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```

GitHub Actions voert `uv sync --locked`, Ruff en pytest met coverage
(ondergrens 90%) uit op pushes naar `main` en op pull requests, en zet de
coveragetabel in de samenvatting van de run. De CI-configuratie staat in
`.github/workflows/ci.yml`. Twee geplande workflows voorspellen elke race en werken na de race de
backtests bij; die staan in [automation.md](automation.md).
