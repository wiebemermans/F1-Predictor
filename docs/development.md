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

- `src/f1_predictor/data/`: Jolpica-client, parsing en seizoensbestanden.
- `src/f1_predictor/features/`: featuregroepen (`groups.py`) en de featuretabel met
  `FeatureConfig` (`build.py`).
- `src/f1_predictor/models/`: voorspellers (baselines, Ridge, XGBoost-regressie en -ranker),
  modelspecificaties en het trainen van het eindmodel.
- `src/f1_predictor/evaluation/`: data-exploratie, metrics, backtests en modelselectie.
- `tests/`: offline tests met synthetische data.
- `notebooks/`: API-verkenning en experimenten.
- `data/raw/` en `data/processed/`: lokale parquet-data (ruw en featuretabel); deze wordt
  niet in Git bijgehouden.
- `output/`: reproduceerbare rapporten en backtestresultaten; `output/models/` (getrainde
  modellen) staat niet in git.
- `docs/`: dataschema, methode en faseverslagen.

## Kwaliteitscontroles

Voer vóór je wijzigingen commit de tests en Ruff uit:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Installeer desgewenst de Git-hooks; die controleren Ruff en algemene
bestandsconventies vóór een commit:

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```

GitHub Actions voert `uv sync --locked`, Ruff en pytest uit op pushes naar
`main` en op pull requests. De CI-configuratie staat in
`.github/workflows/ci.yml`.
