# F1 Predictor

Een machine-learningmodel dat de uitslag van Formule 1-races voorspelt, getest met een walk-forward backtest tegen eenvoudige baselines (kwalificatievolgorde en kampioenschapsstand).

![CI](https://github.com/wiebemermans/F1-Predictor/actions/workflows/ci.yml/badge.svg)

Data: [Jolpica-F1](https://github.com/jolpica/jolpica-f1) API.

## Transparantie

Dit project is grotendeels ontwikkeld met hulp van een large language model (LLM). De code en resultaten worden waar mogelijk gecontroleerd met tests, linting en reproduceerbare analyses; LLM-gebruik vervangt die controles of menselijke beoordeling niet.

## Documentatie

- [Ontwikkelomgeving en kwaliteitscontroles](docs/development.md)
- [Datapipeline en data ophalen](docs/data_pipeline.md)
- [Dataschema en leakagegrenzen](docs/data_schema.md)
- [Fase 4: data-exploratie](docs/phase4_exploration.md)
- [Fase 5: baselines en backtestmethode](docs/phase5_baselines.md)
- [Fase 6: features, leakage-tests en experimentenlogboek](docs/features.md)
- [Fase 7: model, tuning en vergelijking met de baselines](docs/model.md)
- [Volledig stappenplan](F1-voorspelmodel-stappenplan.md)

## Baseline-backtest

Voer de walk-forward-backtest van de kwalificatie- en kampioenschapspuntenbaseline uit:

```powershell
uv run python -c "from f1_predictor.evaluation.backtest import run_baseline_backtest; run_baseline_backtest()"
```

De backtest gebruikt alleen eerdere races als trainingshistorie en slaat scores per race op in
`output/backtest_baselines.csv`. De eerste 20 races worden gebruikt als minimale trainingshistorie
en krijgen zelf geen score.

## Features

Bouw de featuretabel (`data/processed/features.parquet`) en meet de featuregroepen in dezelfde
walk-forward-backtest:

```powershell
uv run f1 features
uv run python -c "from f1_predictor.evaluation.backtest import run_feature_backtest; run_feature_backtest()"
```

De scores per experiment staan in `output/backtest_features.csv`; de conclusies staan in het
[experimentenlogboek](docs/features.md#experimentenlogboek).

## Model

Kies hyperparameters op de validatieraces (2015–2021), vergelijk de gekozen modellen met de
baselines en train het eindmodel (een XGBoost-ranker):

```powershell
uv run python -c "from f1_predictor.evaluation.model_selection import run_model_tuning; run_model_tuning()"
uv run python -c "from f1_predictor.evaluation.model_selection import run_model_backtest; run_model_backtest()"
uv run python -c "from f1_predictor.models.train import train_final_model; train_final_model()"
```

Op de holdout (2022–2026) verslaan de modellen de kwalificatiebaseline bescheiden op MAE en
Spearman, maar voorspellen ze de winnaar niet vaker juist dan de regel "de polesitter wint". Resultaten, interpretatie en beperkingen staan
in [docs/model.md](docs/model.md).

> 🚧 Work in progress
