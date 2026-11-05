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
- [Volledig stappenplan](F1-voorspelmodel-stappenplan.md)

## Baseline-backtest

Voer de walk-forward-backtest van de kwalificatie- en kampioenschapspuntenbaseline uit:

```powershell
uv run python -c "from f1_predictor.evaluation.backtest import run_baseline_backtest; run_baseline_backtest()"
```

De backtest gebruikt alleen eerdere races als trainingshistorie en slaat scores per race op in
`output/backtest_baselines.csv`. De eerste 20 races worden gebruikt als minimale trainingshistorie
en krijgen zelf geen score.

> 🚧 Work in progress
