# F1 Predictor

Een machine-learningmodel dat de uitslag van Formule 1-races voorspelt, getest met een walk-forward backtest tegen eenvoudige baselines (kwalificatievolgorde en kampioenschapsstand).

![CI](https://github.com/wiebemermans/F1-Predictor/actions/workflows/ci.yml/badge.svg)

Data: [Jolpica-F1](https://github.com/jolpica/jolpica-f1) API.

## Laatste voorspelling

<!-- voorspelling:start -->
Nog geen voorspelling. Na de volgende kwalificatie verschijnt hier de voorspelde top 3.
<!-- voorspelling:end -->

Dit blok wordt automatisch bijgewerkt na elke kwalificatie ([docs/automation.md](docs/automation.md)).

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
- [Gebruik: configuratie, CLI-commando's en JSON-uitvoer](docs/usage.md)
- [Fase 9: tests, coverage, integratietest en CI](docs/testing.md)
- [Fase 10: automatisering met GitHub Actions](docs/automation.md)
- [Volledig stappenplan](F1-voorspelmodel-stappenplan.md)

## Snel starten

Met Git en [uv](https://docs.astral.sh/uv/) geïnstalleerd zijn vier commando's genoeg voor een
voorspelling:

```powershell
git clone https://github.com/wiebemermans/F1-Predictor.git; cd F1-Predictor
uv sync --locked
uv run f1 fetch                                # seizoenen 2014 t/m nu -> data/raw/
uv run f1 predict --season 2026 --round 22     # na de kwalificatie, vóór de race
```

`f1 predict` toont de voorspelde finishvolgorde en schrijft die als JSON naar
`output/predictions/`. Een voorspelling kan pas na de kwalificatie: die levert de deelnemers en
startposities. Instellingen staan in [`config/config.yaml`](config/config.yaml); alle
commando's, configuratievelden en het JSON-formaat staan in [docs/usage.md](docs/usage.md).

Elk raceweekend gebeurt dat ook automatisch: een GitHub Actions-workflow voorspelt de volgende
race zodra de kwalificatie verwerkt is (`f1 predict --next`), commit de JSON in
`output/predictions/` en zet de top 3 bovenaan deze README (`f1 readme`). Na de race worden
de backtests bijgewerkt en zet `output/prediction_history.json` elke voorspelling van vooraf
naast de uitslag. Zie [docs/automation.md](docs/automation.md).

## Backtests en model

```powershell
uv run f1 backtest --suite baselines   # fase 5: kwalificatie- en kampioenschapsbaseline
uv run f1 features                     # fase 6: featuretabel -> data/processed/features.parquet
uv run f1 backtest --suite features    # fase 6: featuregroepen één voor één
uv run f1 tune                         # fase 7: hyperparameters op 2015-2021 (~45 min)
uv run f1 backtest                     # fase 7: gekozen modellen tegen de baselines
uv run f1 train                        # fase 7: eindmodel opslaan en interpreteren
```

De backtest is walk-forward: elke race wordt voorspeld met alleen eerdere races als training,
de eerste 20 races zijn alleen trainingshistorie. Op de holdout (2022–2026) verslaan de
modellen de kwalificatiebaseline bescheiden op MAE en Spearman, maar voorspellen ze de winnaar
niet vaker juist dan de regel "de polesitter wint". Resultaten, interpretatie en beperkingen
staan in [docs/model.md](docs/model.md); de featurekeuze in het
[experimentenlogboek](docs/features.md#experimentenlogboek).

> 🚧 Work in progress
