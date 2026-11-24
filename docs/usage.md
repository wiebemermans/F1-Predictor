# Gebruik: configuratie, CLI en uitvoer

Alles loopt via één command line interface (`uv run f1 ...`). Elk commando leest de
instellingen uit `config/config.yaml`. Dit document beschrijft de configuratievelden, de
commando's, de uitvoerbestanden met het JSON-formaat voor een website, en de meest
voorkomende fouten.

## Snel starten

Van een lege machine (met Git en [uv](https://docs.astral.sh/uv/)) tot een voorspelling:

```powershell
git clone https://github.com/wiebemermans/F1-Predictor.git; cd F1-Predictor
uv sync --locked
uv run f1 fetch
uv run f1 predict --season 2026 --round 22
```

`f1 fetch` haalt alle seizoenen vanaf 2014 op (enkele minuten bij de eerste keer; daarna
alleen het lopende seizoen). `f1 predict` voorspelt de finishvolgorde van één race. Dat kan
pas **na de kwalificatie**: daarvoor is de deelnemerslijst met kwalificatieposities nodig.
Het commando toont een tabel en schrijft de voorspelling als JSON naar `output/predictions/`.

## Configuratie: `config/config.yaml`

Een ander bestand gebruiken kan met `uv run f1 --config pad/naar/config.yaml <commando>`.
Onbekende sleutels (bv. een tikfout) en ongeldige waarden worden geweigerd met een melding die
de sleutel noemt. De secties `data`, `features` en `backtest` zijn optioneel; ontbrekende velden
krijgen de default uit de tabel. De sectie `model` is verplicht.

| Veld | Default | Toegestane waarden | Betekenis |
|---|---|---|---|
| `data.start_season` | `2014` | geheel getal vanaf 1950 | eerste seizoen voor `f1 fetch` |
| `data.end_season` | `null` | `null` of geheel getal ≥ `start_season` | laatste seizoen voor `f1 fetch`; `null` is het huidige kalenderjaar |
| `features.groups` | alle zes groepen | `starting_position`, `driver_form`, `team_form`, `championship`, `circuit_history`, `teammate` | welke featuregroepen gebouwd worden; moet minstens de groepen van elk model in `model.tuned` bevatten |
| `features.form_windows` | `[3, 5]` | unieke positieve gehele getallen | vensters (in races) voor coureurs- en teamvorm |
| `features.rate_window` | `10` | positief geheel getal | venster voor uitvalpercentage en teamgenoot-winstpercentage |
| `features.circuit_shrinkage` | `3.0` | eindig getal ≥ 0 | krimp van de circuitafwijking naar 0 bij weinig historie |
| `model.final` | (verplicht) | een familie uit `model.tuned` | eindmodel voor `f1 train` en `f1 predict` |
| `model.tuned.<familie>` | (verplicht) | `ridge`, `xgb_regressor`, `xgb_ranker`, `xgb_ranker_ndcg` | de gekozen configuratie per modelfamilie; `f1 backtest` vergelijkt ze met de baselines |
| `model.tuned.<familie>.feature_set` | `all` | `all`, `upto_championship` | featureset van het model |
| `model.tuned.<familie>.params` | `{}` | Ridge: alleen `alpha`. XGBoost: elke XGBoost-parameter behalve `objective` en `subsample` | hyperparameters; ontbrekende XGBoost-parameters krijgen de defaults uit `DEFAULT_XGB_PARAMS` |
| `backtest.min_train_races` | `20` | geheel getal ≥ 0 | de eerste races die alleen trainingshistorie zijn; ook het minimum aan eerdere races voor `f1 predict` |

De featuregroepen staan in [features.md](features.md), de modelfamilies en de tuning in
[model.md](model.md). `subsample` is verboden omdat rijsampling in XGBoost afhangt van het
aantal threads: dezelfde configuratie zou per machine een ander model geven.

De vastgelegde keuze in `model` volgt uit de selectieregels van fase 7, toegepast op
`output/model_tuning.csv`. De test `test_configured_models_follow_from_the_selection_rules`
bewaakt dat. Kies je bewust een ander model, pas dan ook die test aan en leg de reden vast in
[model.md](model.md). Na een wijziging in `features` is de tuning niet meer vergelijkbaar;
voer dan `f1 tune` opnieuw uit.

## Commando's

Algemene opties staan vóór het commando: `--config`/`-c` (configuratiebestand) en
`--verbose`/`-v` (debuglogging). Statusmeldingen en fouten gaan via logging naar stderr; alleen
de voorspellingstabel van `f1 predict` komt op stdout. Bij een fout eindigt elk commando met
exitcode 1; ongeldige of ontbrekende opties geven exitcode 2.

| Commando | Wat het doet | Duur |
|---|---|---|
| `f1 fetch [--start J] [--end J] [--force]` | haalt seizoenen op bij Jolpica naar `data/raw/` ([data_pipeline.md](data_pipeline.md)) | minuten (eerste keer) |
| `f1 features` | bouwt `data/processed/features.parquet` | seconden |
| `f1 backtest [--suite models\|features\|baselines]` | walk-forward-backtest; standaard de baselines en de modellen uit `model.tuned` | ~1 min |
| `f1 tune` | tuningrooster op de validatieraces 2015–2021; meldt of de regels een andere keuze geven dan de configuratie, maar past die niet aan | ~45 min |
| `f1 train` | traint het eindmodel op alle races en schrijft model, metadata en featurebelang | seconden |
| `f1 predict --season J --round R` | voorspelt één race na de kwalificatie | seconden |

Voorbeelden:

```powershell
uv run f1 fetch --start 2020 --end 2021   # alleen 2020 en 2021
uv run f1 fetch --force                    # ook afgesloten seizoenen opnieuw ophalen
uv run f1 backtest --suite baselines       # fase 5: output/backtest_baselines.csv
uv run f1 backtest --suite features        # fase 6: output/backtest_features.csv
uv run f1 -v predict --season 2026 --round 22
```

### Hoe `f1 predict` werkt

1. De lokale data moet alle eerdere ronden van het seizoen bevatten; anders volgt de melding
   om eerst `f1 fetch` uit te voeren.
2. Staat de race nog niet in de lokale data, dan komen de deelnemers, teams en
   kwalificatieposities live van Jolpica (zonder cache, zodat een lege respons van vóór de
   kwalificatie niet blijft hangen). Staat de race er al in, dan is de voorspelling
   **retrospectief**: de uitslag wordt gewist en alleen eerdere races tellen
   (`result_known: true` in de JSON).
3. Het eindmodel (`model.final`) wordt opnieuw getraind op alle races vóór deze race. Dat
   duurt enkele seconden en geeft exact dezelfde voorspelling als deze race in de backtest.
   Een apart opgeslagen model is daarvoor niet nodig.
4. De voorspelling gaat naar `output/predictions/{seizoen}-{ronde}.json`; `latest.json` wordt
   een kopie van de chronologisch laatste voorspelde race (een test op een oude race vervangt
   de voorspelling voor de komende race dus niet).

Beperkingen:

- Het veld is de kwalificatie-uitslag. Een coureur zonder kwalificatieresultaat ontbreekt in
  de voorspelling; in de historische data is het veld de race-uitslag, inclusief niet-starters.
- Gridstraffen die na de kwalificatie bekend worden, zitten er niet in (`quali_position`, zie
  [phase5_baselines.md](phase5_baselines.md)).
- Er is nog geen kalender: `--season` en `--round` zijn verplicht. Een ronde na het einde van
  het seizoen wordt gemeld als ontbrekende ronden of ontbrekende kwalificatie. Het automatisch
  bepalen van de volgende race volgt in fase 10.
- Voor ronde 1 van een nieuw seizoen kan `f1 predict` niet controleren of het vorige seizoen
  lokaal compleet is. Voer na de laatste race van het jaar dus `f1 fetch` uit.

## Uitvoerbestanden

| Bestand | Commando | In git | Inhoud |
|---|---|---|---|
| `data/raw/results_{seizoen}.parquet` | `fetch` | nee | ruwe uitslagen, kwalificatie en sprintpunten ([data_schema.md](data_schema.md)) |
| `data/processed/features.parquet` | `features` | nee | ruwe kolommen plus features |
| `output/backtest_baselines.csv` | `backtest --suite baselines` | ja | scores per race van de baselines |
| `output/backtest_features.csv` | `backtest --suite features` | ja | scores per race van de feature-experimenten |
| `output/backtest_models.csv` | `backtest` | ja | scores per race van baselines en gekozen modellen |
| `output/model_comparison.csv` | `backtest` | ja | samenvatting per periode (validatie, holdout, alles) |
| `output/model_comparison.json` | `backtest` | ja | dezelfde samenvatting als JSON voor een website |
| `output/model_tuning.csv` | `tune` | ja | samenvatting per configuratie op de validatieraces |
| `output/models/<model>.joblib` en `.json` | `train` | nee | getraind eindmodel en metadata (opnieuw te bouwen) |
| `output/model_feature_importance.csv` | `train` | ja | featurebelang van het eindmodel |
| `output/predictions/{seizoen}-{ronde}.json` | `predict` | ja | voorspelling per race |
| `output/predictions/latest.json` | `predict` | ja | kopie van de laatste voorspelde race |

Alle bestanden worden atomisch geschreven: eerst naar een tijdelijk bestand, daarna hernoemd.

## JSON-formaat (contract voor een website)

Een website hoeft alleen deze bestanden te lezen en niets van de Python-code te weten. De
regels:

- UTF-8, met inspringing. Ontbrekende waarden zijn `null`, nooit `NaN`.
- `schema_version` is nu `1`. Nieuwe velden kunnen erbij komen zonder versiewissel; een veld
  verwijderen, hernoemen of van betekenis veranderen verhoogt `schema_version`.
- Er staan geen volledige namen van coureurs of teams in, omdat de ruwe data die niet bevat.
  Toon `driver_code` (bv. `VER`) en `constructor_id`.

### Voorspelling: `output/predictions/{seizoen}-{ronde}.json`

Ingekort voorbeeld (een retrospectieve voorspelling van ronde 21 van 2026):

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-24T20:40:27+00:00",
  "race": {"season": 2026, "round": 21, "race_name": "Las Vegas Grand Prix", "circuit_id": "vegas", "date": "2026-11-22"},
  "result_known": true,
  "prediction_moment": "na de kwalificatie, vóór de race (quali_position, zonder latere gridstraffen)",
  "model": {
    "name": "xgb_ranker_d2_n50_all",
    "family": "xgb_ranker",
    "feature_set": "all",
    "params": {"max_depth": 2, "n_estimators": 50, "learning_rate": 0.05, "colsample_bytree": 0.8, "tree_method": "hist", "random_state": 0},
    "training": {"first_race": {"season": 2014, "round": 1}, "last_race": {"season": 2026, "round": 20}, "races": 272, "rows": 5545}
  },
  "predictions": [
    {"predicted_position": 1, "driver_id": "antonelli", "driver_code": "ANT", "constructor_id": "mercedes", "quali_position": 1},
    {"predicted_position": 2, "driver_id": "max_verstappen", "driver_code": "VER", "constructor_id": "red_bull", "quali_position": 2}
  ],
  "run": {"git_commit": "951eebb…", "git_dirty": true, "package_version": "0.1.0", "versions": {"python": "3.13.14", "xgboost": "3.4.1", "scikit-learn": "1.9.1", "pandas": "3.0.6"}, "config": {}}
}
```

| Veld | Betekenis |
|---|---|
| `generated_at` | tijdstip van de voorspelling (UTC) |
| `race` | seizoen, ronde, racenaam, `circuit_id` en racedatum (`JJJJ-MM-DD`) |
| `result_known` | `true`: de race had al een uitslag toen de voorspelling gemaakt werd (retrospectief, geen echte voorspelling vooraf) |
| `prediction_moment` | het voorspelmoment in woorden |
| `model` | naam, familie, featureset, de werkelijk gebruikte parameters en de trainingsraces |
| `predictions` | één rij per coureur, gesorteerd op `predicted_position` (uniek, 1..n); `driver_code` en `quali_position` kunnen `null` zijn |
| `run` | herkomst: commit, of er niet-gecommitte wijzigingen waren, versies en de volledige configuratie |

### Modelvergelijking: `output/model_comparison.json`

Dezelfde cijfers als `model_comparison.csv`: per periode (`validation` 2015–2021, `holdout`
2022 en later, `all`) één rij per model.

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-24T20:38:44+00:00",
  "final_model": "xgb_ranker_d2_n50_all",
  "reference": "qualifying_position",
  "periods": {
    "holdout": [
      {"model": "qualifying_position", "family": "baseline", "races": 113, "mae_positions": 3.388, "spearman": 0.655, "winner_hit": 0.593, "podium_overlap": 1.929, "top10_overlap": 7.788, "delta_mae": 0.0, "delta_mae_se": 0.0, "better_share": 0.0, "worse_share": 0.0}
    ]
  },
  "run": {}
}
```

De metrics staan in [phase5_baselines.md](phase5_baselines.md); `delta_mae` en de andere
vergelijkingskolommen in [model.md](model.md). `family` is `baseline` voor de baselines.
`final_model` is de naam van het eindmodel, zodat een website dat kan uitlichten.

## Herkomst en reproduceerbaarheid

- Elke JSON-uitvoer bevat een `run`-blok met de git-commit en `git_dirty`. `git_dirty` telt
  ook niet-gevolgde bestanden mee. Zonder git (bv. een gedownloade zip) zijn beide `null`.
  Een gecommit uitvoerbestand verwijst naar de commit waarop het gemaakt is, dus naar een
  eerdere commit dan die waarin het zelf zit.
- De metadata van `f1 train` (`output/models/<model>.json`) bevat dezelfde git-informatie.
- De seeds liggen vast: `random_state=0` in `DEFAULT_XGB_PARAMS`, geen `subsample`, en Ridge
  is deterministisch. Dezelfde data, configuratie en bibliotheekversies geven dezelfde uitvoer;
  een herhaalde `f1 backtest` gaf byte-identieke CSV's.

## Veelvoorkomende fouten

| Melding | Oorzaak en oplossing |
|---|---|
| `Geen results_*.parquet-bestanden gevonden … Haal eerst data op met f1 fetch.` | er is nog geen lokale data: voer `uv run f1 fetch` uit |
| `Nog geen kwalificatie-uitslag voor 2026 ronde 22; voorspel na de kwalificatie` | de kwalificatie is nog niet verreden of nog niet in Jolpica verwerkt; probeer het na de kwalificatie opnieuw |
| `Ronde(s) [21] van 2026 ontbreken in de lokale data` | de uitslag van een eerdere ronde ontbreekt lokaal: voer `uv run f1 fetch` uit (het lopende seizoen wordt altijd ververst) |
| `De lokale data loopt tot seizoen 2025, te vroeg voor 2027` | er ontbreken hele seizoenen: voer `uv run f1 fetch` uit |
| `Te weinig eerdere races om op te trainen` | minder eerdere races dan `backtest.min_train_races`: haal meer seizoenen op |
| `Onbekende sleutel(s) in …` / `Onbekende XGBoost-parameter …` | tikfout in de configuratie; de melding noemt de sleutel |
| `Configuratiebestand niet gevonden` | verkeerd pad bij `--config` |
| `Jolpica-API niet bereikbaar` | geen netwerk of de API is tijdelijk onbereikbaar; `f1 fetch` probeert HTTP 429/5xx tot vijf keer opnieuw |
| `nog geen race-uitslagen, seizoen overgeslagen` (waarschuwing) | het lopende jaar heeft nog geen race gehad; dat is geen fout en `f1 fetch` eindigt met exitcode 0 |
