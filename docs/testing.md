# Fase 9: Tests

De testsuite bewaakt drie dingen: dat elke stap van de pipeline doet wat de documentatie
belooft, dat er geen informatie van na het voorspelmoment in een voorspelling lekt, en dat
uitvoer voor een website het vastgelegde JSON-formaat volgt. Alle tests draaien offline op
kleine synthetische data: geen netwerk, geen echte Jolpica-data en geen schrijfacties buiten een
tijdelijke map. De volledige suite duurt minder dan een minuut.

## Tests uitvoeren

Vanuit de repository-root:

```powershell
uv run pytest                              # alle tests
uv run pytest tests/test_pipeline.py       # alleen de integratietest
uv run pytest -k leakage                   # tests met "leakage" in de naam
uv run pytest --cov                        # alle tests plus coverage, faalt onder 90%
uv run pytest --cov --cov-report=html      # idem, met een HTML-rapport in htmlcov/
```

`uv run pytest` zonder `--cov` meet geen coverage en blijft zo sneller tijdens het werken.

## Coverage

Coverage wordt gemeten met `pytest-cov`. De instellingen staan in `pyproject.toml` onder
`[tool.coverage.run]` en `[tool.coverage.report]`:

- gemeten wordt het package `f1_predictor`, inclusief branches (beide kanten van elke `if`);
- het rapport toont per bestand de regels die geen test raakt, en slaat volledig gedekte
  bestanden over;
- onder **90%** faalt de run (`fail_under`), lokaal met `--cov` en in CI.

Bij het afronden van fase 9 was de dekking 96,8%, na fase 10 97,0%. Wat overblijft zijn vooral verdedigende
invoercontroles en het hoofdscript van het fase 4-rapport (`explore.py`). Een test die zo'n
regel alleen uitvoert zonder gedrag te controleren voegt weinig toe, dus 100% is geen doel. De
ondergrens van 90% laat ruimte voor zulke regels, maar vangt nieuwe code zonder tests
(bijvoorbeeld uit fase 10 en 11).

## Overzicht van de testbestanden

| Bestand | Wat het afdekt |
|---|---|
| `test_smoke.py` | Het package laadt. |
| `test_data.py` | Jolpica-client: paginering, retries bij 429/5xx (met `Retry-After`), direct falen bij andere 4xx, pauze alleen na echte netwerkcalls, cache. Parsing van API-JSON, het seizoensbestand (kwalificatie en sprint samenvoegen, ontbrekende waarden), `fetch_seasons` (overslaan, verversen, fouten per seizoen) en de racekalender (starttijden in UTC, ontbrekende tijd of kwalificatie, nog niet gepubliceerd seizoen, zonder cache). |
| `test_explore.py` | Het fase 4-rapport: datakwaliteit, domeincontrole, uitval, circuiteffecten. |
| `test_evaluation.py` | Metrics (met de hand gecontroleerd voorbeeld, perfecte en omgekeerde voorspelling, top 10 bij 20 coureurs, gelijke posities, ongeldige invoer), baselines A en B, de walk-forward-backtest (chronologisch, alleen eerdere races, dezelfde races per model, ongeldige invoer) en Ridge. |
| `test_features.py` | Elke featuregroep, `FeatureConfig`, de leakage-tests, de featuretabel, `f1 features` en de featurebacktest. |
| `test_models.py` | Het contract van elke `RacePredictor` (zie [Leakage bewaken](#leakage-bewaken)), de XGBoost-modellen (signaal volgen, determinisme, groepen per race, ontbrekende waarden, parameters) en featurebelang. |
| `test_model_selection.py` | `summarize_backtest` met gepaarde ΔMAE, `ModelSpec`, de tuninggrid en selectieregels, de controle dat `config/config.yaml` uit de vastgelegde tuningrun volgt, tuning alleen op validatieseizoenen, de modelbacktest, het eindmodel, `model_comparison.json` (alleen herschreven bij een inhoudelijke wijziging) en `f1 backtest`, `f1 tune` en `f1 train`. |
| `test_predict.py` | Deelnemerslijst uit de kwalificatie, de volgende race volgens de kalender (`next_race` en `race_to_predict` met een vaste klok: vóór, tijdens en na de kwalificatie, op de starttijd, na de laatste race), `predict_race` (retrospectief gelijk aan de backtestfold, negeert de eigen uitslag, live kwalificatie met nieuwe coureurs, te weinig historie), het JSON-contract, `latest.json`, de schrijfregels (alleen bij een inhoudelijke wijziging, een voorspelling vooraf blijft staan), het `run`-blok en `f1 predict` met `--season`/`--round` of `--next`. |
| `test_history.py` | `f1 evaluate`: met de hand gecontroleerde scores van een voorspelling vooraf tegen de uitslag (ook als beide coureurslijsten verschillen), alleen voorspellingen vooraf met een uitslag, de modelnaam per race, een lege historiek, niet herschrijven zonder wijziging en de tabel. |
| `test_workflows.py` | De workflows in `.github/workflows/`: elke `uv run f1 ...` noemt een bestaand commando met bestaande opties, en de twee automatische workflows zijn gepland, delen de concurrency-groep en committen alleen vanaf `main`. |
| `test_config.py` | Config inlezen, defaults en validatie, `--config`, de opties van `f1 fetch` en de hint naar `f1 fetch` bij ontbrekende data voor elk commando. |
| `test_pipeline.py` | De integratietest en het raceweekend hieronder. |

## Integratietest met de mini-dataset

[`tests/test_pipeline.py`](../tests/test_pipeline.py) voert de commando's uit zoals een nieuwe
gebruiker dat doet: `f1 fetch`, `features`, de drie backtestsuites, `train`, `predict` voor
een al gereden en een komende race, en `evaluate`. De commando's draaien via de echte CLI met
een tijdelijk `config.yaml` (seizoenen 2023–2024, Ridge en een kleine XGBoost-ranker,
`min_train_races: 1`). De pipeline loopt één keer per testrun; zeven tests controleren elk een
stap. Een tweede run speelt een raceweekend na zoals de workflows het doorlopen (zie
[Raceweekend](#raceweekend)).

**De data.** [`tests/fixtures/jolpica_mini.json`](../tests/fixtures/jolpica_mini.json) bevat
race-blokken per API-pad (`2023/results`, `2023/qualifying`, `2024/2/qualifying`, de
kalender `2024`, ...) in de structuur van de echte API. Coureurs, teams en uitslagen zijn verzonnen. Drie gereden races over
twee seizoenen en één race met alleen een kwalificatie:

| Race | Speciale gevallen |
|---|---|
| 2023-1 | Pitlane-start (`grid` 0 na P6 in de kwalificatie); statussen `+1 Lap` en `Engine` (oud vocabulaire); uitvaller met `R`. |
| 2023-2 | Sprintweekend; een coureur zonder kwalificatie. |
| 2024-1 | Zelfde circuit als 2023-1 (circuithistoriek over de seizoensgrens); statussen `Lapped` en `Retired` (nieuw vocabulaire). |
| 2024-2 | Alleen kwalificatie: de race is nog niet gereden. Een rookie zonder historie vervangt een coureur. De uitslag (`_uitslag_2024_2`, geen API-pad) komt pas in het raceweekend binnen; daarin vervangt een reserve zonder kwalificatie een coureur die wel kwalificeerde. |

**Wat echt is en wat niet.** Alleen `JolpicaClient.get` is vervangen door een client die
pagina's uit de fixture teruggeeft; paginering, parsing en alles daarna is de echte code. De
CLI-functies die bestanden lezen of schrijven krijgen via `functools.partial` de paden van de
tijdelijke map mee.

**Wat de test controleert.**

- `fetch`: alleen gereden races komen in de ruwe data; `grid` 0, beide statusvocabulaires, een
  ontbrekende kwalificatie en sprintpunten blijven zoals de API ze levert.
- `features`: alle featurekolommen uit de config; de startpositie komt uit de kwalificatie en
  niet uit de grid; circuithistoriek telt races over de seizoensgrens.
- `backtest`: elk model in elke suite is op exact dezelfde races gescoord;
  `model_comparison.json` is strikte JSON met het juiste eindmodel.
- `train`: model, metadata en featurebelang staan klaar, getraind op alle drie de races.
- `predict`: beide voorspellingen volgen het JSON-contract met rangen 1..n. De retrospectieve
  voorspelling is getraind tot en met de vorige race. De komende race haalt zijn deelnemers live
  uit de kwalificatie (inclusief de rookie), en `latest.json` wijst naar die race.
- `evaluate`: zonder voorspelling vooraf met een uitslag is de historiek leeg, maar het bestand
  bestaat.

**Vangnet.** Vóór en na de pipeline wordt een momentopname gemaakt van alle bestanden in de
`data/`- en `output/`-map van de repository (grootte en wijzigingstijd). Verschillen die, dan
faalt de test. Mist er een omleiding, bijvoorbeeld bij een nieuw commando, dan faalt de test
dus in plaats van stilletjes vastgelegde uitvoer te overschrijven. Voeg bij een nieuw commando
dat bestanden schrijft een omleiding toe in `_redirects` en het commando aan `COMMANDS`.

### Raceweekend

De fixture `weekend` doorloopt in een eigen tijdelijke map de stappen van de workflows rond
2024-2 (kwalificatie zaterdag 15:00 UTC, race zondag 14:00 UTC), met een vaste klok per stap:

1. `fetch`, daarna `predict --next` op vrijdag: niets te voorspellen.
2. `predict --next` tijdens de kwalificatie, terwijl Jolpica nog geen uitslag heeft: niets.
3. `predict --next` na de kwalificatie: `2024-02.json` met `result_known: false`. Een latere run
   die zaterdag herschrijft niets.
4. De uitslag komt binnen: `fetch --force` (2024 is voor `fetch` een afgesloten seizoen).
5. `predict --next` op zondagavond: de volgende race is ronde 3, waarvan de kwalificatie nog
   moet komen. Een retrospectieve `predict --season 2024 --round 2` laat de voorspelling vooraf
   staan.
6. De drie backtestsuites en `evaluate`: de historiek scoort de voorspelling vooraf tegen de
   uitslag, met de coureurs die in beide lijsten staan; de kwalificatiebaseline is met de hand
   gecontroleerd.

Ook hier faalt de test als er iets in de `data/`- of `output/`-map van de repository verandert.

**Niet afgedekt.** De echte API (netwerk), `f1 tune` (vaste validatieseizoenen 2015–2021 en een
grid van ongeveer 45 minuten; de selectieregels hebben eigen tests), modelkwaliteit (de data
is synthetisch; prestaties staan in [model.md](model.md)) en de workflows zelf op GitHub
(`test_workflows.py` controleert alleen hun inhoud; zie [automation.md](automation.md)).

## Leakage bewaken

Een voorspelling voor race N mag alleen eerdere races en de kwalificatie van N gebruiken. Dat
wordt op drie niveaus getest:

1. **Features** ([`test_features.py`](../tests/test_features.py), details in
   [features.md](features.md#leakage-tests)). Uitslag en definitieve grid van race N wijzigen
   verandert de features van race N niet; races na N weglaten verandert niets; de uitslag van een
   teamgenoot in race N telt niet mee. Deze tests draaien met alle groepen aan, dus nieuwe
   groepen vallen er automatisch onder.
2. **Voorspellers** ([`test_models.py`](../tests/test_models.py)). Elke `RacePredictor`, dus beide
   baselines en elke familie in `MODEL_FAMILIES`, moet aan hetzelfde contract voldoen. De
   voorspelling blijft identiek als de labelkolommen van de race ontbreken. De rangen zijn uniek
   1..n met de index van de race. De invoer blijft onaangeroerd. Een nieuwe modelfamilie valt
   er automatisch onder.
3. **Backtest en voorspelling.** De backtest traint alleen op strikt eerdere races
   ([`test_evaluation.py`](../tests/test_evaluation.py)). Een voorspelling van een al gereden race
   is gelijk aan de backtestfold en negeert de uitslag van die race
   ([`test_predict.py`](../tests/test_predict.py)). De integratietest controleert dat de
   retrospectieve voorspelling tot en met de vorige race is getraind.

Daarnaast controleert `test_configured_models_follow_from_the_selection_rules` dat de modelkeuze
in `config/config.yaml` uit de validatieraces volgt en niet uit de holdout.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) draait bij elke pull request en elke
push naar `main`:

1. `uv sync --locked` (de vastgelegde versies uit `uv.lock`);
2. `uv run ruff check .` en `uv run ruff format --check .`;
3. `uv run pytest --cov`: alle tests, en een fout als de coverage onder 90% zakt;
4. de coveragetabel in de samenvatting van de run op GitHub (ook als een eerdere stap faalt).

De geplande workflows `predict.yml` en `post-race.yml` draaien de tests niet; ze staan in
[automation.md](automation.md). Een PR die een van die workflows wijzigt, start die workflow
ook op de PR zelf (zonder commit, met de uitvoer als artefact).

CI draait op Ubuntu, de ontwikkeling gebeurt op Windows. De tests gebruiken daarom `pathlib`
en `tmp_path` in plaats van vaste paden. De XGBoost-modellen gebruiken geen `subsample`, zodat de
uitkomst niet van het aantal threads afhangt.

## Richtlijnen voor nieuwe tests

- **Offline.** Vervang Jolpica door een nepclient (`FakeClient`, `CalendarClient` en
  `ScriptedSession` in `test_data.py`, `QualifyingClient` in `test_predict.py`, `FixtureClient`
  in `test_pipeline.py`) en `time.sleep` door een functie die de wachttijd alleen onthoudt.
- **Deterministisch.** Vaste seeds (`np.random.default_rng(0)`, `random_state`); geen
  afhankelijkheid van de huidige datum, behalve waar dat bewust getest wordt. Code die de klok
  nodig heeft, neemt een `now`-argument (zoals `race_to_predict`); de test geeft een vast
  tijdstip mee.
- **Klein en synthetisch.** Een paar races met vier tot zes coureurs. Tests lezen nooit `data/`
  (die map staat niet in git en ontbreekt in CI). Vastgelegde uitvoer in `output/` mag gelezen
  worden, nooit geschreven.
- **Alleen in `tmp_path` schrijven.** Geef bibliotheekfuncties expliciete paden mee. Een
  CLI-test vervangt de aangeroepen functie of geeft haar tijdelijke paden mee (zie
  `test_pipeline.py`).
- **Nieuw gedrag, nieuwe test; bugfix, eerst een falende test.** Een nieuw model in
  `MODEL_FAMILIES` en een nieuwe featuregroep vallen automatisch onder het contract en de
  leakage-tests. Een nieuw commando dat schrijft hoort in de integratietest.
- Testnamen zijn Engels en beschrijven het verwachte gedrag; commentaar en meldingen zijn
  Nederlands, zoals in de rest van het project.
