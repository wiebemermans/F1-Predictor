# Stappenplan: F1-voorspelmodel in Python

Persoonlijk project om de uitkomst van F1-races te voorspellen. Vink de taken af (`- [x]`) terwijl je vordert.

**Drie principes**

1. **Meet voordat je optimaliseert:** baselines en backtest komen vóór het slimme model.
2. **Geen leakage:** elke feature en elke split houdt rekening met de tijdsvolgorde.
3. **Klein en werkend, dan uitbreiden:** na elke fase moet het project nog draaien.

**Referentie:** dit project is geïnspireerd op [F1-Prediction-Model](https://github.com/ArtuurHeidbuchel/F1-Prediction-Model) en de [bijbehorende demo](https://artuurheidbuchel.github.io/F1-prediction.html). De kern is een voorspelling na de kwalificatie, met alleen informatie uit eerdere races en de kwalificatie van de huidige race. Elke race wordt walk-forward geëvalueerd tegen de startvolgorde en de kampioenschapsstand. Dit project bouwt die aanpak zelfstandig verder op Jolpica-data; verschillen in databron en beschikbare velden worden expliciet bijgehouden.

---

## Overzicht en planning

| Fase | Inhoud | Tijd |
|---|---|---|
| 0-2 | Setup, repo, kwaliteitstooling | 1 weekend |
| 3 | Data ophalen | 1-2 weekends |
| 4 | Data verkennen | 1 weekend |
| 5 | Baselines + backtest | 1 weekend |
| 6 | Feature engineering | 2 weekends |
| 7 | Het model | 1-2 weekends |
| 8-9 | Config, CLI, tests | 1-2 weekends |
| 10 | Automatisering | 1 weekend |
| 11 | Opruimen en herorganiseren | 1 weekend |
| 12 | Uitbreidingen: kansen en Elo | 2-3 weekends |
| 13 | Website (draagbare F1-pagina) | 1-2 weekends |
| 14 | Afwerking en release | 1 weekend |
| – | Backlog (optionele ideeën) | – |

Reken op ongeveer 3-4 maanden als hobbyproject.

**Waarom deze volgorde:** de website komt bewust laat. Eerst wordt de repo opgeruimd (fase 11),
daarna komen er uitbreidingen die iets opleveren om te tonen (fase 12). Pas dan wordt de pagina
gebouwd, op een moment dat er genoeg inhoud is en de paden van de uitvoer vastliggen.

## Doelstructuur van de repo

Zo ziet de repo eruit na fase 11 (opruimen). De website uit fase 13 komt er later bij.

```
f1-predictor/
├── .github/workflows/
│   ├── ci.yml              # tests + linting bij elke push en pull request
│   ├── predict.yml         # elke 3 uur vr-zo: data ophalen + volgende race voorspellen
│   └── post-race.yml       # na de race: uitslag, backtests, voorspelling vs. realiteit
├── config/
│   └── config.yaml         # seizoenen, features, gekozen modellen, backtest
├── src/f1_predictor/
│   ├── data/               # ophalen, cachen en inladen van data, racekalender
│   ├── features/           # feature engineering
│   ├── models/             # baselines, modellen, training, voorspellen
│   ├── evaluation/         # metrics, backtests, modelselectie, historiek, verkenning
│   └── cli.py              # `f1 fetch`, `features`, `backtest`, `tune`, `train`, `predict`, `evaluate`
├── tests/
├── data/                   # in .gitignore (raw/processed)
├── output/
│   ├── predictions/        # JSON per race + latest.json (contract voor de website)
│   ├── backtests/          # CSV's van backtests en tuning
│   ├── exploration/        # CSV's van de dataverkenning
│   ├── model_comparison.json
│   └── prediction_history.json
├── docs/                   # documentatie per onderwerp
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Fase 0: Voorbereiding (±1 uur)

**Doel:** een werkende ontwikkelomgeving.

- [x] Python 3.13 of nieuwer installeren (`python --version`, vereiste volgens `pyproject.toml`)
- [x] Git installeren en een GitHub-account maken
- [x] Editor installeren (bv. VS Code + Python-extensie)
- [x] `uv` installeren (docs.astral.sh/uv), of klassiek `venv` + `pip`
- [x] Git instellen (eenmalig):
  ```bash
  git config --global user.name "Jouw Naam"
  git config --global user.email "jij@voorbeeld.com"
  ```
- [x] **Documentatie bijgewerkt:** vereiste Python-versie, uv-installatie en omgevingscontroles staan in [`docs/development.md`](docs/development.md).

**Klaar wanneer:** `python`, `git` en `uv` werken in je terminal.

---

## Fase 1: Repo en projectskelet (±1-2 uur)

**Doel:** een lege maar professionele basis op GitHub.

- [x] Repo maken op GitHub (bv. `f1-predictor`) met Python-`.gitignore` en MIT-licentie, daarna clonen
- [x] Project initialiseren:

  ```bash
  uv init --package f1-predictor
  cd f1-predictor
  uv add requests requests-cache pandas numpy pyarrow scikit-learn xgboost pyyaml typer
  uv add --dev pytest ruff pre-commit jupyter
  # Optioneel, pas bij een uitbreiding uit de backlog (weer, rondetijden, telemetrie):
  # uv add --optional enrich fastf1
  ```

- [x] Mappenstructuur aanmaken (zie "Doelstructuur") met een leeg `__init__.py` in elke package-map
- [x] `.gitignore` aanvullen: `data/`, `.venv/`, `*.parquet`, `*.sqlite` (requests-cache), `.ipynb_checkpoints/`
- [x] `.gitkeep` in `data/` en `output/`
- [x] Minimale README (titel, 2 zinnen doel, bronvermelding: data via Jolpica-F1, "work in progress")
- [x] Eerste commit en push, daarna kleine en beschrijvende commits ("add jolpica client")
- [x] Werken met feature-branches (`feat/fetch-data`) en merge via pull requests
- [x] **Documentatie bijgewerkt:** ontwikkelomgeving, actuele projectindeling en kwaliteitscommando's beschreven in [`docs/development.md`](docs/development.md) en de [README](README.md).

**Klaar wanneer:** `uv run python -c "import f1_predictor"` werkt en de repo staat op GitHub.

---

## Fase 2: Kwaliteitstooling (±1 uur)

**Doel:** automatische bewaking vanaf het begin.

- [x] Ruff configureren in `pyproject.toml` (lijnlengte, import-sortering); gebruik `uv run ruff check .` en `uv run ruff format .`
- [x] `.pre-commit-config.yaml` met ruff, daarna `uv run pre-commit install`
- [x] Eerste test in `tests/test_smoke.py`; `uv run pytest` slaagt
- [x] CI-workflow `.github/workflows/ci.yml`: `uv sync`, `ruff check` en `pytest` bij elke push en pull request
- [x] **Documentatie bijgewerkt:** test-, lint-, format- en pre-commit-instructies staan in [`docs/development.md`](docs/development.md); CI-gedrag is beschreven.

**Klaar wanneer:** je ziet een groen vinkje in GitHub Actions.

---

## Fase 3: Data ophalen (±1-2 weekends)

**Doel:** een lokale, gecachete dataset met meerdere seizoenen (Jolpica-F1).

- [x] Branch `feat/fetch-data`
- [x] API verkennen in `notebooks/00_api_exploration.ipynb` (structuur, strings vs. getallen, paginering)
- [x] Dataschema vastleggen in `docs/data_schema.md`
- [x] `JolpicaClient` met paginering, pauze, retry bij 429/5xx en `requests-cache`
- [x] Parse-functies voor results, qualifying en sprint (`parse.py`)
- [x] `build_season_table` + opslaan als parquet in `data/raw/` (atomisch schrijven)
- [x] CLI: `f1 fetch --start 2014 --end 2026` (bestaande seizoenen overslaan, lopend seizoen verversen)
- [x] Foutafhandeling per seizoen, exitcode 1 bij mislukkingen
- [x] Tests op synthetische data, geen live API (pytest + CI groen)
- [x] Gemerged naar `main`
- [x] **Documentatie bijgewerkt:** pipeline, API-cache, retries, seizoensbestanden en CLI-gebruik staan in [`docs/data_pipeline.md`](docs/data_pipeline.md); kolommen en databetekenis staan in [`docs/data_schema.md`](docs/data_schema.md).

**Aandachtspunten:** pitlane-starts (grid = 0), diskwalificaties/niet-starters, sprintpunten. Niets "oplossen" in de rauwe data, alleen documenteren.

**Klaar wanneer:** `uv run f1 fetch` levert parquet-bestanden voor 5+ seizoenen en de tests slagen.

---

## Fase 4: Data verkennen (±1 weekend)

**Doel:** de data begrijpen vóór je modelleert met `src/f1_predictor/evaluation/explore.py` en de gegenereerde analyse-uitvoer.

- [x] Ontbrekende waarden, duplicaten en domein-afwijkingen controleren; geldige uitzonderingen zoals `grid = 0` en `laps = 0` niet als fouten tellen
- [x] Spearman-correlatie tussen gridpositie en finishpositie over alle races
- [x] Uitvalpercentage per team berekenen en rapporteren
- [x] Uitvalpercentages ook per circuit en seizoen vergelijken
- [x] Circuit-effecten: zijn sommige coureurs structureel beter op bepaalde banen?
- [x] Teamsterkte per seizoen berekenen en rapporteren (gemiddelde finish per team per jaar)
- [x] Bevindingen noteren in [`docs/phase4_exploration.md`](docs/phase4_exploration.md)
- [x] **Documentatie bijgewerkt:** methode, datasetkwaliteit, domeincontrole en bevindingen staan in [`docs/phase4_exploration.md`](docs/phase4_exploration.md); de rapportgenerator bewaart de domeincontrole bij opnieuw genereren.

**Klaar wanneer:** je kunt in eigen woorden uitleggen wat de beste voorspeller lijkt en waar de ruis zit.

---

## Fase 5: Baselines en evaluatiekader (±1 weekend)

**Doel:** vastleggen hoe je succes meet vóór je een model bouwt. Dit is de belangrijkste fase.

**Voorspelmoment en datacontract**

- De voorspelling is voor de race, nadat de kwalificatie heeft plaatsgevonden en vóór de race start.
- Voor race N mogen features alleen eerdere races gebruiken, plus kwalificatiegegevens die vóór race N beschikbaar zijn. Resultaten, punten en uitvalstatus van race N zijn uitsluitend labels voor evaluatie.
- De huidige Jolpica-dataset bevat `quali_position`, maar geen kwalificatietijden. Gebruik daarom voor de eerste versie de kwalificatiepositie; voeg kwalificatietijden pas toe als aparte data-uitbreiding en documenteer die afwijking van de referentie.
- Leg vast welke startvolgorde de voorspelling gebruikt. Gebruik `quali_position` als voorspelling vóór verwerking van gridstraffen; gebruik `grid` alleen wanneer de voorspelling ná publicatie van de definitieve startgrid plaatsvindt. Meng die definities niet tussen baselines en model.
- Er is geen actuele weerdata of kennis van de betrouwbaarheid van de auto voor de race. Historische uitval mag alleen als feature worden gebruikt als die vóór de voorspelde race bekend was. Onverwacht weer, nieuwe defecten en gridstraffen die na het gekozen voorspelmoment bekend worden, worden niet voorspeld.

**Metrics** (`evaluation/metrics.py`), per race berekend:

- MAE van de voorspelde rang versus `finish_position` (lager is beter)
- Spearman-rangcorrelatie tussen voorspelde en echte volgorde (hoger is beter)
- Winnaar juist (ja/nee; aggregeren als percentage races)
- Podiumoverlap (0-3 coureurs; eventueel ook als percentage met exact podium)
- Top-10-overlap (0-10 coureurs)

Alle modellen en baselines moeten exact dezelfde races en dezelfde evaluatieregels gebruiken. Leg een vaste, reproduceerbare tie-break vast voor gelijke voorspelscores en ontbrekende/ongeldige klasseringen; controleer de metrics handmatig op een klein voorbeeld.

**Baselines** (`models/baselines.py`), met dezelfde interface als het latere model: `predict(race_df) -> voorspelde rang`.

- Baseline A: voorspelde rang is de startvolgorde volgens het gekozen voorspelmoment (kwalificatiepositie of definitieve `grid`)
- Baseline B: kampioenschapsstand vóór de race, berekend uitsluitend met punten uit eerdere races (inclusief sprintpunten waar van toepassing); leg tie-break en puntenbron vast
- Optioneel Baseline C: gemiddelde finish van de laatste 5 races

**Walk-forward backtest** (`evaluation/backtest.py`):

```python
results = run_backtest(
    df,
    {
        "qualifying_position": QualifyingPositionBaseline,
        "championship_points": ChampionshipPointsBaseline,
    },
    min_train_races=20,
)
```

De racevolgorde wordt uit `season` en `round` bepaald. De eerste twintig races zijn trainingshistorie; elke daaropvolgende race wordt apart getest met uitsluitend eerdere races als training. Metrics, tie-breaks en outputformaat staan in [`docs/phase5_baselines.md`](docs/phase5_baselines.md).

- [x] Voorspelmoment, gebruikte startvolgorde en uitsluitingen (weer, betrouwbaarheid, late gridstraffen) vastgelegd
- [x] Metrics geïmplementeerd en met de hand gecontroleerd op een klein voorbeeld
- [x] Baselines A en B geïmplementeerd; kampioenschapspunten zijn strikt van vóór de race
- [x] Walk-forward backtest geïmplementeerd: trainen op alle eerdere races, testen op de volgende race; geen willekeurige splitsing
- [x] Resultaten per model, seizoen en race opgeslagen; baseline-uitvoer in `output/backtest_baselines.csv`
- [x] **Documentatie bijgewerkt:** metricdefinities, baseline-regels, tie-breaks, backtestprotocol, output en actuele resultaten staan in [`docs/phase5_baselines.md`](docs/phase5_baselines.md).

**Klaar wanneer:** je reproduceerbare scores per baseline hebt over dezelfde races, inclusief gekozen voorspelmoment en evaluatieregels. Dit zijn de getallen waarmee je latere model wordt vergeleken; de startgrid en kampioenschapsstand zijn betekenisvolle referenties, geen garantie die het model moet verslaan.

---

## Fase 6: Feature engineering (±2 weekends)

**Doel:** informatie omzetten in voorspellende variabelen, zonder leakage.

**Gouden regel:** een feature voor race N gebruikt alleen data uit races vóór N (plus de kwalificatie van N zelf). In pandas: `.shift(1)` vóór rolling/expanding berekeningen.

```python
df = df.sort_values(["driver", "race_index"])
df["driver_form_5"] = df.groupby("driver")["finish_position"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).mean()
)
```

**Featuregroepen** (voeg ze één voor één toe en meet telkens de backtest):

- [x] Startpositie: `grid` of `quali_position` volgens het gekozen voorspelmoment; een kwalificatiegat tot polesitter vereist kwalificatietijden en valt buiten de eerste versie zolang die niet worden opgehaald
- [x] Coureursvorm: gemiddelde finish en punten over laatste 3 en 5 races, uitvalpercentage
- [x] Teamvorm: gemiddelde team-finish en punten over recente races (vangt autosterkte op)
- [x] Kampioenschapspositie vóór de race
- [x] Circuit-historiek van de coureur (met shrinkage naar het gemiddelde bij weinig data)
- [x] Teamgenoot-vergelijking
- [ ] Later: weer, bandenstrategie, rondetijden uit trainingen (FP2-longruns); zie de backlog

**Praktisch**

- [x] Elke featuregroep in een eigen functie in `features/`, aan/uit te zetten via de config (`FeatureConfig`, ingelezen uit `config/config.yaml`)
- [x] Leakage-test in pytest: wijzig de uitslag van race N kunstmatig en controleer dat de features van race N niet veranderen
- [x] Experimentenlogboek bijhouden (feature, backtestscore, conclusie); scores in `output/backtest_features.csv`
- [x] CLI-commando `f1 features` dat een tabel bouwt in `data/processed/`
- [x] **Documentatie bijgewerkt:** betekenis, databron, beschikbaarheid op het voorspelmoment en behandeling van ontbrekende waarden per feature, het bouwen van de featuretabel, de leakage-tests en het experimentenlogboek staan in [`docs/features.md`](docs/features.md).

**Klaar wanneer:** `f1 features` werkt en de leakage-test slaagt.

---

## Fase 7: Het model (±1-2 weekends)

**Doel:** een eerste ML-model dat de baselines probeert te verslaan.

- [x] Eerst simpel: `LinearRegression` of `Ridge` (als dat de baseline al verslaat, werken je features)
- [x] Daarna gradient boosting:
  - Regressie (`XGBRegressor`): voorspel de finishpositie en sorteer
  - Learning-to-rank (`XGBRanker`, `rank:pairwise`): groepeer per race, label = `n_coureurs - finishpositie`
- [x] Model achter dezelfde `fit`/`predict`-interface als de baselines
- [x] Conservatieve hyperparameters (`max_depth` 3-4, lage `learning_rate`, weinig `n_estimators`); tune met tijdsgebaseerde validatie, niet met een willekeurige split (getuned op 2015–2021, gerapporteerd op de holdout 2022–2026; de beste bomen bleken nog ondieper: diepte 2)
- [x] Model opslaan (`joblib` of `save_model`) in `output/models/` (niet in git; opnieuw te bouwen met `train_final_model`)
- [x] Interpretatie via feature importance of SHAP (gain en TreeSHAP uit XGBoost zelf)
- [x] **Documentatie bijgewerkt:** modelvarianten, labels, interface, hyperparameters, tuningprotocol, reproduceerbaarheid, vergelijking met de baselines, interpretatie en beperkingen staan in [`docs/model.md`](docs/model.md).

**Klaar wanneer:** het model draait in de backtest en je hebt een eerlijke vergelijking met de baselines. Verslaat het ze nog niet? Dat is normaal; ga terug naar fase 6.

---

## Fase 8: Configuratie en CLI afronden (±1 weekend)

**Doel:** het project bruikbaar maken zonder code aan te passen.

Oorspronkelijke schets van `config/config.yaml` (de uiteindelijke velden staan in
[`docs/usage.md`](docs/usage.md)):

```yaml
data:
  start_season: 2018
  end_season: 2025
features:
  form_windows: [3, 5]
  use_circuit_history: true
model:
  type: xgb_ranker
  params: {max_depth: 3, learning_rate: 0.05, n_estimators: 200}
backtest:
  min_train_races: 20
```

- [x] Config laden in één module (`dataclasses`, geen extra dependency): `config.py` met validatie die onbekende sleutels weigert; de modelkeuze uit fase 7 staat nu in `config.yaml`
- [x] Typer-commando's: `f1 fetch`, `f1 features`, `f1 backtest`, `f1 predict --season 2026 --round 10`, plus `f1 tune` en `f1 train` (vervangen de `python -c`-aanroepen); `f1 predict` traint het eindmodel telkens opnieuw op alle races vóór de gevraagde race
- [x] Outputs als JSON in `output/` (rechtstreeks inleesbaar door de website): `output/predictions/{seizoen}-{ronde}.json` met `latest.json`, en `output/model_comparison.json`
  - Behandel het JSON-formaat als een vast contract: `schema_version`, `generated_at`, modelnaam, race-info (seizoen, ronde, naam), voorspelde volgorde en de backtestsamenvatting (`summarize_backtest`)
  - Documenteer het formaat (in [`docs/usage.md`](docs/usage.md#json-formaat-contract-voor-een-website)), zodat elke latere site (GitHub Pages of een eigen persoonlijke website) alleen van dit formaat afhangt en niet van de Python-code
- [x] `logging` in plaats van `print`
- [x] Vaste random seeds (`random_state=0`, geen `subsample`); sla config en commit-hash mee op bij elke run (`run`-blok in elke JSON, git-info in de modelmetadata)
- [x] **Documentatie bijgewerkt:** configuratievelden met defaults en toegestane waarden, CLI-commando's met voorbeelden, outputbestanden, het JSON-formaat en veelvoorkomende fouten staan in [`docs/usage.md`](docs/usage.md); README, `CLAUDE.md` en de fasedocumenten gebruiken de CLI.

**Klaar wanneer:** een vreemde kan je repo clonen en met vier commando's tot een voorspelling komen.

---

## Fase 9: Tests uitbreiden (doorlopend)

- [x] Unit tests voor metrics, feature-functies en baselines (aangevuld met de retries van de Jolpica-client, ongeldige invoer en de ontbrekende CLI-paden)
- [x] Leakage-test (zie fase 6), nu op drie niveaus: features (ook `grid` van race N), een contracttest voor elke `RacePredictor` (zelfde voorspelling zonder labelkolommen) en backtest/voorspelling
- [x] Integratietest met een mini-dataset (3 gereden races en 1 komende race, `tests/fixtures/jolpica_mini.json`) door de hele CLI-pipeline, met een vangnet dat faalt als de repo-uitvoer wijzigt
- [x] Coverage meten met `pytest-cov` (zinnig percentage, geen 100%-obsessie): branch coverage, ondergrens 90%, bij afronding 96,8%
- [x] CI draait alle tests bij elke pull request, nu met coverage en een coveragetabel in de samenvatting van de run
- [x] **Documentatie bijgewerkt:** tests en coverage lokaal uitvoeren, overzicht per testbestand, wat de integratietest afdekt, leakage- en CI-controles en richtlijnen voor nieuwe tests staan in [`docs/testing.md`](docs/testing.md).

---

## Fase 10: Automatisering met GitHub Actions (±1 weekend)

**Doel:** het project draait zichzelf.

- [x] `predict.yml` met `schedule` (cron) en `workflow_dispatch`: data ophalen, volgende race voorspellen, naar `output/` schrijven en committen (`f1 fetch` + `f1 predict --next`; de bot commit alleen `output/` naar `main`, en alleen bij een inhoudelijke wijziging)
- [x] Slim plannen: kwalificatie is meestal zaterdag, dus cron op zaterdagavond en zondagochtend; bepaal de "volgende race" via de F1-kalender (de kwalificatie valt tussen zaterdag 04:00 en 21:00 UTC en soms op vrijdag, dus draait de cron elke 3 uur van vrijdag tot en met zondag; de Jolpica-kalender beslist of de kwalificatie al begonnen en verwerkt is)
- [x] Tweede workflow na de race (maandag): echte uitslag ophalen, backtest bijwerken, voorspelling vs. realiteit vergelijken (`post-race.yml` op maandag, dinsdag als herkansing: de drie backtestsuites en `f1 evaluate` naar `output/prediction_history.json`; een voorspelling van vooraf wordt nooit door een retrospectieve vervangen)
- [x] Dependencies en FastF1-cache cachen in Actions (uv-cache via `setup-uv`; FastF1 wordt niet gebruikt, dus in de plaats daarvan de ruwe Jolpica-data `data/raw/`, per jaar in de Actions-cache)
- [x] Onthoud: geplande workflows kunnen vertraagd of overgeslagen worden en worden na 60 dagen inactiviteit uitgeschakeld (minstens zes pogingen per kwalificatie-tot-race-venster; na de winterstop weer aanzetten met `gh workflow enable`)
- [x] Houd de berekening hier, ook als er later een eigen (thuis)server is: die server haalt dan alleen de nieuwste JSON op (`git pull` of raw-URL) en hoeft geen Python/ML te draaien. Alternatief: dezelfde CLI via cron op de server (`uv run f1 fetch && uv run f1 predict --next`)
- [x] **Documentatie bijgewerkt:** triggers, planning, rechten (geen secrets, alleen `contents: write`), gegenereerde bestanden, handmatig opnieuw uitvoeren en problemen oplossen staan in [`docs/automation.md`](docs/automation.md); `--next`, `f1 evaluate` en het JSON-contract van `prediction_history.json` in [`docs/usage.md`](docs/usage.md), de nieuwe tests in [`docs/testing.md`](docs/testing.md).

---

## Fase 11: Opruimen en herorganiseren (±1 weekend)

**Doel:** een opgeruimde repo met een logische indeling, vóór er nieuwe functionaliteit bijkomt.
Na deze fase werkt alles nog zoals voorheen; alleen de structuur en documentatie veranderen.

**Timing:** de bot commit tijdens raceweekenden naar `output/`. Herindelen van `output/` gebeurt
pas na de laatste race van 2026 (6 december), en altijd na een `git pull`.

**Repo-hygiëne**

- [ ] Branches die al in `main` gemerged zijn verwijderen (lokaal en op `origin`); op GitHub "Automatically delete head branches" aanzetten
- [ ] `notebooks/` verwijderen (de bevindingen staan in [`docs/data_schema.md`](docs/data_schema.md)) en de dev-dependency `jupyter` schrappen
- [ ] `.gitignore` inkorten tot de regels die voor dit project gelden (nu ~200 regels GitHub-template)
- [ ] `tests/test_smoke.py` verwijderen (elke andere test importeert het pakket al)

**Codestructuur**

- [ ] `load_all_results` van `evaluation/explore.py` naar `data/` verplaatsen: `features/` en `models/` mogen niet afhangen van het verkenningsrapport
- [ ] Kalenderlogica (`next_race`, `race_to_predict`) uit `models/predict.py` in een eigen module zetten, met de bijbehorende tests apart
- [ ] Eenmalig controleren op ongebruikte code en dependencies (bv. `uvx vulture`, `uvx deptry`); geen nieuwe dependency toevoegen

**Map `output/`**

- [ ] CSV's groeperen: `output/exploration/` (verkenning, nu `phase4_*.csv`) en `output/backtests/` (backtests, tuning, modelvergelijking, feature importance)
- [ ] De JSON voor de website blijft bovenaan: `predictions/`, `model_comparison.json`, `prediction_history.json`
- [ ] `paths.py`, workflows, tests (`test_workflows.py`, de repo-snapshotcontrole in `test_pipeline.py`) en docs aanpassen; oude bestanden verwijderen
- [ ] Na de verplaatsing beide automatiseringsworkflows één keer handmatig draaien (`workflow_dispatch`) en controleren dat ze niets onverwachts committen

**Documentatie**

- [ ] Docs per onderwerp in plaats van per fase: bv. `phase4_exploration.md` → `exploration.md`, `phase5_baselines.md` → `baselines.md`; README-links zonder fasenummers
- [ ] Eén plek voor de actuele resultaten (model vs. baselines); andere docs linken ernaar in plaats van getallen te herhalen
- [ ] Dubbele uitleg wegwerken (bv. projectindeling in `CLAUDE.md` en `docs/development.md`)
- [ ] `.github/agent-instructions.md` en `CLAUDE.md` op elkaar afstemmen (o.a. de Copilot-co-auteursregel)
- [ ] Release/tag `v0.1.0`: mijlpaal "de pipeline werkt van data tot automatische voorspelling"
- [ ] **Documentatie bijgewerkt:** nieuwe paden, bestandsnamen en indeling staan in README, `CLAUDE.md` en de docs; geen dode links.

**Klaar wanneer:** tests, coverage en Ruff slagen, beide workflows draaien met de nieuwe paden, en
de README leidt een nieuwe lezer zonder omwegen naar de juiste docs.

---

## Fase 12: Uitbreidingen: kansen en Elo (±2-3 weekends)

**Doel:** het project rijker maken met resultaten die het tonen waard zijn, vóór de website
wordt gebouwd. Elke uitbreiding volgt dezelfde regels als fase 5-7: geen leakage, keuzes op de
validatieseizoenen (≤ 2021), de holdout (≥ 2022) alleen om te rapporteren.

**Kansen in plaats van één volgorde (Monte-Carlo)**

- [ ] Per race een kansverdeling simuleren: kans op winst, podium en top 10 per coureur (bv. de modelscores verstoren met ruis die gekalibreerd is op backtestfouten, plus uitvalkans uit historische uitval vóór de race)
- [ ] Vaste seed en vast aantal simulaties, zodat de uitvoer reproduceerbaar is
- [ ] Kalibratie meten in de walk-forward backtest (bv. Brier-score en een betrouwbaarheidstabel: komt "30% kans" ook ongeveer 30% van de keren uit?)
- [ ] De kansen toevoegen aan de voorspellings-JSON als nieuwe velden (toevoegen vereist geen nieuwe `schema_version`; contract bijwerken in [`docs/usage.md`](docs/usage.md))

**Elo-rating voor coureurs en teams**

- [ ] Elo-ratings die na elke race worden bijgewerkt (paarsgewijs binnen het veld), met strikt de rating van vóór de race als feature
- [ ] K-factor en terugval naar het gemiddelde tussen seizoenen kiezen op de validatieseizoenen
- [ ] Als featuregroep (aan/uit via `FeatureConfig`, gedekt door de leakage-tests) én als losse `RacePredictor` die de contracttest haalt
- [ ] Meten met `run_backtest` en de feature-backtest; conclusie in het experimentenlogboek, ook als het niets oplevert

**Historiek van voorspellingen**

- [ ] Controleren of `output/prediction_history.json` na de laatste races van 2026 genoeg bevat om te tonen (aantal races, gemiddelde scores vs. de kwalificatiebaseline); zo nodig velden toevoegen die de website nodig heeft

- [ ] **Documentatie bijgewerkt:** methode, kalibratie en beperkingen van de kansen; definitie, parameters en resultaten van Elo; nieuwe JSON-velden in [`docs/usage.md`](docs/usage.md) en de nieuwe tests in [`docs/testing.md`](docs/testing.md).

**Klaar wanneer:** de voorspelling van de volgende race bevat gekalibreerde kansen, Elo is eerlijk
vergeleken met het bestaande model, en de automatisering levert beide zonder handwerk.

---

## Fase 13: Website: draagbare F1-pagina (±1-2 weekends)

**Doel:** een statische pagina die de F1-predictor uitlegt en de resultaten toont, los van waar
ze uiteindelijk draait.

**Waar de pagina kan draaien:** de pagina is een zelfstandig onderdeel dat onder een subpad werkt.
Mogelijke plaatsen, zonder dat de pagina zelf verandert:

- GitHub Pages van deze repo: `<gebruikersnaam>.github.io/F1-Predictor/`. Een latere persoonlijke
  GitHub Pages-site (`<gebruikersnaam>.github.io`) kan vanaf de hoofdpagina gewoon naar dit
  subpad linken, en krijgt bij een eigen domein automatisch hetzelfde subpad.
- Een eigen website op een thuisserver, bv. `voornaamachternaam.be/projects/f1`: de map kopiëren
  of de JSON periodiek ophalen; de server hoeft geen Python te draaien.

**Bouwen**

- [ ] Statische pagina (HTML, CSS, een beetje JavaScript) in een eigen map, bv. `site/`, zonder build-stap of framework
- [ ] Draagbaar: alleen relatieve paden, de data-URL in één variabele, niets dat alleen op GitHub Pages werkt; de pagina werkt onder elk subpad
- [ ] Leest alleen de JSON uit `output/` (contract in [`docs/usage.md`](docs/usage.md)), nooit Python-code of CSV's
- [ ] Tonen: voorspelling van de volgende race (met kansen uit fase 12), model vs. baselines uit de backtest, historiek van voorspellingen vs. uitslag, Elo-ranglijst
- [ ] Uitleg in gewone taal: wat het model doet, het voorspelmoment, hoe de backtest werkt, en wat het model niet weet
- [ ] Toont wanneer de data het laatst is bijgewerkt (`generated_at`) en meldt duidelijk als er nog geen voorspelling is
- [ ] Grafieken alleen als ze iets toevoegen (bv. Chart.js); geen Streamlit of andere server-app

**Publiceren**

- [ ] Kiezen hoe de JSON bij de pagina komt (bv. een workflow die `site/` en de JSON samen naar Pages deployt, of ophalen via een raw-URL) en die keuze vastleggen
- [ ] Publiceren via GitHub Pages van deze repo als eerste, werkende versie
- [ ] Lokaal testen beschrijven (bv. `uv run python -m http.server` vanuit de juiste map)
- [ ] **Documentatie bijgewerkt:** welke JSON de site inleest, lokaal testen, publiceren via Pages, overzetten naar een andere host, en de betekenis en actualiteit van wat getoond wordt (bv. in `docs/website.md`).

**Klaar wanneer:** de pagina staat online, werkt ook vanuit een andere map of host zonder
aanpassingen, en wordt na elke race automatisch actueel.

---

## Fase 14: Afwerking en release (±1 weekend)

- [ ] README: wat het doet, link en screenshot van de site, actuele backtestresultaten, installatie en gebruik, projectstructuur, methodologie, beperkingen
- [ ] Badges (CI-status, Python-versie, licentie)
- [ ] Beschrijving, website-link en topics op GitHub (`formula1`, `machine-learning`, `xgboost`)
- [ ] Sectie "ideeën voor uitbreiding" (verwijst naar de backlog) of `CONTRIBUTING.md`
- [ ] Eerlijk vermelden wat het model niet weet (weer, uitvallers, strategie, gridstraffen na kwalificatie)
- [ ] Release/tag `v1.0.0`
- [ ] **Documentatie afronden:** controleer README en docs op kloppende installatie- en gebruiksinstructies, actuele backtestresultaten, methodologie, databronnen, beperkingen en links. Verwijder verouderde of dubbele instructies.

---

## Backlog (optioneel, geen vaste volgorde)

Ideeën voor later. Wat hiervan wordt opgepakt, krijgt dan een eigen fase met dezelfde regels.

- [ ] Kwalificatie voorspellen met trainingsdata (FP1-FP3)
- [ ] Weerdata (bv. Open-Meteo, gratis) en regenkans als feature
- [ ] Bandenstrategie en rondetijden uit trainingen (FP2-longruns), bv. via FastF1
- [ ] Modelvergelijking: LightGBM, CatBoost
- [ ] Docker voor een reproduceerbare omgeving
- [ ] **Documentatie bijwerken (bij elke opgepakte uitbreiding):** databronnen, configuratie, gebruik, validatie en nieuwe beperkingen; README en relevante docs bijwerken.

---

## Valkuilen om te onthouden

- **Data leakage:** gebruik nooit informatie die je vóór de race niet had; splits altijd op tijd, nooit willekeurig.
- **Weinig data:** ~24 races per seizoen, dus houd het model eenvoudig en let op overfitting.
- **Regelwijzigingen:** reglementswissels (bv. 2022, 2026) maken oude seizoenen minder relevant; overweeg recente seizoenen zwaarder te wegen.
- **Ruis:** safety cars, regen, uitvallers en gridstraffen zijn onvoorspelbaar. Een model dat de startgrid maar licht verslaat is al een goed resultaat.
