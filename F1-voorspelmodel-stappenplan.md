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
| 10-11 | Automatisering + website | 2 weekends |
| 12 | Afwerking | 1 weekend |

Reken op ongeveer 2-3 maanden als hobbyproject.

## Doelstructuur van de repo

```
f1-predictor/
├── .github/workflows/
│   ├── ci.yml              # tests + linting bij elke push
│   └── predict.yml         # geplande run vóór elke race
├── config/
│   └── config.yaml         # seizoenen, features, hyperparameters
├── src/f1_predictor/
│   ├── data/               # ophalen + cachen van data
│   ├── features/           # feature engineering
│   ├── models/             # training, predict, baselines
│   ├── evaluation/         # backtest + metrics
│   └── cli.py              # `f1 fetch`, `f1 train`, `f1 predict`
├── tests/
├── notebooks/              # exploratie, geen productiecode
├── data/                   # in .gitignore (raw/processed)
├── output/                 # voorspellingen als JSON/CSV
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
  # Optioneel, pas in fase 13 (weer, rondetijden, telemetrie):
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
- [ ] Uitvalpercentages ook per circuit en seizoen vergelijken
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

- [ ] Startpositie: `grid` of `quali_position` volgens het gekozen voorspelmoment; een kwalificatiegat tot polesitter vereist kwalificatietijden en valt buiten de eerste versie zolang die niet worden opgehaald
- [ ] Coureursvorm: gemiddelde finish en punten over laatste 3 en 5 races, uitvalpercentage
- [ ] Teamvorm: gemiddelde team-finish en punten over recente races (vangt autosterkte op)
- [ ] Kampioenschapspositie vóór de race
- [ ] Circuit-historiek van de coureur (met shrinkage naar het gemiddelde bij weinig data)
- [ ] Teamgenoot-vergelijking
- [ ] Later: weer, bandenstrategie, rondetijden uit trainingen (FP2-longruns)

**Praktisch**

- [ ] Elke featuregroep in een eigen functie in `features/`, aan/uit te zetten via de config
- [ ] Leakage-test in pytest: wijzig de uitslag van race N kunstmatig en controleer dat de features van race N niet veranderen
- [ ] Experimentenlogboek bijhouden (feature, backtestscore, conclusie)
- [ ] CLI-commando `f1 features` dat een tabel bouwt in `data/processed/`
- [ ] **Documentatie bijwerken:** leg per feature de betekenis, databron, beschikbaarheid op het voorspelmoment en behandeling van ontbrekende waarden vast (bijvoorbeeld in `docs/features.md`). Beschrijf ook hoe je features bouwt en hoe de leakage-regel is getest.

**Klaar wanneer:** `f1 features` werkt en de leakage-test slaagt.

---

## Fase 7: Het model (±1-2 weekends)

**Doel:** een eerste ML-model dat de baselines probeert te verslaan.

- [ ] Eerst simpel: `LinearRegression` of `Ridge` (als dat de baseline al verslaat, werken je features)
- [ ] Daarna gradient boosting:
  - Regressie (`XGBRegressor`): voorspel de finishpositie en sorteer
  - Learning-to-rank (`XGBRanker`, `rank:pairwise`): groepeer per race, label = `n_coureurs - finishpositie`
- [ ] Model achter dezelfde `fit`/`predict`-interface als de baselines
- [ ] Conservatieve hyperparameters (`max_depth` 3-4, lage `learning_rate`, weinig `n_estimators`); tune met tijdsgebaseerde validatie, niet met een willekeurige split
- [ ] Model opslaan (`joblib` of `save_model`) in `output/models/`
- [ ] Interpretatie via feature importance of SHAP
- [ ] **Documentatie bijwerken:** beschrijf de gekozen modelvariant, labels, trainings- en voorspelinterface, hyperparameters, reproduceerbaarheid en vergelijking met de baselines (bijvoorbeeld in `docs/model.md`). Noteer ook beperkingen en hoe interpretaties gelezen moeten worden.

**Klaar wanneer:** het model draait in de backtest en je hebt een eerlijke vergelijking met de baselines. Verslaat het ze nog niet? Dat is normaal; ga terug naar fase 6.

---

## Fase 8: Configuratie en CLI afronden (±1 weekend)

**Doel:** het project bruikbaar maken zonder code aan te passen.

`config/config.yaml`:

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

- [ ] Config laden in één module (`dataclasses` of `pydantic`)
- [ ] Typer-commando's: `f1 fetch`, `f1 features`, `f1 backtest`, `f1 predict --season 2026 --round 10`
- [ ] Outputs als JSON in `output/` (rechtstreeks inleesbaar door de website)
- [ ] `logging` in plaats van `print`
- [ ] Vaste random seeds; sla config en commit-hash mee op bij elke run
- [ ] **Documentatie bijwerken:** documenteer configuratievelden met defaults en toegestane waarden, CLI-commando's met voorbeelden, outputbestanden en veelvoorkomende fouten (README of `docs/usage.md`).

**Klaar wanneer:** een vreemde kan je repo clonen en met vier commando's tot een voorspelling komen.

---

## Fase 9: Tests uitbreiden (doorlopend)

- [ ] Unit tests voor metrics, feature-functies en baselines
- [ ] Leakage-test (zie fase 6)
- [ ] Integratietest met een mini-dataset (2-3 races) door de hele pipeline
- [ ] Coverage meten met `pytest-cov` (zinnig percentage, geen 100%-obsessie)
- [ ] CI draait alle tests bij elke pull request
- [ ] **Documentatie bijwerken:** beschrijf hoe je tests en coverage lokaal uitvoert, wat de integratietest afdekt en hoe leakage- en CI-controles werken (bijvoorbeeld in `docs/testing.md`).

---

## Fase 10: Automatisering met GitHub Actions (±1 weekend)

**Doel:** het project draait zichzelf.

- [ ] `predict.yml` met `schedule` (cron) en `workflow_dispatch`: data ophalen, volgende race voorspellen, naar `output/` schrijven en committen
- [ ] Slim plannen: kwalificatie is meestal zaterdag, dus cron op zaterdagavond en zondagochtend; bepaal de "volgende race" via de F1-kalender
- [ ] Tweede workflow na de race (maandag): echte uitslag ophalen, backtest bijwerken, voorspelling vs. realiteit vergelijken
- [ ] Dependencies en FastF1-cache cachen in Actions
- [ ] Onthoud: geplande workflows kunnen vertraagd of overgeslagen worden en worden na 60 dagen inactiviteit uitgeschakeld
- [ ] **Documentatie bijwerken:** leg triggers, planning, benodigde secrets/rechten, gegenereerde artefacten en handmatig opnieuw uitvoeren/troubleshooten vast (bijvoorbeeld in `docs/automation.md`).

---

## Fase 11: Website via GitHub Pages (±1-2 weekends)

**Doel:** resultaten zichtbaar maken.

- [ ] Eén statische `index.html` die JSON uit `output/` inlaadt via `fetch`
- [ ] Tonen: voorspelde uitslag volgende race, backtesttabel (model vs. baselines), korte uitleg + beperkingen
- [ ] Publiceren via Settings → Pages
- [ ] Optioneel: grafieken (Chart.js/Plotly) of een Streamlit-app (heeft hosting nodig)
- [ ] **Documentatie bijwerken:** beschrijf het dataformaat dat de site inleest, lokaal testen en publiceren via Pages, plus de betekenis en actualiteit van getoonde voorspellingen (bijvoorbeeld in `docs/website.md`).

---

## Fase 12: Afwerking (±1 weekend)

- [ ] README: wat het doet, link/screenshot van de site, backtestresultaten, installatie en gebruik, projectstructuur, methodologie, beperkingen
- [ ] Badges (CI-status, Python-versie, licentie)
- [ ] Beschrijving en topics op GitHub (`formula1`, `machine-learning`, `xgboost`)
- [ ] Sectie "ideeën voor uitbreiding" of `CONTRIBUTING.md`
- [ ] Release/tag `v0.1.0`
- [ ] Eerlijk vermelden wat het model niet weet (weer, uitvallers, strategie, gridstraffen na kwalificatie)
- [ ] **Documentatie afronden:** controleer README en docs op kloppende installatie- en gebruiksinstructies, actuele backtestresultaten, methodologie, databronnen, beperkingen en links. Verwijder verouderde of dubbele instructies.

---

## Fase 13: Uitbreidingen (optioneel)

- [ ] Probabilistische voorspellingen (bv. "35% kans op podium") via Monte-Carlo-simulatie
- [ ] Kwalificatie voorspellen met trainingsdata (FP1-FP3)
- [ ] Weerdata (bv. Open-Meteo, gratis) en regenkans als feature
- [ ] Modelvergelijking: LightGBM, CatBoost, Elo-systeem voor coureurs en teams
- [ ] Docker voor een reproduceerbare omgeving
- [ ] Dashboard met historiek van eigen voorspellingen vs. de realiteit
- [ ] **Documentatie bijwerken (als deze fase wordt uitgevoerd):** documenteer per toegevoegde uitbreiding de databronnen, configuratie, gebruik, validatie en nieuwe beperkingen; werk README en relevante docs bij.

---

## Valkuilen om te onthouden

- **Data leakage:** gebruik nooit informatie die je vóór de race niet had; splits altijd op tijd, nooit willekeurig.
- **Weinig data:** ~24 races per seizoen, dus houd het model eenvoudig en let op overfitting.
- **Regelwijzigingen:** reglementswissels (bv. 2022, 2026) maken oude seizoenen minder relevant; overweeg recente seizoenen zwaarder te wegen.
- **Ruis:** safety cars, regen, uitvallers en gridstraffen zijn onvoorspelbaar. Een model dat de startgrid maar licht verslaat is al een goed resultaat.
