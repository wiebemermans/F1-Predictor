# Fase 7: Het model

Deze pagina beschrijft de modellen die de raceuitslag voorspellen na de kwalificatie en vóór de
race: welke varianten er zijn, hoe hyperparameters en het eindmodel gekozen zijn, hoe ze scoren
tegen de baselines en hoe je de interpretatie leest. De features staan in
[features.md](features.md), de backtest- en metricregels in
[phase5_baselines.md](phase5_baselines.md).

## Samenvatting

Run van 2026-11-24 op data van 2014 tot en met ronde 21 van 2026: 273 races, waarvan 253 testraces.

- Vier modellen zijn getuned op de testraces van 2015–2021 en daarna gescoord op een holdout van
  2022–2026 (113 races): Ridge, XGBoost-regressie, een XGBoost-ranker met `rank:pairwise` en een
  XGBoost-ranker met `rank:ndcg`.
- **Eindmodel.** Het eindmodel is het gekozen model met de beste gemiddelde top-3-voorspelling
  (podiumoverlap) op de validatieraces: de pairwise ranker. Ridge scoort daar even hoog, maar de
  ranker voorspelt vaker de juiste winnaar.
- **Top 3.** Op de holdout voorspelt het eindmodel gemiddeld 2.018 van de 3 podiumplaatsen juist,
  tegen 1.929 voor de kwalificatievolgorde (+0.088, SE 0.043).
- **Volgorde.** Ridge, de regressie en de pairwise ranker verslaan de kwalificatiebaseline op
  MAE, Spearman, podium- en top-10-overlap. Het eindmodel verbetert de MAE met 0.082 posities
  (1.6 standaardfouten); Ridge doet het met 0.128 (2.7 standaardfouten) het best.
- **Winnaar.** Geen enkel model voorspelt de winnaar vaker juist dan de regel "de polesitter
  wint" (59.3% op de holdout). Ridge en de ndcg-ranker komen op één race na (58.4%). De
  pairwise ranker haalt 54.0%.
- **ndcg.** De ndcg-ranker haalt zijn winnaarsscore door vaker gewoon de polesitter te kiezen.
  Daarbij verliest hij de winst op de rest van de volgorde (MAE −0.007, gelijk aan de baseline).
- **Eenvoud wint.** De beste boommodellen zijn klein (diepte 1–2), en de modellen liggen zo dicht
  bij elkaar dat de keuze ertussen grotendeels ruis is.

## Modelvarianten

| Model | Klasse | Label | Volgorde | Ontbrekende waarden |
|---|---|---|---|---|
| Ridge | `RidgeFeaturePredictor` ([linear.py](../src/f1_predictor/models/linear.py)) | `finish_position` | lagere voorspelde finish = betere rang | mediaan van de trainingsfold, daarna standaardisatie |
| XGBoost-regressie | `XGBRegressorPredictor` ([boosting.py](../src/f1_predictor/models/boosting.py)) | `finish_position` (`reg:squarederror`) | lagere voorspelde finish = betere rang | niet aangevuld; XGBoost leert per split de richting voor NaN |
| XGBoost-ranker (pairwise) | `XGBRankerPredictor` ([boosting.py](../src/f1_predictor/models/boosting.py)) | per race: aantal deelnemers − `finish_position` (`rank:pairwise`) | hogere score = betere rang | idem |
| XGBoost-ranker (ndcg) | `XGBNDCGRankerPredictor` ([boosting.py](../src/f1_predictor/models/boosting.py)) | idem (`rank:ndcg`) | hogere score = betere rang | idem |

- **Interface.** Alle modellen volgen het `RacePredictor`-protocol: `fit(train)` op alleen eerdere
  races (dat bewaakt `run_backtest`) en `predict(race)` geeft per rij een unieke rang 1..n met
  dezelfde index. `predict` leest alleen de featurekolommen; een test controleert dat de
  uitslagkolommen van de race geen invloed hebben.
- **Tie-break.** Gelijke voorspellingen worden op `driver_id` alfabetisch gesorteerd, zoals bij de
  baselines.
- **Rankers.** Elke race is één groep (`qid`). Het label geeft de winnaar het hoogste getal en de
  laatste 0, dus het model leert alleen de volgorde binnen een race. XGBoost vergelijkt alle paren
  binnen een race (standaard `lambdarank_pair_method = "topk"` met een onbeperkt aantal paren).
  De trainingsrijen worden in `fit` per race en daarbinnen op `driver_id` gesorteerd, zodat de
  invoervolgorde het model niet beïnvloedt.
- **Pairwise of ndcg.** `rank:pairwise` weegt elk verkeerd geordend paar even zwaar, ook achteraan
  het veld. `rank:ndcg` weegt paren naar hun effect op NDCG met een exponentiële gain
  (2^label − 1): de winnaar telt ongeveer twee keer zo zwaar als P2, en de achterhoede
  nauwelijks. Die variant is toegevoegd om het winnaarsprobleem uit fase 6 te onderzoeken.
- **Labels.** De officiële klassering wordt ook voor uitvallers gebruikt, net als in de metrics.
- **Featuresets.** `all` gebruikt alle 17 features. `upto_championship` gebruikt de 13 features
  uit de groepen die in fase 6 de Ridge-backtest verbeterden (`starting_position`, `driver_form`,
  `team_form`, `championship`).

Een model wordt beschreven door een `ModelSpec` (familie, featureset en parameters) in
[train.py](../src/f1_predictor/models/train.py). Die levert in elke fold een verse voorspeller.

## Hyperparameters

Vaste instellingen voor alle XGBoost-modellen (`DEFAULT_XGB_PARAMS`):

| Parameter | Waarde | Waarom |
|---|---|---|
| `learning_rate` | 0.05 | lage stapgrootte tegen overfitting |
| `colsample_bytree` | 0.8 | elke boom ziet 80% van de features |
| `tree_method` | `hist` | standaard in XGBoost, snel |
| `random_state` | 0 | reproduceerbaar |

`max_depth` en `n_estimators` worden getuned. Bewust is er geen `subsample` (rijsampling). Gemeten
op de featuretabel geeft `subsample = 0.8` tot 2.9 posities verschil in voorspelde finish tussen
runs met 1, 3 en 20 threads. Dezelfde code zou dus op een andere machine (of in GitHub Actions)
andere voorspellingen geven. Kolomsampling en volledige rijen geven bij elk aantal threads exact
dezelfde uitkomst; dat is voor alle drie de XGBoost-doelen nagemeten.

## Tuning- en evaluatieprotocol

Alle scores komen uit dezelfde walk-forward-backtest als in fase 5 en 6: elke testrace wordt
voorspeld door een model dat op alle eerdere races is getraind, met `min_train_races = 20`.

- **Validatie:** de testraces van 2015 (ronde 2) tot en met 2021, 140 races. Hierop worden
  hyperparameters, featureset en eindmodel gekozen (`run_model_tuning`). De backtest krijgt
  alleen de seizoenen tot en met 2021. Dat geeft exact dezelfde folds als op alle data, omdat
  features alleen eerdere races gebruiken; dat is ook als controle nagerekend.
- **Holdout:** de testraces van 2022 tot en met ronde 21 van 2026, 113 races, beginnend bij de
  reglementswissel van 2022. De holdout is pas gescoord nadat de keuze vastlag.
- **Waarom.** Wie de beste van 80 configuraties kiest op dezelfde races waarop hij rapporteert,
  krijgt een te rooskleurige score. De validatiescores van de gekozen modellen zijn dus
  optimistisch; de holdout is het eerlijke getal.

**Rooster** (`tuning_candidates` in
[model_selection.py](../src/f1_predictor/evaluation/model_selection.py)), elk met beide
featuresets:

| Familie | Parameters | Configuraties |
|---|---|---:|
| Ridge | `alpha` ∈ {0.1, 1, 10, 100} | 8 |
| XGBoost-regressie | `max_depth` ∈ {1, 2, 3, 4} × `n_estimators` ∈ {50, 100, 300} | 24 |
| XGBoost-ranker (pairwise) | idem | 24 |
| XGBoost-ranker (ndcg) | idem | 24 |

**Hoe het rooster zo is geworden**

1. Het oorspronkelijke rooster had `max_depth` ∈ {2, 3, 4} en `n_estimators` ∈ {100, 300}. In die
   eerste run waren voor de regressie en de pairwise ranker de kleinste instellingen (diepte 2,
   100 bomen) het best, op de rand van het rooster. Daarom zijn diepte 1 en 50 bomen toegevoegd
   en is het hele rooster opnieuw gedraaid; dat gebruikte alleen validatieraces.
2. In die tweede run ligt de beste pairwise ranker weer op een rand (50 bomen). Het rooster is
   toch niet verder uitgebreid. De buren (diepte 2 met 100 bomen, diepte 1 met 300 bomen) zijn
   slechts ongeveer 0.02 posities slechter, veel minder dan de standaardfout. Nog meer zoeken op
   dezelfde 140 races zou vooral ruis fitten.
3. De ndcg-ranker is daarna als vierde familie op hetzelfde rooster toegevoegd, en het volledige
   rooster is opnieuw gedraaid. De 56 bestaande configuraties gaven exact dezelfde scores.

**Selectieregels**, allebei alleen op de validatieraces:

1. **Hyperparameters** (`select_tuned_models`): per familie de configuratie met de laagste
   gemiddelde MAE. Bij gelijkspel de eenvoudigste: minder features, dan ondiepere bomen, minder
   bomen en sterkere regularisatie. Deze regel lag vast vóór de holdout werd bekeken, ook voor de
   ndcg-ranker. Kiezen op het aandeel juiste winnaars over 140 races zou vooral ruis selecteren:
   één race is 0.7 procentpunt. MAE kijkt naar het hele veld en is stabieler.
2. **Eindmodel** (`select_final_model`): van de vier gekozen configuraties die met de hoogste
   gemiddelde podiumoverlap (de beste top-3-voorspelling). Bij gelijkstand beslist het hoogste
   aandeel juiste winnaars, daarna de laagste MAE.

**Over regel 2.** Die was eerst "de familie met de laagste validatie-MAE". Op 2026-11-24 is hij
vervangen door de podiumregel, omdat de top 3 het belangrijkste doel van de voorspelling is.

- De wijziging kwam nadat de holdout bekeken was, maar de regel gebruikt alleen validatieraces.
- De oude en de nieuwe regel geven hetzelfde eindmodel.
- Ook de podiumoverlap is over 140 races ruizig: tussen de vier modellen scheelt het hoogstens 8
  podiumplaatsen, en Ridge en de pairwise ranker eindigen exact gelijk.

## Tuningresultaat

De volledige samenvatting per configuratie staat in `output/model_tuning.csv`.

| Familie | Gekozen | Validatie-MAE | ΔMAE (SE) t.o.v. kwalificatie | Podiumoverlap | Winnaar juist |
|---|---|---:|---:|---:|---:|
| Baseline kwalificatiepositie | – | 3.621 | – | 2.100 | 53.6% |
| Ridge | `ridge_a10_upto_championship` | 3.574 | −0.048 (0.036) | **2.086** | 49.3% |
| XGBoost-regressie | `xgb_regressor_d2_n100_all` | 3.562 | −0.059 (0.043) | 2.029 | 50.7% |
| XGBoost-ranker (pairwise) | `xgb_ranker_d2_n50_all` | 3.533 | −0.088 (0.037) | **2.086** | **53.6%** |
| XGBoost-ranker (ndcg) | `xgb_ranker_ndcg_d1_n300_all` | 3.577 | −0.045 (0.046) | 2.071 | 52.9% |

Eindmodel: **`xgb_ranker_d2_n50_all`**. Ridge en de pairwise ranker hebben dezelfde podiumoverlap
(292 van de 420 podiumplaatsen juist). De ranker voorspelt vaker de juiste winnaar (75 tegen 69
races) en wint daarmee de gelijkstand.

Op de validatieraces is geen enkel model beter in de top 3 dan de kwalificatievolgorde zelf (2.100).
Op de holdout zijn ze dat alle vier wel (zie hieronder).

De keuze staat in de sectie `model` van [config.yaml](../config/config.yaml): `tuned` per
familie en `final` voor het eindmodel. De test `test_configured_models_follow_from_the_selection_rules`
past de regels toe op `output/model_tuning.csv` en controleert dat de configuratie eruit volgt.

**Patronen in de tuning**

- **Meer capaciteit is slechter**, bij de regressie en de pairwise ranker. Vanaf diepte 2 wordt de
  validatie-MAE vrijwel steeds slechter met diepere bomen en meer bomen; diepte 4 met 300 bomen
  is slechter dan de kwalificatiebaseline. Stumps (diepte 1) zijn iets minder goed dan diepte 2.
  Met zo'n 5000 rijen en een luidruchtig label overfitten bomen snel, en er zit weinig
  niet-lineair signaal bovenop wat Ridge al vangt.
- **De ndcg-ranker leert trager.** Zijn MAE verbetert bij elke diepte met meer bomen, en de
  beste configuratie heeft 300 bomen, de rand van het rooster. Met meer bomen of een hogere
  `learning_rate` kan de MAE nog iets dalen; dat is niet verder onderzocht.
- **Winnaar per familie.** Gemiddeld over alle 24 configuraties heeft de ndcg-ranker het hoogste
  aandeel juiste winnaars op de validatieraces (54.0%, tegen 53.1% voor de pairwise ranker,
  50.7% voor Ridge en 47.0% voor de regressie). Dat is nauwelijks meer dan de 53.6% van de
  kwalificatiebaseline, minder dan één race op 140.
- **Featuresets.** Ridge doet het beter met de 13 features van `upto_championship`; de
  boommodellen kiezen `all`, maar het verschil is klein (0.002–0.02).
- **Ridge-regularisatie.** `alpha = 10` is het best en ligt binnen het rooster.

## Vergelijking met de baselines

Alle modellen en baselines komen uit één `run_model_backtest`-aanroep, dus op exact dezelfde
races; de baselinescores zijn identiek aan fase 5. ΔMAE is het gemiddelde verschil per race met
de kwalificatiebaseline (gepaard), met de standaardfout tussen haakjes; negatief is beter.
"Beter/slechter" is het aandeel races met een lagere of hogere MAE dan de kwalificatiebaseline.

**Holdout 2022–2026 (113 races): de eerlijke vergelijking**

| Model | MAE | Spearman | Winnaar juist | Podiumoverlap | Top-10-overlap | ΔMAE (SE) | Beter/slechter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline kwalificatiepositie | 3.388 | 0.655 | 59.3% | 1.929 | 7.788 | – | – |
| Baseline kampioenschapspunten | 3.971 | 0.573 | 44.2% | 1.637 | 7.451 | +0.583 (0.105) | 27% / 69% |
| Ridge | 3.260 | 0.680 | 58.4% | 1.991 | 7.920 | −0.128 (0.047) | 50% / 37% |
| XGBoost-regressie | 3.300 | 0.674 | 52.2% | 2.035 | 7.885 | −0.088 (0.050) | 50% / 44% |
| XGBoost-ranker pairwise (eindmodel) | 3.306 | 0.672 | 54.0% | 2.018 | 7.876 | −0.082 (0.051) | 50% / 41% |
| XGBoost-ranker ndcg | 3.381 | 0.660 | 58.4% | 2.035 | 7.841 | −0.007 (0.060) | 49% / 47% |

Gepaard ten opzichte van Ridge op de holdout:

| Model | ΔMAE (SE) t.o.v. Ridge |
|---|---:|
| XGBoost-regressie | +0.041 (0.026) |
| pairwise ranker | +0.046 (0.028) |
| ndcg-ranker | +0.122 (0.033) |

**Alle 253 testraces (2015–2026), vergelijkbaar met fase 5 en 6**

| Model | MAE | Spearman | Winnaar juist | Podiumoverlap | Top-10-overlap | ΔMAE (SE) | Beter/slechter |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline kwalificatiepositie | 3.517 | 0.637 | 56.1% | 2.024 | 7.648 | – | – |
| Baseline kampioenschapspunten | 4.006 | 0.561 | 41.1% | 1.763 | 7.265 | +0.489 (0.067) | 31% / 66% |
| Ridge | 3.433 | 0.656 | 53.4% | 2.043 | 7.798 | −0.084 (0.029) | 50% / 39% |
| XGBoost-regressie | 3.445 | 0.655 | 51.4% | 2.032 | 7.783 | −0.072 (0.032) | 49% / 45% |
| XGBoost-ranker pairwise (eindmodel) | 3.432 | 0.656 | 53.8% | 2.055 | 7.775 | −0.085 (0.031) | 50% / 42% |
| XGBoost-ranker ndcg | 3.490 | 0.645 | 55.3% | 2.055 | 7.711 | −0.028 (0.037) | 51% / 44% |

De validatieperiode afzonderlijk staat in `output/model_comparison.csv` (`period = validation`).

## Winnaar versus polesitter

De voorspelde winnaar van de kwalificatiebaseline is de polesitter. Op de holdout won de
polesitter 67 van de 113 races (59.3%). Per model, uit dezelfde walk-forward-folds:

| Model | Winnaar juist | Kiest de polesitter | Kiest iemand anders: model juist / polesitter juist | McNemar p |
|---|---:|---:|---:|---:|
| Ridge | 58.4% | 61% (69 races) | 16 / 17 | 1.00 |
| XGBoost-regressie | 52.2% | 65% (74 races) | 11 / 19 | 0.20 |
| XGBoost-ranker pairwise | 54.0% | 75% (85 races) | 6 / 12 | 0.24 |
| XGBoost-ranker ndcg | 58.4% | 81% (91 races) | 7 / 8 | 1.00 |

- **Waar het misgaat.** Het verlies zit volledig in de races waarin een model de polesitter
  overrulet. Typische gevallen zijn Verstappen kiezen boven een andere polesitter in 2022–2024
  (op basis van vorm) en Norris of Piastri boven een Verstappen-pole in 2025.
- **Wat ndcg doet.** De ndcg-ranker volgt de polesitter vaker (81% tegen 75%) en overrulet
  zuiniger. Dat brengt het winnaarspercentage terug naar het niveau van de baseline, maar niet
  erboven.
- **Significantie.** De McNemar-test vergelijkt alleen de races waarin model en polesitter van
  elkaar verschillen. Geen enkel verschil is significant: er is geen bewijs dat de modellen
  slechter zijn in de winnaar, maar ook geen dat ze beter zijn.

## Nieuwe races als onafhankelijke test

De races van 2026 ronde 18–21 (Verenigde Staten, Mexico-Stad, Brazilië, Las Vegas) zijn na de
modelkeuze opgehaald en zaten dus nergens in de selectie.

| Ronde | Polesitter | Winnaar | Pairwise ranker | ndcg-ranker | Ridge |
|---|---|---|---|---|---|
| 18 Verenigde Staten | antonelli | antonelli | antonelli | antonelli | antonelli |
| 19 Mexico-Stad | hamilton | max_verstappen | hamilton | hamilton | max_verstappen |
| 20 Brazilië | norris | norris | antonelli | antonelli | antonelli |
| 21 Las Vegas | antonelli | max_verstappen | antonelli | antonelli | max_verstappen |

- **MAE.** Gemiddeld over deze vier races: kwalificatiebaseline 3.73; Ridge, de regressie en de
  pairwise ranker alle drie 3.23; ndcg-ranker 3.34.
- **Winnaar.** De polesitter won 2 van de 4 races; Ridge koos er 3 juist, beide rankers 1.
- **Lezen.** Vier races zijn een anekdote, geen bewijs.

## Conclusies

1. **De modellen verslaan de startvolgorde op de volgorde, maar bescheiden.** Op de holdout is de
   MAE van Ridge, de regressie en de pairwise ranker lager dan die van de kwalificatiebaseline,
   en Spearman, podium- en top-10-overlap zijn hoger. Per race is een model in ongeveer de helft
   van de races beter en in 37–44% slechter. Alleen het verschil van Ridge is groter dan twee
   standaardfouten.
2. **De winnaar blijft voor geen enkel model beter dan "de polesitter wint".** De ndcg-ranker
   maakt het verschil met de pairwise ranker goed op winnaars, maar alleen door de polesitter
   vaker te volgen, en verliest daarbij de winst op de rest van het veld. Het doel `rank:ndcg`
   is daarom geen verbetering als eindmodel.
3. **Het eindmodel is niet het beste op de holdout.**
   - **Top 3:** het voorspelt gemiddeld 2.018 podiumplaatsen juist. Dat is beter dan de
     kwalificatievolgorde (1.929), maar iets minder dan de regressie en de ndcg-ranker (2.035,
     twee podiumplaatsen meer over 113 races).
   - **Volgorde:** Ridge is 0.046 posities beter in MAE (SE 0.028), terwijl de ranker op de
     validatieraces 0.040 beter was.
   - **Waarom de keuze blijft:** zulke omkeringen passen bij modellen die vrijwel gelijkwaardig
     zijn, waartussen de selectie vooral ruis oppikt. Het eindmodel nu op basis van de holdout
     wisselen zou de holdout zijn onafhankelijkheid ontnemen.
4. **Wat dit betekent.** Volgens het stappenplan is een model dat de startgrid licht verslaat al
   een goed resultaat. Wie later toch een ander model kiest op basis van de holdout, neemt
   een beslissing die door de holdout is beïnvloed. Voorbeelden zijn de regressie of de
   ndcg-ranker voor de top 3, of Ridge voor de hele volgorde. Vanaf dan zijn alleen nieuwe races
   nog een onafhankelijke test.

Dit zijn historische backtestscores, geen garantie voor toekomstige races.

## Interpretatie van het eindmodel

`train_final_model` schrijft het featurebelang van het eindmodel naar
`output/model_feature_importance.csv`, berekend op alle trainingsrijen (5567 rijen, 2014 tot en met
2026 ronde 21).

| Feature | Aandeel gain | Gem. \|SHAP\| |
|---|---:|---:|
| `start_position` | 58.0% | 0.163 |
| `team_points_mean_5` | 19.0% | 0.059 |
| `driver_points_mean_5` | 9.4% | 0.027 |
| `champ_position_before` | 4.4% | 0.016 |
| `team_finish_mean_5` | 2.1% | 0.010 |
| overige zeven gebruikte features | 7.0% | ≤ 0.009 per feature |
| `driver_finish_mean_3`, `driver_retirement_rate_10`, `start_quali_missing`, `circuit_prior_races`, `teammate_beat_rate_10` | 0% | 0 |

**Hoe lees je dit**

- **Aandeel gain** is het deel van de totale verbetering van de trainingsdoelfunctie dat uit
  splits op deze feature komt. **Gem. |SHAP|** is de gemiddelde absolute bijdrage van de feature
  aan de modeluitvoer per rij (exacte TreeSHAP uit XGBoost zelf, zonder extra dependency).
- Bij de ranker staat SHAP op de schaal van de rangscore: alleen de verhoudingen tussen features
  zijn betekenisvol, het zijn geen posities. Bij de XGBoost-regressie zou SHAP wel in
  finishposities staan. Ridge rapporteert in plaats daarvan gestandaardiseerde coëfficiënten.
- Beide maten zijn **in-sample** en **niet causaal**. Ze beschrijven waar dit model op leunt,
  niet wat een uitslag veroorzaakt.
- Gecorreleerde features delen hun belang. Team- en coureurspunten, of punten- en
  finishgemiddelden, kunnen elkaar vervangen. Een feature met 0% wordt door dit kleine model (50
  bomen van diepte 2) niet gekozen, maar is daarmee niet bewezen nutteloos.

**Wat het zegt.** Het model is vooral de startvolgorde, gecorrigeerd voor recente team- en
coureursvorm in punten. Dat sluit aan bij fase 6: `driver_form`, `team_form` en `championship`
voegden iets toe. `circuit_history` en `teammate` worden nauwelijks gebruikt
(`circuit_delta_shrunk` en `teammate_quali_gap` elk 0.2%).

## Eindmodel en opslag

```powershell
uv run f1 train
```

`f1 train` traint het eindmodel (`model.final` in `config/config.yaml`) op alle races met een
uitslag en schrijft:

- `output/models/xgb_ranker_d2_n50_all.joblib`: het getrainde voorspellerobject (met
  featurekolommen), te laden met `joblib.load` en direct bruikbaar met `predict(race)`. De
  invoer moet met `build_features` en dezelfde `FeatureConfig` gebouwd zijn.
- `output/models/xgb_ranker_d2_n50_all.json`: metadata (klasse, parameters, featurekolommen,
  `FeatureConfig`, eerste en laatste trainingsrace, aantal rijen, versies van Python, XGBoost,
  scikit-learn en pandas, git-commit met `dirty`-vlag, aanmaakdatum).
- `output/model_feature_importance.csv`: de interpretatie hierboven.

`output/models/` staat in `.gitignore`: het model is binair en met één commando opnieuw te bouwen.
Joblib-bestanden zijn pickles. Laad alleen bestanden die je zelf hebt gemaakt, met dezelfde
XGBoost- en scikit-learn-versies als in de metadata.

Voor een voorspelling is dit opgeslagen model niet nodig: `f1 predict` traint het eindmodel
telkens opnieuw op alle races vóór de gevraagde race (enkele seconden). Zo is een voorspelling
identiek aan die race in de backtest, ook voor een race die al gereden is. Zie
[usage.md](usage.md).

## Reproduceren

```powershell
uv run f1 fetch      # ververst het lopende seizoen
uv run f1 tune       # ~45 min
uv run f1 backtest   # ~1 min
uv run f1 train      # seconden
```

| Bestand | Inhoud |
|---|---|
| `output/model_tuning.csv` | samenvatting per configuratie op de validatieraces, met familie, featureset en parameters |
| `output/backtest_models.csv` | scores per race voor de baselines en de vier gekozen modellen, zelfde kolommen als de fase 5/6-CSV's |
| `output/model_comparison.csv` | samenvatting per periode (`validation`, `holdout`, `all`) |
| `output/model_comparison.json` | dezelfde samenvatting als JSON voor een website ([usage.md](usage.md)) |
| `output/model_feature_importance.csv` | featurebelang van het eindmodel |

De uitkomsten zijn deterministisch: twee runs van `run_model_backtest` gaven byte-identieke CSV's.
In fase 8 gaf `f1 backtest` (met de keuze uit `config.yaml` in plaats van constanten in de code)
opnieuw byte-identieke CSV's voor alle drie de suites.
Controles die bij de run van 2026-11-24 zijn uitgevoerd:

- de baselinerijen in `backtest_models.csv` zijn identiek aan `backtest_baselines.csv`;
- de validatiescores van de gekozen modellen zijn identiek aan hun rij in `model_tuning.csv`;
- na het ophalen van ronde 18–21 van 2026 zijn ronde 1–17 en alle validatiescores ongewijzigd;
- het opgeslagen model geeft na herladen dezelfde voorspellingen.

De winnaarsanalyse hierboven komt uit een losse analyse op dezelfde folds; die is niet als
functie in het pakket opgenomen.

## Beperkingen

- **De holdout is niet volledig onaangeroerd.**
  - De featuregroepen zijn in fase 6 op alle races beoordeeld, inclusief 2022 en later. De
    featureset `upto_championship` (die Ridge gebruikt) komt uit die analyse, wat Ridge op de
    holdout iets kan bevoordelen.
  - De ndcg-ranker is toegevoegd nadat het winnaarsprobleem op alle races (fase 6) en op de
    holdout was gezien.
- **Er is meerdere keren gezocht.** Het rooster is na de eerste run uitgebreid, en de
  ndcg-familie kwam er later bij. Dat gebeurde alleen op validatieraces, maar meer zoeken maakt de
  validatiescores van de gekozen modellen optimistischer.
- **Kleine verschillen.** De verschillen tussen modellen (0.04–0.05 posities, de ndcg-ranker
  uitgezonderd) zijn van dezelfde orde als hun standaardfout, en per race wisselt de winst sterk.
- **Training en hyperparameters.**
  - Elk model traint op alle races sinds 2014, zonder zwaarder gewicht voor recente seizoenen en
    dwars door de reglementswissels van 2022 en 2026.
  - Er is geen early stopping; het aantal bomen ligt vast uit de tuning.
  - De ndcg-ranker had binnen het rooster mogelijk nog niet genoeg bomen.
- **Interpretatie.** De interpretatie is in-sample en beschrijft één model, getraind op alle
  data.
- **Features.** Alle beperkingen van de features gelden ook hier (zie
  [features.md](features.md#beperkingen)): geen kwalificatietijden, weer, strategie of late
  gridstraffen, en geen koppeling van team-ID's bij naamswijzigingen.

## Volgende stappen

- **Een aparte winnaarsvoorspelling**, als de winnaar belangrijk blijft. Bijvoorbeeld een
  classifier op "wint deze race" met kansen per coureur, of een regel die de polesitter op P1
  houdt tenzij het model met grote marge iemand anders kiest. Dat moet net zo op de validatieraces
  worden afgesteld.
- **Recente races zwaarder wegen** (sample weights met een halfwaardetijd), vanwege
  reglementswissels.
- **Terug naar fase 6 voor sterkere features:** kwalificatietijden (gat tot pole) en team-ID's
  koppelen bij naamswijzigingen.
- **Fase 10 (gedaan):** de volgende race wordt automatisch bepaald en na elke kwalificatie
  voorspeld; na de race zet `f1 evaluate` de voorspelling naast de uitslag
  ([automation.md](automation.md)). De cijfers in dit document zijn een momentopname bij het
  afronden van fase 7; `output/` loopt sindsdien met elke race mee.
