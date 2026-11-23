# Fase 6: Features

Deze pagina beschrijft de features voor de voorspelling na de kwalificatie en vóór de race:
wat elke feature betekent, hoe de featuretabel wordt gebouwd, hoe leakage is getest en wat
elke featuregroep in de walk-forward-backtest oplevert.

## Gouden regel

Een feature voor race N gebruikt alleen races vóór N, plus wat na de kwalificatie van race N
bekend is: `quali_position`, `constructor_id`, `circuit_id` en de deelnemerslijst. In de code
betekent dat per reeks eerst `shift(1)` en pas daarna een rolling- of expanding-berekening.
Teamfeatures worden eerst per (team, race) samengevat en pas daarna verschoven; een `shift(1)`
op coureursrijen zou anders het resultaat van de teamgenoot in dezelfde race meenemen.

`grid` wordt bewust niet gebruikt: die kan gridstraffen bevatten die pas na het voorspelmoment
bekend zijn (zie [phase5_baselines.md](phase5_baselines.md)). De uitslag, punten, status,
rondes en sprintpunten van race N zijn uitsluitend labels.

## Featuretabel bouwen

```powershell
uv run f1 fetch --start 2014 --end 2026   # ruwe data, als die er nog niet is
uv run f1 features                        # -> data/processed/features.parquet
```

`f1 features` laadt alle `data/raw/results_*.parquet`, bouwt de features met de standaardconfig
en schrijft de tabel atomisch weg. Zonder ruwe data stopt het commando met exitcode 1. De tabel
heeft één rij per coureur per race: alle ruwe kolommen ongewijzigd, `race_index` (chronologisch
racenummer 0..n-1 op `season` en `round`) en de featurekolommen. Labels zoals
`finish_position` blijven in de tabel staan om modellen te trainen; een model leest alleen de
kolommen uit `feature_columns(config)`.

In code:

```python
from f1_predictor.features.build import FeatureConfig, build_features, feature_columns

config = FeatureConfig(groups=("starting_position", "driver_form"))
table = build_features(raw_df, config)
features = table[feature_columns(config)]
```

## Configuratie

`FeatureConfig` in [`build.py`](../src/f1_predictor/features/build.py) bepaalt welke groepen
aanstaan. In fase 8 wordt `config/config.yaml` in deze dataclass ingelezen.

| Veld | Standaard | Betekenis |
|---|---|---|
| `groups` | alle zes groepen | ingeschakelde featuregroepen; de kolomvolgorde ligt vast |
| `form_windows` | `(3, 5)` | vensters (aantal races) voor coureurs- en teamvorm |
| `rate_window` | `10` | venster voor uitvalpercentage en teamgenoot-vergelijking |
| `circuit_shrinkage` | `3.0` | k in de shrinkage van de circuitfeature; hoger trekt sterker naar 0 |

Onbekende of dubbele groepen, lege, dubbele of niet-positieve vensters en een negatieve of
oneindige shrinkage geven een `ValueError`.

## Features per groep

Kolomnamen met `{w}` bestaan voor elk venster in `form_windows`; `{r}` is `rate_window`.
"Vorige races" zijn de eigen vorige races van de coureur of het team, over seizoenen heen.
Elke groep is één functie in [`groups.py`](../src/f1_predictor/features/groups.py).

### `starting_position`

| Kolom | Betekenis | Bron | Beschikbaar | Ontbrekend |
|---|---|---|---|---|
| `start_position` | kwalificatiepositie van race N | `quali_position` | na de kwalificatie | geen kwalificatiepositie: aantal deelnemers van race N (laatste plaats) |
| `start_quali_missing` | 1 als de kwalificatiepositie ontbrak, anders 0 | `quali_position` | na de kwalificatie | nooit leeg |

### `driver_form`

| Kolom | Betekenis | Bron | Beschikbaar | Ontbrekend |
|---|---|---|---|---|
| `driver_finish_mean_{w}` | gemiddelde finishpositie in de vorige w races | `finish_position` | vóór race N | leeg bij de eerste race van een coureur |
| `driver_points_mean_{w}` | gemiddelde racepunten (zonder sprint) in de vorige w races | `points` | vóór race N | idem |
| `driver_retirement_rate_{r}` | aandeel uitval (DNF of DNS) in de vorige r races | `status` | vóór race N | idem |

Uitval volgt dezelfde definitie als fase 4: alles behalve gefinisht (`Finished`, `+N Lap(s)` in
oudere seizoenen, `Lapped` in recentere) en gediskwalificeerd. De finishpositie is de officiële
klassering, dus ook voor uitvallers.

### `team_form`

| Kolom | Betekenis | Bron | Beschikbaar | Ontbrekend |
|---|---|---|---|---|
| `team_finish_mean_{w}` | gemiddelde van de gemiddelde teamfinish per race, over de vorige w teamraces | `finish_position`, `constructor_id` | vóór race N | leeg bij de eerste race van een `constructor_id` |
| `team_points_mean_{w}` | gemiddelde racepunten van het team (alle coureurs samen) per race, over de vorige w teamraces | `points`, `constructor_id` | vóór race N | idem |

### `championship`

| Kolom | Betekenis | Bron | Beschikbaar | Ontbrekend |
|---|---|---|---|---|
| `champ_points_before` | race- plus sprintpunten uit eerdere ronden van hetzelfde seizoen | `points`, `sprint_points`, `season` | vóór race N | 0 in ronde 1 en voor nieuwe coureurs |
| `champ_position_before` | rang van die punten onder de deelnemers van race N; meeste punten = 1, gelijke punten delen de beste rang | idem | vóór race N | in ronde 1 heeft iedereen rang 1 |

De sprint van race N telt niet mee, ook al wordt die vóór de race verreden: `sprint_points` is
in het [dataschema](data_schema.md) een uitkomst. Dit is gelijk aan
`ChampionshipPointsBaseline`.

### `circuit_history`

| Kolom | Betekenis | Bron | Beschikbaar | Ontbrekend |
|---|---|---|---|---|
| `circuit_prior_races` | aantal vorige races van de coureur op dit circuit (n) | `circuit_id` | vóór race N | 0 |
| `circuit_delta_shrunk` | n / (n + k) × (gemiddelde finish op dit circuit − gemiddelde finish over alle vorige races) | `finish_position`, `circuit_id` | vóór race N | 0 zonder circuithistorie |

Negatief betekent beter dan normaal. De shrinkage beschermt tegen ruis bij weinig races per
coureur-circuitcombinatie, wat in fase 4 als risico naar voren kwam.

### `teammate`

| Kolom | Betekenis | Bron | Beschikbaar | Ontbrekend |
|---|---|---|---|---|
| `teammate_quali_gap` | eigen kwalificatiepositie min het gemiddelde van de teamgenoten in race N; negatief = voor de teamgenoot | `quali_position`, `constructor_id` | na de kwalificatie | leeg zonder teamgenoot of als een van de kwalificatieposities ontbreekt |
| `teammate_beat_rate_{r}` | aandeel verslagen teamgenoten in de vorige r races van de coureur | `finish_position`, `constructor_id` | vóór race N | leeg zonder eerdere race met teamgenoot; races zonder teamgenoot tellen niet mee |

## Speciale gevallen

- De ruwe data blijft ongewijzigd: `build_features` geeft een nieuwe tabel terug met dezelfde
  index en rijvolgorde.
- Ontbrekende kwalificatie (22 rijen in 2014–2026): laatste plaats plus vlag in
  `starting_position`, leeg in `teammate_quali_gap`.
- Pitlane-starts (`grid = 0`) spelen geen rol, want `grid` wordt niet gebruikt.
- Niet-starters (DNS) tellen als uitval, diskwalificaties niet.
- Lege historische features (NaN) worden in de tabel niet ingevuld; elk model beslist zelf hoe
  het daarmee omgaat. Het Ridge-meetinstrument vult ze met de mediaan van de trainingsfold.
  In de huidige data zijn 63 rijen zonder coureursvorm (de eerste race van 2014 en debutanten)
  en 44 rijen zonder teamvorm (de eerste race van 2014 en de eerste race van elke nieuwe of
  hernoemde `constructor_id`).

## Leakage-tests

De tests in [`tests/test_features.py`](../tests/test_features.py) draaien offline op
synthetische data met twee seizoenen, zes races, twee circuits, wisselende statussen en een
sprint:

1. **Uitslag van race N wijzigen.** Voor elke race worden `finish_position`, `position_text`,
   `status`, `points`, `laps` en `sprint_points` van race N veranderd. De features van race N
   moeten identiek blijven. Als controle dat de test iets meet, moeten de features van latere
   races wél veranderen.
2. **Toekomst weglaten.** De features van race N zijn gelijk, of de races na N nu in de data
   zitten of niet. Dit vangt ook globale normalisatie en gecentreerde vensters.
3. **Teamgenoot in dezelfde race.** Andere uitslagen in race N veranderen de teamvorm van race N
   niet.

Tests 1 en 2 draaien met alle groepen aan, dus nieuwe groepen vallen er automatisch onder. Om de
tests zelf te controleren zijn tijdens de ontwikkeling bewust twee lekken ingebouwd: een rolling
gemiddelde zonder `shift(1)` en een `shift(1)` op coureursrijen voor teamvorm. Beide werden door
de tests gevangen.

## Meetinstrument

Het echte model volgt in fase 7. Om featuregroepen nu al te meten, gebruikt fase 6
`RidgeFeaturePredictor` ([`linear.py`](../src/f1_predictor/models/linear.py)): mediaan-imputatie,
standaardisatie en Ridge-regressie (`alpha = 1.0`) op `finish_position`, in elke
walk-forward-fold opnieuw getraind op alleen eerdere races. De voorspelde finish wordt omgezet in
een rangschikking; gelijke voorspellingen worden op `driver_id` gesorteerd. De groepen worden
cumulatief toegevoegd in de vaste volgorde van `FeatureConfig`. Alle experimenten en beide
baselines draaien in één `run_backtest`-aanroep met `min_train_races=20`, dus op exact dezelfde
races.

```powershell
uv run python -c "from f1_predictor.evaluation.backtest import run_feature_backtest; run_feature_backtest()"
```

De output is `output/backtest_features.csv`, met één rij per model en testrace en dezelfde
kolommen als `output/backtest_baselines.csv`.

## Experimentenlogboek

Run van 2026-11-23 op 13 seizoenen (2014–2026): 269 races, waarvan 249 testraces per model.
De baselines geven exact dezelfde scores als in fase 5. ΔMAE is het gemiddelde verschil per race
met de vorige regel (gepaard over dezelfde races), met de standaardfout tussen haakjes; negatief
is beter.

| Experiment | MAE | Spearman | Winnaar juist | Podiumoverlap | Top-10-overlap | ΔMAE (SE) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline kwalificatiepositie | 3.514 | 0.635 | 56.2% | 2.036 | 7.651 | – |
| Baseline kampioenschapspunten | 4.018 | 0.558 | 41.4% | 1.771 | 7.241 | – |
| `starting_position` | 3.523 | 0.634 | 56.2% | 2.036 | 7.651 | +0.010 (0.005)* |
| + `driver_form` | 3.468 | 0.648 | 54.6% | 1.992 | 7.767 | −0.056 (0.023) |
| + `team_form` | 3.457 | 0.651 | 54.2% | 2.036 | 7.759 | −0.011 (0.013) |
| + `championship` | 3.440 | 0.653 | 53.4% | 2.040 | 7.791 | −0.017 (0.009) |
| + `circuit_history` | 3.444 | 0.653 | 52.6% | 2.032 | 7.783 | +0.004 (0.005) |
| + `teammate` | 3.449 | 0.652 | 53.4% | 2.016 | 7.775 | +0.005 (0.010) |

\* ten opzichte van de kwalificatiebaseline.

**Conclusies**

1. `starting_position`: Ridge op alleen de kwalificatiepositie reproduceert de
   kwalificatiebaseline, zoals verwacht. Het kleine verschil komt van de 22 rijen zonder
   kwalificatiepositie, die de baseline altijd achteraan zet en Ridge via de vlag anders kan
   plaatsen.
2. `driver_form`: de duidelijkste winst (MAE −0.056, ongeveer 2.4 standaardfouten; Spearman
   +0.014). Recente vorm voegt informatie toe bovenop de startpositie.
3. `team_form`: een kleine verbetering die binnen de ruis valt. Teamsterkte zit waarschijnlijk
   al grotendeels in de kwalificatiepositie en de coureursvorm.
4. `championship`: een kleine, randgeldige verbetering (ongeveer 1.9 standaardfouten) en de
   beste MAE en Spearman van de reeks.
5. `circuit_history`: geen winst in een lineair model, in lijn met de kleine aantallen per
   coureur-circuitcombinatie uit fase 4.
6. `teammate`: geen winst; de kwalificatieposities bevatten de onderlinge vergelijking
   waarschijnlijk al grotendeels.
7. Over alles: de beste combinatie (t/m `championship`) verbetert de MAE met 0.074 posities ten
   opzichte van de kwalificatiebaseline (ongeveer 2.5 standaardfouten) en is in 51% van de races
   beter en in 39% slechter. Het aandeel juiste winnaars daalt wel van 56.2% naar 53.4%. Een
   waarschijnlijke verklaring is dat Ridge de fout over het hele veld minimaliseert en de
   polesitter door vormfeatures soms terugzet. Fase 7 kan dit met een rangschikkingsmodel
   (`XGBRanker`) of niet-lineaire interacties opnieuw bekijken.

De verschillen zijn klein, en dit zijn historische backtestscores, geen garantie voor
toekomstige races. Alle groepen blijven voorlopig standaard aan; de featurekeuze voor het model
valt in fase 7.

## Beperkingen

- Team-ID's worden niet gekoppeld bij naamswijzigingen (bijvoorbeeld `force_india` →
  `racing_point` → `aston_martin`, `toro_rosso` → `alphatauri` → `rb`, `renault` → `alpine`). Een
  nieuwe `constructor_id` begint zonder teamhistorie.
- Puntensystemen verschillen per periode: dubbele punten in de laatste race van 2014, een extra
  punt voor de snelste ronde in 2019–2024 en sprints vanaf 2021. Puntenfeatures zijn daardoor niet
  volledig vergelijkbaar over seizoenen.
- Vormvensters lopen over seizoensgrenzen en reglementswissels (2022, 2026) heen.
- De circuitfeature corrigeert niet voor teamsterkte: een coureur die van team wisselt, vergelijkt
  zijn circuitresultaten met een gemiddelde uit verschillende auto's.
- Finishposities zijn niet genormaliseerd voor de veldgrootte (18–22 deelnemers).
- Kwalificatietijden, weer, bandenstrategie en trainingsdata zitten niet in de data en vallen
  buiten deze fase.
