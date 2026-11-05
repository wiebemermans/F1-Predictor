# Fase 5: Baselines en walk-forward-backtest

## Voorspelmoment en input

De voorspelling is bedoeld voor na de kwalificatie en vóór de race. De eerste
baseline gebruikt `quali_position`, omdat de huidige parquet-data geen
kwalificatietijden bevat. `grid` is de feitelijke startpositie en kan
gridstraffen bevatten; die baseline hoort bij een ander voorspelmoment en
wordt niet met de kwalificatiebaseline vermengd.

De uitslag van de test race wordt alleen gebruikt als evaluatielabel
(`finish_position`). De kampioenschapsbaseline gebruikt uitsluitend
`points` en `sprint_points` uit eerdere races van hetzelfde seizoen. De stand
wordt aan het begin van elk seizoen opnieuw opgebouwd. Punten worden niet
teruggewogen met de officiële countbackregels.

## Baselines en tie-breaks

- `QualifyingPositionBaseline`: lagere kwalificatiepositie voorspelt een
  hogere eindrang. Ontbrekende posities komen achteraan. Gelijke posities
  worden op `driver_id` alfabetisch gesorteerd.
- `ChampionshipPointsBaseline`: meer eerdere race- plus sprintpunten voorspelt
  een hogere eindrang. Een coureur zonder punten in dat seizoen krijgt nul;
  gelijke punten worden op `driver_id` alfabetisch gesorteerd.

Beide predictors bieden `fit(train).predict(race)` en geven per inputrij een
unieke voorspelde rang van 1 tot en met het aantal deelnemers terug.

## Metrics

`score_race` rapporteert per race:

- `mae_positions`: gemiddelde absolute afwijking van voorspelde rang versus
  officiële `finish_position`; lager is beter.
- `spearman`: Spearman-correlatie tussen voorspelde rang en finishpositie;
  hoger is beter. Gelijke officiële posities krijgen de standaard gemiddelde
  rang. Bij één deelnemer is de correlatie niet gedefinieerd en wordt deze
  leeg (`None`) gerapporteerd.
- `winner_hit`: 1 als de voorspelde en officiële winnaar overeenkomen,
  anders 0. Bij gelijke winnaarspositie bepaalt `driver_id` de volgorde.
- `podium_overlap`: aantal gedeelde coureurs in voorspeld en officieel
  podium (0–3).
- `top10_overlap`: aantal gedeelde coureurs in voorspelde en officiële top 10;
  bij minder dan tien deelnemers is de overlap maximaal het deelnemersaantal.

De officiële finishpositie wordt ook voor uitvallers gebruikt. De evaluatie
vereist een unieke, eindige voorspelde rangschikking en een geldige
finishpositie voor iedere coureur.

## Walk-forward en output

`run_backtest` sorteert op seizoen en ronde. Met de standaard
`min_train_races=20` zijn de eerste twintig historische races alleen
trainingshistorie; vanaf de volgende race wordt iedere race afzonderlijk
gescoord. Elke predictor krijgt alleen races die strikt eerder zijn
verreden. Beide baselines gebruiken identieke test-races.

Genereer de CSV vanuit de beschikbare parquet-bestanden:

```powershell
uv run python -c "from f1_predictor.evaluation.backtest import run_baseline_backtest; run_baseline_backtest()"
```

De output is `output/backtest_baselines.csv`, met één rij per baseline en
test-race. De data wordt atomisch weggeschreven. Om de testfolds op een kleine
synthetische dataset te inspecteren, kan `run_backtest` rechtstreeks worden
aangeroepen met een DataFrame en predictor factories.

## Resultaten van de huidige run

Op de aanwezige 13 seizoenen (2014–2026) staan 269 races. Na de eerste
twintig trainingsraces zijn 249 races gescoord voor elk van de twee baselines
(498 rijen totaal); alle scorevelden zijn gevuld.

| Baseline | MAE (lager beter) | Spearman (hoger beter) | Winnaar juist | Podiumoverlap | Top-10-overlap |
|---|---:|---:|---:|---:|---:|
| Kwalificatiepositie | 3.514 | 0.635 | 56.2% | 2.036 | 7.651 |
| Kampioenschapspunten | 4.018 | 0.558 | 41.4% | 1.771 | 7.241 |

Dit zijn geaggregeerde historische scores, geen garantie voor toekomstige
races of onafhankelijk gehouden testdata. Ze vormen de reproduceerbare
referentie voor het eerste model. De implementatiedetails staan in
[`metrics.py`](../src/f1_predictor/evaluation/metrics.py),
[`baselines.py`](../src/f1_predictor/models/baselines.py) en
[`backtest.py`](../src/f1_predictor/evaluation/backtest.py).
