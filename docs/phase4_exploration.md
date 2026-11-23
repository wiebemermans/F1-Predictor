# Fase 4: Data verkennen

Dit rapport wordt gegenereerd door `uv run python -m f1_predictor.evaluation.explore`.

## Datasetkwaliteit

- Aantal rijen: 5479
- Aantal seizoenen: 13
- Aantal unieke races: 269
- Duplicates op (`season`, `round`, `driver_id`): 0
- Ontbrekende waarden: {'quali_position': 22, 'season': 0, 'race_name': 0, 'round': 0, 'date': 0, 'driver_id': 0, 'driver_code': 0, 'circuit_id': 0, 'constructor_id': 0, 'grid': 0, 'position_text': 0, 'finish_position': 0, 'status': 0, 'points': 0, 'laps': 0, 'sprint_points': 0}
- Spearman correlatie tussen startpositie en finishpositie: 0.6257
- Gemiddelde finishpositie: 10.70
- Mediaan finishpositie: 11.00

## Methode en definities

- **Startpositie:** `grid`, waarbij een pitlane-start (`grid = 0`) als laatste startplaats van het veld telt. De ruwe data blijft ongewijzigd; deze vertaling gebeurt alleen in de analyse.
- **Uitval:** elke status behalve gefinisht en gediskwalificeerd, dus DNF en niet-starters (DNS). Gefinisht is `Finished`, `+N Lap(s)` (oudere seizoenen) of `Lapped` (recentere seizoenen). Diskwalificaties tellen niet als uitval. De status is gebruikt in plaats van `position_text`, omdat uitvallers die ver genoeg kwamen toch een numerieke klassering krijgen.
- **Circuit-effecten:** de gemiddelde finishpositie per circuit is geen bruikbare maat (bij 20 deelnemers altijd 10,5). Daarom wordt per race de Spearman-correlatie tussen startpositie en finish berekend en per circuit gemiddeld; lager betekent een minder voorspelbaar circuit.
- **Driver-circuit bias:** alleen combinaties met minstens 3 races; ruw verschil zonder correctie voor teamsterkte.

## Bereiken en domeincontrole

Numerieke bereiken zijn gecontroleerd op onmogelijke waarden. Een generieke IQR-regel is hier niet toegepast: startplaatsen, rondes, punten en rondetotalen zijn discrete waarden die per seizoen en racecontext verschillen. `grid = 0` en `laps = 0` zijn geldige pitlane-starten/niet gestarte of vroeg uitgevallen deelnemers en worden niet als fout aangemerkt.

| Kolom | Minimum | Maximum |
|---|---:|---:|
| grid | 0 | 22 |
| finish_position | 1 | 22 |
| points | 0 | 50 |
| sprint_points | 0 | 8 |
| laps | 0 | 87 |

| Domeincontrole | Ongeldige rijen |
|---|---:|
| negative grid | 0 |
| finish position outside race field | 0 |
| negative points | 0 |
| negative sprint points | 0 |
| negative laps | 0 |

## Belangrijkste observaties

1. De startpositie is een sterke maar niet perfecte voorspeller van de eindklassering (Spearman 0.6257). Race-ruis en uitvallers blijven belangrijk.
2. Er zijn geen duplicaten en slechts 22 ontbrekende `quali_position`-waarden; de dataset is consistent.
3. Het uitvalpercentage per seizoen ligt tussen 10.9% en 23.8%; per team tussen 7.8% en 38.2%. Teams met weinig starts (nieuwe teams, lopend seizoen) hebben onbetrouwbare percentages; kijk daarom ook naar het aantal starts.
4. Circuits verschillen in voorspelbaarheid: de gemiddelde Spearman-correlatie tussen start en finish loopt van 0.35 tot 0.75 (circuits met minstens 3 races). Op chaotische circuits weegt de startpositie minder.
5. Sommige coureurs presteren op bepaalde circuits beter of slechter dan hun eigen gemiddelde, maar het aantal races per combinatie is klein en teamsterkte is niet gecorrigeerd. Een circuit-feature in fase 6 heeft shrinkage nodig.

## Uitvalpercentage per seizoen

| Seizoen | Starts | Uitvalpercentage |
|---|---:|---:|
| 2014 | 407 | 20.88% |
| 2015 | 378 | 21.43% |
| 2016 | 462 | 18.18% |
| 2017 | 400 | 23.75% |
| 2018 | 420 | 19.52% |
| 2019 | 420 | 13.81% |
| 2020 | 340 | 16.76% |
| 2021 | 440 | 13.18% |
| 2022 | 440 | 16.82% |
| 2023 | 440 | 14.55% |
| 2024 | 479 | 10.86% |
| 2025 | 479 | 11.27% |
| 2026 | 374 | 20.86% |

## Uitvalpercentage per team (hoogste 5)

| Team | Starts | Uitvalpercentage |
|---|---:|---:|
| cadillac | 34 | 38.24% |
| caterham | 34 | 38.24% |
| lotus_f1 | 76 | 36.84% |
| toro_rosso | 242 | 25.62% |
| renault | 200 | 23.50% |

## Sterkste team per seizoen

Het team met de laagste gemiddelde finishpositie per seizoen; lager is beter. Alle teams staan in `output/phase4_team_strength_by_season.csv`.

| Seizoen | Team | Gemiddelde teamfinish |
|---|---|---:|
| 2014 | mercedes | 4.21 |
| 2015 | mercedes | 3.21 |
| 2016 | mercedes | 3.48 |
| 2017 | mercedes | 3.17 |
| 2018 | mercedes | 3.86 |
| 2019 | mercedes | 3.24 |
| 2020 | mercedes | 3.74 |
| 2021 | mercedes | 5.41 |
| 2022 | red_bull | 4.45 |
| 2023 | red_bull | 3.57 |
| 2024 | mclaren | 4.71 |
| 2025 | mclaren | 4.23 |
| 2026 | mercedes | 5.15 |

## Circuit-effecten

Circuits met minstens 3 races. Alle circuits staan in `output/phase4_circuit_effects.csv`.

### Minst voorspelbaar

| Circuit | Races | Spearman start-finish | Gem. abs. positiewijziging | Uitval |
|---|---:|---:|---:|---:|
| vegas | 3 | 0.35 | 4.73 | 13.33% |
| hockenheimring | 4 | 0.40 | 4.52 | 21.43% |
| americas | 11 | 0.48 | 4.18 | 17.27% |
| albert_park | 11 | 0.53 | 4.07 | 26.91% |
| baku | 10 | 0.53 | 4.22 | 22.06% |

### Meest voorspelbaar

| Circuit | Races | Spearman start-finish | Gem. abs. positiewijziging | Uitval |
|---|---:|---:|---:|---:|
| suzuka | 11 | 0.75 | 2.79 | 11.06% |
| miami | 5 | 0.74 | 3.15 | 13.73% |
| jeddah | 5 | 0.73 | 2.96 | 18.00% |
| ricard | 4 | 0.71 | 3.00 | 12.50% |
| rodriguez | 10 | 0.70 | 3.24 | 15.84% |

## Driver-circuit bias

De sterkste afwijkingen ten opzichte van de eigen gemiddelde finish van de coureur.

### Positieve bias (beter dan normaal)

| Coureur | Circuit | Races | Verschil t.o.v. eigen gemiddelde |
|---|---|---:|---:|
| russell | vegas | 3 | -5.64 |
| alonso | losail | 4 | -5.54 |
| bottas | sochi | 8 | -5.47 |
| sainz | vegas | 3 | -5.24 |
| piastri | losail | 3 | -4.70 |

### Negatieve bias (slechter dan normaal)

| Coureur | Circuit | Races | Verschil t.o.v. eigen gemiddelde |
|---|---|---:|---:|
| kvyat | baku | 3 | 7.78 |
| norris | vegas | 3 | 7.67 |
| norris | villeneuve | 6 | 7.01 |
| button | bahrain | 3 | 6.87 |
| hamilton | losail | 4 | 6.53 |

## Conclusie

De dataset is goed genoeg voor fase 5 en fase 6. De startpositie is de sterkste enkele voorspeller; teamsterkte, uitvalrisico en circuitkarakter zijn aanvullende signalen. Uitval- en circuitfeatures moeten rekening houden met kleine aantallen en met beide statusvocabulaires van Jolpica.
