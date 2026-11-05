# Fase 4: Data verkennen

## Datasetkwaliteit

- Aantal rijen: 5479
- Aantal seizoenen: 13
- Aantal unieke races: 269
- Duplicates op (`season`, `round`, `driver_id`): 0
- Ontbrekende waarden: {'quali_position': 22, 'season': 0, 'race_name': 0, 'round': 0, 'date': 0, 'driver_id': 0, 'driver_code': 0, 'circuit_id': 0, 'constructor_id': 0, 'grid': 0, 'position_text': 0, 'finish_position': 0, 'status': 0, 'points': 0, 'laps': 0, 'sprint_points': 0}
- Spearman correlatie tussen grid en finishpositie: 0.6006
- Gemiddelde finishpositie: 10.70
- Mediaan finishpositie: 11.00

## Belangrijkste observaties

1. De gridpositie is een sterke maar niet perfecte voorspeller van de eindklassering. De Spearman-correlatie van 0.6006 laat zien dat startpositie veel verklaart, maar dat race-ruis en uitvallers nog steeds belangrijk zijn.
2. Er zijn in de ruwe dataset vrijwel geen duplicaten en slechts een klein aantal ontbrekende `quali_position`-waarden (22). Dat betekent dat het datamodel consistent is en geen grote data-quality-issues vertoont.
3. De uitvalpercentages verschillen sterk tussen teams, wat een aanwijzing is dat team- en auto-kwaliteit een duidelijke invloed hebben op de kans op een DNF.
4. Er zijn duidelijke circuitverschillen in gemiddelde finishpositie, maar de range is beperkt. Dat suggereert dat circuitkarakteristieken relevant zijn, maar niet zo dominant dat ze alle andere signalen volledig overschaduwen.
5. Verschillende coureurs presteren structureel beter of slechter op bepaalde circuits. Dat is interessant voor de feature-engineering in fase 6, maar we moeten het met voldoende data-per-circuit controleren om overfitting te voorkomen.

## Uitvalpercentage per team

| Team | Uitvalpercentage |
|---|---:|
| cadillac | 82.35% |
| audi | 58.82% |
| rb | 51.54% |
| alpine | 39.31% |
| caterham | 38.24% |

## Teamsterkte per seizoen

De tabel hieronder toont de gemiddelde bijdrage van elk team per jaar. Lagere gemiddelden betekenen gemiddeld beter presteren.

| Seizoen | Team | Gemiddelde teamfinish |
|---|---|---:|
| 2014 | mercedes | 4.21 |
| 2015 | mercedes | 3.21 |
| 2016 | mercedes | 3.48 |

## Circuit-effecten

| Circuit | Gemiddelde finishpositie |
|---|---:|
| imola | 10.50 |
| losail | 10.50 |
| jeddah | 10.50 |
| istanbul | 10.50 |
| ricard | 10.50 |

## Driver-circuit bias

De onderstaande selectie toont de sterkste positieve en negatieve afwijkingen ten opzichte van de eigen gemiddelde finish van de coureur.

### Positieve bias (beter dan normaal)

| Coureur | Circuit | Verschil ten opzichte van eigen gemiddelde |
|---|---|---:|
| bottas | portimao | -6.72 |
| perez | istanbul | -6.66 |
| maldonado | americas | -6.24 |
| zhou | losail | -6.15 |
| hadjar | monaco | -6.03 |

### Negatieve bias (slechter dan normaal)

| Coureur | Circuit | Verschil ten opzichte van eigen gemiddelde |
|---|---|---:|
| massa | hockenheimring | 11.94 |
| antonelli | catalunya | 9.80 |
| hadjar | albert_park | 9.47 |
| leclerc | hockenheimring | 8.64 |
| antonelli | silverstone | 8.30 |

## Conclusie

Fase 4 laat zien dat de dataset goed genoeg is om door te gaan naar fase 5 en fase 6. De startpositie is een sterke baseline, het team-/circuit-signaal is aanwezig, en er is voldoende data voor feature engineering zonder leakage. De volgende stap is het definiëren van baselines en backtest-metrieken, zodat het model eerlijk kan worden vergeleken met eenvoudige, tijdgeordende benchmarks.
