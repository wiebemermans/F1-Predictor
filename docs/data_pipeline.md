# Datapipeline

## Bron en ophalen

Racedata komt van de Ergast-compatibele Jolpica-F1 API. De client haalt
resultaten, kwalificatie en sprintuitslagen per seizoen op. Requests worden
gepageerd (maximaal 100 races per pagina), gecachet in `data/cache/` en met
een timeout van 30 seconden uitgevoerd. Succesvolle live requests krijgen
een pauze; HTTP 429- en 5xx-responses krijgen maximaal vijf pogingen. De cache
verloopt standaard na één uur.

Haal data op met:

```powershell
uv run f1 fetch
uv run f1 fetch --start 2014 --end 2026   # expliciet bereik
```

Zonder opties komen `--start` en `--end` uit `data.start_season` (2014) en
`data.end_season` in `config/config.yaml`; `end_season: null` is het huidige
kalenderjaar. Een bestaand bestand van een afgesloten seizoen wordt
overgeslagen. Het lopende seizoen wordt opnieuw opgehaald; `--force` haalt ook
afgesloten seizoenen opnieuw op. Een fout in één seizoen wordt gelogd, waarna
de andere seizoenen nog worden verwerkt. Als een of meer seizoenen mislukken,
eindigt de CLI met exitcode 1. Heeft het lopende jaar nog geen enkele
race-uitslag (vóór de eerste race), dan wordt dat seizoen met een waarschuwing
overgeslagen; dat telt niet als fout. Een afgelopen seizoen zonder uitslagen
blijft wel een fout.

## Deelnemers van een komende race

De seizoensbestanden bevatten alleen races met een uitslag. Voor een race die
nog moet komen haalt `f1 predict` na de kwalificatie de kwalificatie-uitslag
van die ene race op (`/{seizoen}/{ronde}/qualifying`). Daaruit komen de
deelnemers, hun team, `driver_code` en `quali_position`, plus racenaam,
circuit en datum. Dit verzoek gaat bewust niet via de cache: een lege respons
van vóór de kwalificatie zou anders tot een uur lang blijven hangen. De
seizoensbestanden in `data/raw/` worden hierdoor niet aangepast. Zie
[usage.md](usage.md).

## Racekalender

`f1 predict --next` haalt de kalender van het lopende jaar op (`/{seizoen}/`, ook zonder
cache: een verplaatste of geschrapte race moet meteen tellen). Per race komen daaruit seizoen,
ronde, racenaam, circuit, racedatum en de start van race en kwalificatie in UTC (`date`/`time`
van de race en van het blok `Qualifying`). Ontbreekt de racetijd, dan telt middernacht UTC van
de racedag; zo geldt een race eerder als begonnen en wordt er nooit na de start voorspeld.
Zonder blok `Qualifying` is de kwalificatiestart leeg. Sprintsessies worden niet gebruikt. Een
seizoen waarvan de kalender nog niet gepubliceerd is, geeft een lege kalender. De kalender wordt
niet opgeslagen. Hoe de volgende race gekozen wordt, staat in
[automation.md](automation.md#hoe-f1-predict---next-beslist).

## Opbouw en opslag

Voor elk seizoen worden results, qualifying en sprint op
(`season`, `round`, `driver_id`) samengevoegd. Als er geen kwalificatiedata is,
blijft `quali_position` leeg; ontbrekende sprintpunten worden `0.0`. Een
seizoentabel wordt eerst naar een tijdelijk bestand geschreven en daarna
atomisch hernoemd naar `data/raw/results_{season}.parquet`.

De kolommen en betekenis staan in [data_schema.md](./data_schema.md). Ruwe
data wordt bewust niet aangepast om bijvoorbeeld pitlane-starten, niet-starters
of diskwalificaties te verbergen. Gebruik voor alle afgeleide interpretaties
de brontabel en de gedocumenteerde regels van de betreffende analyse.

## Featuretabel

Vanuit de seizoensbestanden bouwt `f1 features` één featuretabel voor alle races:

```powershell
uv run f1 features
```

De tabel wordt atomisch weggeschreven naar `data/processed/features.parquet` en bevat de ruwe
kolommen ongewijzigd, plus `race_index` en de featurekolommen. Zonder bestanden in `data/raw/`
eindigt het commando met exitcode 1. Featuredefinities, configuratie en leakage-tests staan in
[features.md](./features.md).
