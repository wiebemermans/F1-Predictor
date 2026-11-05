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
uv run f1 fetch --start 2014 --end 2026
```

`--start` is standaard 2014 en `--end` het huidige kalenderjaar. Een bestaand
bestand van een afgesloten seizoen wordt overgeslagen. Het lopende seizoen
wordt opnieuw opgehaald; `--force` haalt ook afgesloten seizoenen opnieuw op.
Een fout in één seizoen wordt gelogd, waarna de andere seizoenen nog worden
verwerkt. Als een of meer seizoenen mislukken, eindigt de CLI met exitcode 1.

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
