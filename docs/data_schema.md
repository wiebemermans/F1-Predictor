# Dataschema: `data/raw/results_{season}.parquet`

Eén rij per coureur per race. Sleutel: (`season`, `round`, `driver_id`).

| Kolom | Type | Bron | Betekenis |
|---|---|---|---|
| season | int | results | Seizoen |
| round | int | results | Rondenummer in het seizoen |
| race_name | str | results | Naam van de GP |
| circuit_id | str | results | Circuit-ID (voor circuit-historiek) |
| date | datetime | results | Racedatum |
| driver_id | str | results | Stabiele coureur-ID |
| driver_code | str | results | Driveletter-code (kan ontbreken) |
| constructor_id | str | results | Team-ID |
| grid | int | results | Werkelijke startpositie; 0 = pitlane-start |
| quali_position | Int64 (nullable) | qualifying | Positie in de kwalificatie; leeg als geen kwalificatietijd |
| finish_position | int | results | Eindklassering, ook voor uitvallers |
| position_text | str | results | Cijfer = geklasseerd; letter = R/D/E/W/F/N (uitgevallen, gediskwalificeerd, ...) |
| status | str | results | Reden van finish/uitval ("Finished", "+1 Lap", "Engine", ...) |
| points | float | results | Punten uit de race |
| laps | int | results | Afgelegde ronden |
| sprint_points | float | sprint | Sprintpunten (0.0 als er geen sprint was) |

## Vóór de race bekend (mogen features worden)
`grid`, `quali_position`, `circuit_id`, `date`, `constructor_id`

## Uitkomsten (nooit als feature voor dezelfde race)
`finish_position`, `position_text`, `status`, `points`, `laps`, `sprint_points`

## Bewust niet opgelost in de rauwe data
Pitlane-starts (grid = 0), diskwalificaties, niet-starters: we bewaren ze ongewijzigd.
Afhandeling gebeurt in feature engineering (Fase 6).
