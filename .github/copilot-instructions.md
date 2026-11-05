# LLM-instructions

## Werkwijze

- Lees de relevante implementatie, aanroepers, tests en projectdocumentatie voordat je code wijzigt. Volg bestaande projectpatronen; laat niet-gerelateerde problemen ongemoeid.
- Voer de kleinste wijziging uit die de vraag volledig oplost. Vermijd speculatieve abstraheringen, nieuwe dependencies en scope-uitbreiding.
- Als requirements onduidelijk zijn of een wijziging de architectuur of het zichtbare gedrag wezenlijk beïnvloedt, vraag eerst om verduidelijking. Benoem aannames als je veilig met redelijke defaults kunt doorgaan.
- Houd namen en functies duidelijk en doelgericht. Voeg alleen commentaar toe voor niet-vanzelfsprekende keuzes; werk verouderde documentatie bij wanneer gedrag verandert.

## Python, tests en fouten

- Respecteer de Python-versie en tooling in `pyproject.toml`; voer tests en Ruff uit met `uv run`.
- Voeg of wijzig tests voor nieuw gedrag en bugfixes. Tests horen deterministisch en offline waar mogelijk te zijn; vervang netwerk, klok of externe diensten aan de grens.
- Valideer invoer aan de rand van de applicatie. Vang alleen fouten op als de code er betekenisvol op kan reageren; log of toon fouten expliciet en verberg ze niet met stille defaults.
- Voer gerichte tests uit tijdens werk en de volledige relevante suite vóór afronding. Meld eerlijk welke controles wel of niet zijn uitgevoerd.

## F1-data en evaluatie

- Voorkom datalekken: een voorspelling voor race N mag alleen gegevens gebruiken die op het gekozen voorspelmoment beschikbaar zijn. Resultaten, punten en uitvalstatus van race N zijn labels, geen features voor die race.
- Houd kwalificatiepositie en definitieve startgrid uit elkaar: `quali_position` is beschikbaar na kwalificatie; `grid` kan latere gridstraffen bevatten.
- Behoud de ruwe Jolpica-data. Pas speciale gevallen zoals pitlane-starten, niet-starters, uitvallers en ontbrekende kwalificatie niet stilzwijgend aan; documenteer expliciet hoe een feature of metric ermee omgaat.
- Vergelijk modellen en baselines op dezelfde races, met chronologische splitsingen en reproduceerbare metrics.

## Git-branches en commits

- Controleer vóór wijzigingen, commits of pushes de huidige branch, remote en werkboomstatus; behoud bestaande wijzigingen van de gebruiker.
- Controleer of de actieve branch past bij de volledige scope. Gebruik voor nieuw werk een beschrijvende branchnaam, bijvoorbeeld `feat/phase5-backtest` of `docs/phase5-documentation`; commit niet automatisch op een bestaande branch met een niet-passende naam.
- Hernoem of verlaat nooit stilzwijgend een bestaande branch. Als wisselen wijzigingen kan verplaatsen of overschrijven, stop en vraag om bevestiging.
- Commit alleen bestanden die bij de afgesproken wijziging horen, met een beknopte, beschrijvende boodschap. Push alleen naar de geverifieerde remote en branch; push niet naar `main` tenzij de gebruiker dat expliciet vraagt.
- Voeg bij commits de afgesproken Copilot-co-auteursvermelding toe. Commit nooit secrets, credentials, caches of gegenereerde bestanden tenzij het project die output bewust versiebeheert.
