# Fase 10: Automatisering met GitHub Actions

Twee geplande workflows laten het project zichzelf draaien. Na elke kwalificatie voorspelt
`predict.yml` de race. Na het raceweekend haalt `post-race.yml` de uitslag op, werkt de
backtests bij en zet de voorspelling van vooraf naast de werkelijkheid. Al het rekenwerk gebeurt
in GitHub Actions; het resultaat zijn JSON- en CSV-bestanden in `output/` op `main`. Een website
of een eigen server hoeft die bestanden alleen te lezen.

## Overzicht

| Workflow | Wanneer | Wat | Commit naar `main` |
|---|---|---|---|
| [`predict.yml`](../.github/workflows/predict.yml) ("Voorspelling") | elke 3 uur van vrijdag tot en met zondag (UTC, op :17); handmatig; op een PR die deze workflow wijzigt | `f1 fetch`, daarna `f1 predict --next` | `output/predictions/{seizoen}-{ronde}.json` en `latest.json` |
| [`post-race.yml`](../.github/workflows/post-race.yml) ("Na de race") | maandag en dinsdag 06:41 UTC; handmatig; op een PR die deze workflow wijzigt | `f1 fetch`, de drie backtestsuites, `f1 evaluate` | `output/backtest_*.csv`, `model_comparison.csv/.json`, `prediction_history.json` |
| [`ci.yml`](../.github/workflows/ci.yml) | elke push naar `main` en elke PR | lint en tests ([testing.md](testing.md#ci)) | nooit |

Komt er niets nieuws bij, dan verandert er geen bestand en volgt er geen commit. Er zijn dus veel
runs, maar weinig commits: normaal één per raceweekend van elke workflow.

## Planning

De voorspelling kan pas na de kwalificatie en moet vóór de start van de race klaar zijn. Dat
venster schuift per circuit: van zaterdag 04:00 UTC (Las Vegas) tot zaterdag 21:00 UTC (Austin,
Mexico). Sommige weekenden hebben een race op zaterdag en de kwalificatie op vrijdag (Bahrein en
Saoedi-Arabië in 2024). Eén vaste cron op "zaterdagavond en zondagochtend" mist daarom races: in
Australië en Japan begint de race al zondag rond 05:00 UTC.

Daarom draait `predict.yml` vaak en beslist de kalender of er iets te doen is (zie hieronder).
Elke 3 uur van vrijdag tot en met zondag levert minstens zes pogingen in elk venster op. Een
vertraagde of overgeslagen run kost dus hooguit een paar uur. De runs vallen op :17, omdat
GitHub geplande runs op het hele uur het vaakst vertraagt. Een run zonder werk is kort:
installeren, het lopende seizoen ophalen en de kalender bekijken (een publieke repository heeft
onbeperkt Actions-minuten).

| Race (2026) | Kwalificatie (UTC) | Race (UTC) | Eerste run na de kwalificatie |
|---|---|---|---|
| Las Vegas | za 04:00 | zo 04:00 | za 06:17 |
| Qatar | za 18:00 | zo 16:00 | za 21:17 (de run van 18:17 valt tijdens de kwalificatie) |
| Verenigde Staten | za 21:00 | zo 20:00 | zo 00:17 |

`post-race.yml` draait maandagochtend, als de uitslag van zondag zeker in Jolpica staat. De run
van dinsdag is een herkansing; als maandag gelukt is, verandert er niets en volgt er geen commit.

## Hoe `f1 predict --next` beslist

1. Haal de racekalender van het lopende jaar (UTC) op bij Jolpica (`/{seizoen}/`, zonder cache:
   een verplaatste race moet meteen tellen).
2. De volgende race is de eerste race waarvan de start nog niet voorbij is. Is er geen meer
   (seizoen voorbij, of de kalender van het nieuwe jaar is nog niet gepubliceerd): niets te
   doen.
3. Is de kwalificatie van die race nog niet begonnen: niets te doen. Daardoor faalt een run op
   zondagavond ook niet als de uitslag van de race van die middag nog niet in Jolpica staat.
4. Staat de kwalificatie-uitslag nog niet in Jolpica (de sessie loopt nog, of is nog niet
   verwerkt): niets te doen; een volgende run probeert het opnieuw.
5. Anders: voorspel de race met `f1 predict`, zoals beschreven in [usage.md](usage.md).

"Niets te doen" eindigt met exitcode 0 en een regel in het log; de run is groen. Er wordt nooit
na de start van een race voorspeld. Echte fouten blijven fouten (exitcode 1, een rode run en een
mail van GitHub): Jolpica onbereikbaar, of een eerdere ronde die in de data ontbreekt.

## Wat de workflows doen

Beide workflows beginnen hetzelfde:

1. Checkout van de nieuwste stand van de branch bij de start van de job (`ref: github.ref`),
   met de volledige historie (`fetch-depth: 0`), zodat `git pull --rebase` werkt. Zonder die
   `ref` neemt GitHub de commit van het moment waarop de run werd aangemaakt. Een run die in de
   wachtrij stond, deed dan het werk van de run vóór hem opnieuw en botste bij het committen op
   diens commit (zo gebeurd bij de eerste run na de merge).
2. uv installeren en `uv sync --locked`, met `UV_NO_DEV=1`: zonder de dev-tools (jupyter,
   pytest, ruff), ook niet bij `uv run`.
3. `data/raw/` uit de cache halen en `uv run f1 fetch` uitvoeren. Afgesloten seizoenen staan al
   in de cache, dus normaal wordt alleen het lopende seizoen opgehaald. Is de data veranderd,
   dan gaat ze meteen terug in de cache, ook als een latere stap faalt.

Daarna doet `predict.yml` `f1 predict --next`, of `f1 predict --season S --round R` als die
invoer bij een handmatige run is ingevuld. `post-race.yml` doet `f1 backtest --suite baselines`,
`--suite features` en `f1 backtest` (samen enkele minuten), daarna `f1 evaluate`.

Tot slot de commit, alleen op `main`. `git add output/`; is er niets gewijzigd, dan stopt de
stap. Anders volgt een commit door `github-actions[bot]`, `git pull --rebase` en een push. Op een
andere branch of een PR wordt er nooit gecommit: de uitvoer komt dan als artefact bij de run. Zo
kan een wijziging aan een workflow op de PR zelf getest worden.

Beide workflows delen de concurrency-groep `output-commit`: er commit er altijd maar één
tegelijk. Wacht er al een run, dan vervangt een nieuwe geplande run die wachtende run; omdat de
runs elkaars werk overdoen, gaat er niets verloren.

## Wanneer er een commit komt

- **Alleen bij een inhoudelijke wijziging.** JSON-bestanden worden alleen herschreven als hun
  inhoud verandert. `generated_at` en het `run`-blok tellen niet mee, want die veranderen bij
  elke run. `generated_at` is daardoor het moment waarop deze inhoud voor het eerst ontstond:
  bij een voorspelling de run na de kwalificatie, niet de laatste run van het weekend. De CSV's
  van de backtests zijn deterministisch en veranderen alleen bij nieuwe of gecorrigeerde races.
- **De voorspelling van vooraf blijft staan.** Een voorspelling met `result_known: false`
  wordt nooit vervangen door een retrospectieve (`result_known: true`), ook niet bij een
  handmatige run na de race. Een gecorrigeerde kwalificatie vóór de race geeft wel een nieuwe
  versie. Zo kan `f1 evaluate` na de race precies de voorspelling scoren die vooraf vastlag.
- **Linux en Windows verschillen bij XGBoost.** De vastgelegde CSV's zijn op Windows gemaakt,
  de workflows draaien op Linux. Een test op de PR (november 2026, zelfde data en
  bibliotheekversies) gaf voor de baselines, Ridge en de featurebacktest byte-identieke CSV's.
  De drie XGBoost-modellen rangschikken in een deel van de races anders: de MAE verschilde in
  446 van de 1518 rijen en de voorspelde winnaar in 20 races. Op de holdout verschuiven de
  gemiddelden weinig: het eindmodel haalde een MAE van 3,306 op Windows en 3,291 op Linux, en een
  podiumoverlap van 2,018 en 2,053. Op dezelfde machine blijven de resultaten reproduceerbaar.
  Daarom vervangt de eerste commit van `post-race.yml` de modelrijen door de Linux-cijfers. Vanaf
  dan rekenen backtest en voorspellingen op dezelfde runner, maar de cijfers in
  [model.md](model.md) en een lokale run op Windows kunnen licht afwijken.

## Rechten en secrets

Er zijn geen secrets nodig: Jolpica is een open API. De workflows vragen voor het automatische
`GITHUB_TOKEN` alleen `contents: write`, om te kunnen pushen. De bot pusht rechtstreeks naar
`main` en raakt daarbij alleen `output/`. Dat is een bewuste keuze. De regel "nooit naar `main`
pushen" uit de LLM-instructies geldt voor agents die code wijzigen, niet voor deze bot. Een push
met `GITHUB_TOKEN` start geen nieuwe workflows, dus ook geen CI. Dat is hier geen gemis: de bot
wijzigt geen code.

`main` heeft geen branch protection. Komt die er, dan moet de bot een uitzondering krijgen, of
moeten de workflows naar een aparte branch committen.

## Caching

- **Pakketten.** `setup-uv` bewaart de uv-cache (eigen sleutel met `cache-suffix: automation`,
  omdat deze workflows andere pakketten installeren dan `ci.yml`).
- **Ruwe data.** `data/raw/*.parquet` (ongeveer 200 kB) staat in de Actions-cache met de sleutel
  `jolpica-raw-{jaar}-{hash van de bestanden}`. Een run haalt de jongste versie van dat jaar op
  en bewaart alleen een nieuwe versie als de data veranderd is. Het jaar staat in de sleutel
  omdat `f1 fetch` een afgesloten seizoen niet opnieuw ophaalt zolang het bestand bestaat.
  In januari begint zo elke cache leeg: één volledige fetch (ongeveer 140 requests, 1 à 2
  minuten) legt dan ook de laatste races van vorig jaar vast. GitHub wist een cache die 7 dagen
  niet gebruikt is; ook dan volgt gewoon een volledige fetch.
- **Niet gecachet.** De HTTP-cache van de Jolpica-client (`data/cache/`, verloopt na een uur)
  en FastF1. FastF1 wordt (nog) niet gebruikt; het stappenplan noemde het als mogelijke
  uitbreiding.

## Gegenereerde bestanden

| Bestand | Workflow | Inhoud |
|---|---|---|
| `output/predictions/{seizoen}-{ronde}.json` | `predict.yml` | de voorspelling van vóór de race ([contract](usage.md#voorspelling-outputpredictionsseizoen-rondejson)) |
| `output/predictions/latest.json` | `predict.yml` | kopie van de chronologisch laatste voorspelling |
| `output/backtest_baselines.csv`, `backtest_features.csv`, `backtest_models.csv`, `model_comparison.csv` | `post-race.yml` | scores per race en samenvattingen, nu inclusief de laatste race |
| `output/model_comparison.json` | `post-race.yml` | samenvatting voor een website ([contract](usage.md#modelvergelijking-outputmodel_comparisonjson)) |
| `output/prediction_history.json` | `post-race.yml` | elke voorspelling van vooraf naast de uitslag ([contract](usage.md#historiek-outputprediction_historyjson)) |

De cijfers in de fasedocumenten ([phase5_baselines.md](phase5_baselines.md),
[features.md](features.md), [model.md](model.md)) zijn momentopnamen bij het afronden van die
fase. De bestanden in `output/` lopen voortaan met elke race mee.

## Handmatig uitvoeren

Via het tabblad Actions → workflow → "Run workflow", of met de GitHub CLI:

```powershell
gh workflow run predict.yml                                # volgende race, zoals de planning
gh workflow run predict.yml -f season=2026 -f round=22     # één bepaalde race
gh workflow run post-race.yml                              # uitslag, backtests, historiek
gh run list --workflow predict.yml                         # recente runs
gh run view --log                                          # log van een run (kies er een)
gh run rerun <run-id>                                      # een mislukte run opnieuw
```

Met `season` en `round` voorspelt de workflow die race, zoals `f1 predict --season --round`. Is
de race al gereden, dan is dat een retrospectieve voorspelling. Die vervangt een bestaande
voorspelling van vooraf niet en telt niet mee in de historiek.

Lokaal werken dezelfde commando's ([usage.md](usage.md)). Doe eerst `git pull`: de bot commit
in `output/`, en een lokale voorspelling van een race die de bot al voorspelde, geeft anders een
conflict.

## Aandachtspunten

- **Vertraagde of overgeslagen runs.** GitHub garandeert geen exacte starttijd voor geplande
  workflows; bij drukte schuiven ze op of vallen ze weg. De vele pogingen per venster vangen dat
  op.
- **Uitgeschakeld na 60 dagen.** In een publieke repository schakelt GitHub geplande workflows
  uit als er 60 dagen geen activiteit (commits) is geweest, en mailt het vooraf. De winterstop is
  langer: na de laatste race (6 december 2026) en de bijbehorende commit valt dat rond begin
  februari 2027, tenzij er tussendoor gecommit wordt. Zet de workflows vóór het nieuwe seizoen
  weer aan met `gh workflow enable predict.yml` en `gh workflow enable post-race.yml` (of via het
  tabblad Actions); `gh workflow list --all` toont de status. Een commit die de cron van een
  workflow wijzigt, zet die workflow ook weer aan.
- **Stille misser.** Verwerkt Jolpica de kwalificatie pas na de start van de race, dan komt er
  geen voorspelling, en ook geen foutmelding: elke run eindigt met "niets te doen". Kijk bij
  twijfel zondag in de log van de laatste run, of start de workflow met de hand.
- **`git_dirty` in bot-uitvoer.** In `post-race.yml` schrijven de backtests hun CSV's vóór het
  JSON-bestand, dus het `run`-blok van een nieuwe `model_comparison.json` meldt
  `git_dirty: true`. Dat zijn die net geschreven CSV's, geen handmatige wijzigingen.
- **Sprintweekends.** Het model voorspelt de hoofdrace na de gewone kwalificatie. De
  sprintkwalificatie op vrijdag en de sprint zelf veranderen daar niets aan; de sprintpunten van
  dat weekend zijn een label.

## Later een eigen server

Houd het rekenwerk in GitHub Actions; een eigen (thuis)server hoeft dan geen Python of
machine learning te draaien. Twee manieren om de uitvoer daar te krijgen:

- **De JSON rechtstreeks lezen.** Bijvoorbeeld
  `https://raw.githubusercontent.com/wiebemermans/F1-Predictor/main/output/predictions/latest.json`
  (idem voor `model_comparison.json` en `prediction_history.json`). Het formaat is een vast
  contract ([usage.md](usage.md#json-formaat-contract-voor-een-website)).
- **Een kopie bijhouden.** Een cronjob met `git pull` in een clone van de repository.

Moet de server toch zelf rekenen, dan volstaat dezelfde CLI in een cronjob:
`uv run f1 fetch && uv run f1 predict --next` tijdens het raceweekend, en na de race
`uv run f1 backtest && uv run f1 evaluate`.

## Problemen oplossen

| Wat je ziet | Oorzaak en oplossing |
|---|---|
| Rode run bij "Data ophalen" of "Voorspellen" met `Jolpica-API niet bereikbaar` | Jolpica is tijdelijk onbereikbaar of beperkt het aantal requests (de client probeert HTTP 429/5xx tot vijf keer). Wacht en start de run opnieuw (`gh run rerun`). |
| `Ronde(s) [21] van 2026 ontbreken in de lokale data` | De uitslag van de vorige race staat nog niet in Jolpica, of de fetch mislukte. Start de run later opnieuw. |
| Geen voorspelling na de kwalificatie | Kijk in de log van de laatste run: "begint pas op" (de kalender wijkt af), "staat nog niet in Jolpica" (nog niet verwerkt). Start `predict.yml` met de hand zodra de uitslag er is. |
| De commitstap faalt bij `git pull --rebase` | Iemand wijzigde tegelijk dezelfde bestanden in `output/`. Start de workflow opnieuw: hij begint dan van de nieuwste stand en commit alleen wat nog ontbreekt. |
| De push faalt met 403 | `GITHUB_TOKEN` mag niet schrijven. Controleer Settings → Actions → General → Workflow permissions en eventuele branch protection op `main`. |
| Geen geplande runs meer | De workflow is uitgeschakeld (60 dagen zonder activiteit) of staat nog niet op `main`. Zet hem aan met `gh workflow enable`. |
| CSV-wijzigingen zonder nieuwe race | Jolpica corrigeerde een eerdere uitslag van het lopende seizoen (straf, diskwalificatie), of het is de eerste run op Linux (zie hierboven). |
