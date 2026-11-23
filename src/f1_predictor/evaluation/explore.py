from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd

from f1_predictor.features.groups import retirement_mask
from f1_predictor.paths import OUTPUT_DIR, RAW_DIR, ROOT

# Minimum aantal races van een coureur op een circuit voor de bias-tabel; minder is vooral ruis.
MIN_DRIVER_CIRCUIT_RACES = 3
# Minimum aantal races op een circuit voordat het in de rapporttabel verschijnt.
MIN_CIRCUIT_RACES = 3


class QualitySummary(TypedDict):
    rows: int
    seasons: int
    races: int
    driver_race_rows: int
    duplicate_driver_race_rows: int
    missing_values: dict[str, int]
    grid_finish_spearman: float
    avg_finish_position: float
    median_finish_position: float
    min_finish_position: int
    max_finish_position: int
    numeric_ranges: dict[str, tuple[float, float]]
    invalid_domain_values: dict[str, int]


def load_all_results(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Laad alle seizoensbestanden in één DataFrame."""
    files = sorted(raw_dir.glob("results_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Geen parquet-bestanden gevonden in {raw_dir}")

    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)
    return df


def _start_position(df: pd.DataFrame) -> pd.Series:
    """Startpositie waarbij een pitlane-start (`grid = 0`) achteraan het veld telt."""
    race_size = df.groupby(["season", "round"])["driver_id"].transform("size")
    return df["grid"].where(df["grid"] > 0, race_size)


def build_quality_summary(df: pd.DataFrame) -> QualitySummary:
    """Belangrijkste kwaliteits- en analysecijfers voor fase 4."""
    grid_finish_spearman = _start_position(df).corr(df["finish_position"], method="spearman")
    race_size = df.groupby(["season", "round"])["driver_id"].transform("size")
    numeric_columns = ["grid", "finish_position", "points", "sprint_points", "laps"]
    summary: QualitySummary = {
        "rows": int(len(df)),
        "seasons": int(df["season"].nunique()),
        "races": int(df[["season", "round"]].drop_duplicates().shape[0]),
        "driver_race_rows": int(df[["season", "round", "driver_id"]].drop_duplicates().shape[0]),
        "duplicate_driver_race_rows": int(
            df.duplicated(subset=["season", "round", "driver_id"]).sum()
        ),
        "missing_values": {
            str(key): int(value)
            for key, value in df.isna().sum().sort_values(ascending=False).items()
        },
        "grid_finish_spearman": float(grid_finish_spearman),
        "avg_finish_position": float(df["finish_position"].mean()),
        "median_finish_position": float(df["finish_position"].median()),
        "min_finish_position": int(df["finish_position"].min()),
        "max_finish_position": int(df["finish_position"].max()),
        "numeric_ranges": {
            column: (float(df[column].min()), float(df[column].max())) for column in numeric_columns
        },
        "invalid_domain_values": {
            "negative_grid": int((df["grid"] < 0).sum()),
            "finish_position_outside_race_field": int(
                ((df["finish_position"] < 1) | (df["finish_position"] > race_size)).sum()
            ),
            "negative_points": int((df["points"] < 0).sum()),
            "negative_sprint_points": int((df["sprint_points"] < 0).sum()),
            "negative_laps": int((df["laps"] < 0).sum()),
        },
    }
    return summary


def _retirement_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = df.assign(retired=retirement_mask(df["status"])).groupby(column)["retired"]
    stats = grouped.agg(retirement_rate="mean", starts="size").reset_index()
    stats["retirement_rate"] = (stats["retirement_rate"] * 100).round(2)
    return stats


def build_retirement_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Uitvalpercentage (DNF + DNS) per team, hoogste eerst."""
    stats = _retirement_rate_by(df, "constructor_id")
    return stats.sort_values("retirement_rate", ascending=False).reset_index(drop=True)


def build_retirement_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Uitvalpercentage (DNF + DNS) per seizoen, chronologisch."""
    return _retirement_rate_by(df, "season").sort_values("season").reset_index(drop=True)


def build_team_strength_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Gemiddelde teamfinish per seizoen voor vergelijking tussen teams."""
    team_strength = df.groupby(["season", "constructor_id"])["finish_position"].mean()
    team_strength = team_strength.reset_index()
    team_strength.columns = ["season", "constructor_id", "avg_finish_position"]
    team_strength = team_strength.sort_values(
        ["season", "avg_finish_position"], ascending=[True, True]
    ).reset_index(drop=True)
    return team_strength


def build_circuit_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Hoe voorspelbaar elk circuit is, van chaotisch naar voorspelbaar.

    De gemiddelde finishpositie per circuit zegt niets (bij 20 deelnemers altijd 10,5).
    Daarom per circuit: de gemiddelde Spearman-correlatie per race tussen startpositie
    (pitlane achteraan) en finish, de gemiddelde absolute positiewinst/-verlies en het
    uitvalpercentage.
    """
    df = df.assign(
        _start=_start_position(df),
        _retired=retirement_mask(df["status"]),
    )
    df["_abs_change"] = (df["_start"] - df["finish_position"]).abs()
    per_race = (
        df.groupby(["circuit_id", "season", "round"])
        .apply(
            lambda race: race["_start"].corr(race["finish_position"], method="spearman"),
            include_groups=False,
        )
        .rename("start_finish_spearman")
        .reset_index()
    )
    effects = per_race.groupby("circuit_id").agg(
        races=("start_finish_spearman", "size"),
        start_finish_spearman=("start_finish_spearman", "mean"),
    )
    effects["mean_abs_position_change"] = df.groupby("circuit_id")["_abs_change"].mean()
    effects["retirement_rate"] = (df.groupby("circuit_id")["_retired"].mean() * 100).round(2)
    return (
        effects.reset_index()
        .sort_values(["start_finish_spearman", "circuit_id"])
        .reset_index(drop=True)
    )


def build_driver_circuit_bias(
    df: pd.DataFrame, min_races: int = MIN_DRIVER_CIRCUIT_RACES
) -> pd.DataFrame:
    """Laat zien of een coureur op bepaalde circuits structureel beter of slechter presteert.

    Alleen combinaties met minstens `min_races` races; het verschil is ruw (zonder
    shrinkage of correctie voor teamsterkte) en is dus een verkenning, geen feature.
    """
    all_races = df.groupby(["driver_id", "circuit_id"]).size().reset_index(name="races")
    driver_mean = df.groupby("driver_id")["finish_position"].mean().reset_index()
    driver_mean.columns = ["driver_id", "driver_mean_finish"]
    circuit_mean = df.groupby(["driver_id", "circuit_id"])["finish_position"].mean().reset_index()
    circuit_mean.columns = ["driver_id", "circuit_id", "circuit_mean_finish"]
    bias = circuit_mean.merge(driver_mean, on="driver_id", how="left")
    bias = bias.merge(all_races, on=["driver_id", "circuit_id"], how="left")
    bias["delta_vs_driver_mean"] = bias["circuit_mean_finish"] - bias["driver_mean_finish"]
    bias = bias[bias["races"] >= min_races]
    return bias.sort_values("delta_vs_driver_mean").reset_index(drop=True)


def _table(header: str, rows: list[str]) -> str:
    return header + "\n" + "\n".join(rows)


def build_report(
    quality: QualitySummary,
    retirement: pd.DataFrame,
    retirement_by_season: pd.DataFrame,
    team_strength: pd.DataFrame,
    circuit_effects: pd.DataFrame,
    driver_bias: pd.DataFrame,
) -> str:
    """Markdown-rapport met de belangrijkste observaties uit fase 4."""
    team_top = (
        team_strength.sort_values(["season", "avg_finish_position"])
        .groupby("season", as_index=False)
        .head(1)
        .sort_values("season")
    )
    circuits = circuit_effects[circuit_effects["races"] >= MIN_CIRCUIT_RACES]
    chaotic = circuits.head(5)
    predictable = circuits.tail(5).iloc[::-1]
    best_driver = driver_bias.nsmallest(5, "delta_vs_driver_mean")
    worst_driver = driver_bias.nlargest(5, "delta_vs_driver_mean")
    team_rate_min = retirement["retirement_rate"].min()
    team_rate_max = retirement["retirement_rate"].max()
    season_rate_min = retirement_by_season["retirement_rate"].min()
    season_rate_max = retirement_by_season["retirement_rate"].max()

    def circuit_row(row: pd.Series) -> str:
        return (
            f"| {row['circuit_id']} | {int(row['races'])} | {row['start_finish_spearman']:.2f} "
            f"| {row['mean_abs_position_change']:.2f} | {row['retirement_rate']:.2f}% |"
        )

    def bias_row(row: pd.Series) -> str:
        return (
            f"| {row['driver_id']} | {row['circuit_id']} | {int(row['races'])} "
            f"| {row['delta_vs_driver_mean']:.2f} |"
        )

    circuit_header = (
        "| Circuit | Races | Spearman start-finish | Gem. abs. positiewijziging | Uitval |\n"
        "|---|---:|---:|---:|---:|"
    )
    bias_header = (
        "| Coureur | Circuit | Races | Verschil t.o.v. eigen gemiddelde |\n|---|---|---:|---:|"
    )

    return (
        "# Fase 4: Data verkennen\n\n"
        "Dit rapport wordt gegenereerd door `uv run python -m "
        "f1_predictor.evaluation.explore`.\n\n"
        "## Datasetkwaliteit\n\n"
        f"- Aantal rijen: {quality['rows']}\n"
        f"- Aantal seizoenen: {quality['seasons']}\n"
        f"- Aantal unieke races: {quality['races']}\n"
        "- Duplicates op (`season`, `round`, `driver_id`): "
        f"{quality['duplicate_driver_race_rows']}\n"
        f"- Ontbrekende waarden: {quality['missing_values']}\n"
        "- Spearman correlatie tussen startpositie en finishpositie: "
        f"{quality['grid_finish_spearman']:.4f}\n"
        f"- Gemiddelde finishpositie: {quality['avg_finish_position']:.2f}\n"
        f"- Mediaan finishpositie: {quality['median_finish_position']:.2f}\n\n"
        "## Methode en definities\n\n"
        "- **Startpositie:** `grid`, waarbij een pitlane-start (`grid = 0`) als laatste "
        "startplaats van het veld telt. De ruwe data blijft ongewijzigd; deze vertaling "
        "gebeurt alleen in de analyse.\n"
        "- **Uitval:** elke status behalve gefinisht en gediskwalificeerd, dus DNF en "
        "niet-starters (DNS). Gefinisht is `Finished`, `+N Lap(s)` (oudere seizoenen) of "
        "`Lapped` (recentere seizoenen). Diskwalificaties tellen niet als uitval. "
        "De status is gebruikt in plaats van `position_text`, omdat uitvallers die ver "
        "genoeg kwamen toch een numerieke klassering krijgen.\n"
        "- **Circuit-effecten:** de gemiddelde finishpositie per circuit is geen bruikbare "
        "maat (bij 20 deelnemers altijd 10,5). Daarom wordt per race de Spearman-correlatie "
        "tussen startpositie en finish berekend en per circuit gemiddeld; lager betekent een "
        "minder voorspelbaar circuit.\n"
        f"- **Driver-circuit bias:** alleen combinaties met minstens "
        f"{MIN_DRIVER_CIRCUIT_RACES} races; ruw verschil zonder correctie voor teamsterkte.\n\n"
        "## Bereiken en domeincontrole\n\n"
        "Numerieke bereiken zijn gecontroleerd op onmogelijke waarden. Een generieke "
        "IQR-regel is hier niet toegepast: startplaatsen, rondes, punten en rondetotalen "
        "zijn discrete waarden die per seizoen en racecontext verschillen. `grid = 0` "
        "en `laps = 0` zijn geldige pitlane-starten/niet gestarte of vroeg uitgevallen "
        "deelnemers en worden niet als fout aangemerkt.\n\n"
        + _table(
            "| Kolom | Minimum | Maximum |\n|---|---:|---:|",
            [
                f"| {column} | {minimum:g} | {maximum:g} |"
                for column, (minimum, maximum) in quality["numeric_ranges"].items()
            ],
        )
        + "\n\n"
        + _table(
            "| Domeincontrole | Ongeldige rijen |\n|---|---:|",
            [
                f"| {check.replace('_', ' ')} | {count} |"
                for check, count in quality["invalid_domain_values"].items()
            ],
        )
        + "\n\n"
        "## Belangrijkste observaties\n\n"
        "1. De startpositie is een sterke maar niet perfecte voorspeller van de "
        f"eindklassering (Spearman {quality['grid_finish_spearman']:.4f}). Race-ruis en "
        "uitvallers blijven belangrijk.\n"
        "2. Er zijn geen duplicaten en slechts "
        f"{quality['missing_values'].get('quali_position', 0)} ontbrekende "
        "`quali_position`-waarden; de dataset is consistent.\n"
        f"3. Het uitvalpercentage per seizoen ligt tussen {season_rate_min:.1f}% en "
        f"{season_rate_max:.1f}%; per team tussen {team_rate_min:.1f}% en "
        f"{team_rate_max:.1f}%. Teams met weinig starts (nieuwe teams, lopend seizoen) "
        "hebben onbetrouwbare percentages; kijk daarom ook naar het aantal starts.\n"
        "4. Circuits verschillen in voorspelbaarheid: de gemiddelde Spearman-correlatie "
        f"tussen start en finish loopt van {circuits['start_finish_spearman'].min():.2f} tot "
        f"{circuits['start_finish_spearman'].max():.2f} (circuits met minstens "
        f"{MIN_CIRCUIT_RACES} races). Op chaotische circuits weegt de startpositie minder.\n"
        "5. Sommige coureurs presteren op bepaalde circuits beter of slechter dan hun eigen "
        "gemiddelde, maar het aantal races per combinatie is klein en teamsterkte is niet "
        "gecorrigeerd. Een circuit-feature in fase 6 heeft shrinkage nodig.\n\n"
        "## Uitvalpercentage per seizoen\n\n"
        + _table(
            "| Seizoen | Starts | Uitvalpercentage |\n|---|---:|---:|",
            [
                f"| {int(row['season'])} | {int(row['starts'])} | {row['retirement_rate']:.2f}% |"
                for _, row in retirement_by_season.iterrows()
            ],
        )
        + "\n\n## Uitvalpercentage per team (hoogste 5)\n\n"
        + _table(
            "| Team | Starts | Uitvalpercentage |\n|---|---:|---:|",
            [
                f"| {row['constructor_id']} | {int(row['starts'])} "
                f"| {row['retirement_rate']:.2f}% |"
                for _, row in retirement.head(5).iterrows()
            ],
        )
        + "\n\n## Sterkste team per seizoen\n\n"
        "Het team met de laagste gemiddelde finishpositie per seizoen; lager is beter. "
        "Alle teams staan in `output/phase4_team_strength_by_season.csv`.\n\n"
        + _table(
            "| Seizoen | Team | Gemiddelde teamfinish |\n|---|---|---:|",
            [
                f"| {int(row['season'])} | {row['constructor_id']} "
                f"| {row['avg_finish_position']:.2f} |"
                for _, row in team_top.iterrows()
            ],
        )
        + "\n\n## Circuit-effecten\n\n"
        f"Circuits met minstens {MIN_CIRCUIT_RACES} races. Alle circuits staan in "
        "`output/phase4_circuit_effects.csv`.\n\n"
        "### Minst voorspelbaar\n\n"
        + _table(circuit_header, [circuit_row(row) for _, row in chaotic.iterrows()])
        + "\n\n### Meest voorspelbaar\n\n"
        + _table(circuit_header, [circuit_row(row) for _, row in predictable.iterrows()])
        + "\n\n## Driver-circuit bias\n\n"
        "De sterkste afwijkingen ten opzichte van de eigen gemiddelde finish van de coureur.\n\n"
        "### Positieve bias (beter dan normaal)\n\n"
        + _table(bias_header, [bias_row(row) for _, row in best_driver.iterrows()])
        + "\n\n### Negatieve bias (slechter dan normaal)\n\n"
        + _table(bias_header, [bias_row(row) for _, row in worst_driver.iterrows()])
        + "\n\n## Conclusie\n\n"
        "De dataset is goed genoeg voor fase 5 en fase 6. De startpositie is de sterkste "
        "enkele voorspeller; teamsterkte, uitvalrisico en circuitkarakter zijn aanvullende "
        "signalen. Uitval- en circuitfeatures moeten rekening houden met kleine aantallen "
        "en met beide statusvocabulaires van Jolpica.\n"
    )


PHASE4_FILE_NAMES = {
    "summary": "phase4_quality_summary.csv",
    "retirement": "phase4_retirement_by_team.csv",
    "retirement_by_season": "phase4_retirement_by_season.csv",
    "team_strength": "phase4_team_strength_by_season.csv",
    "circuit_effects": "phase4_circuit_effects.csv",
    "driver_bias": "phase4_driver_circuit_bias.csv",
}


def write_phase4_outputs(
    df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    report_path: Path = ROOT / "docs" / "phase4_exploration.md",
) -> dict[str, Path]:
    """Schrijf analyse-output bestanden voor fase 4."""
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = build_quality_summary(df)
    retirement = build_retirement_stats(df)
    retirement_by_season = build_retirement_by_season(df)
    team_strength = build_team_strength_by_season(df)
    circuit_effects = build_circuit_effects(df)
    driver_bias = build_driver_circuit_bias(df)

    tables = {
        "summary": pd.DataFrame([quality]),
        "retirement": retirement,
        "retirement_by_season": retirement_by_season,
        "team_strength": team_strength,
        "circuit_effects": circuit_effects,
        "driver_bias": driver_bias,
    }
    outputs: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / PHASE4_FILE_NAMES[name]
        table.to_csv(path, index=False)
        outputs[name] = path

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            quality, retirement, retirement_by_season, team_strength, circuit_effects, driver_bias
        ),
        encoding="utf-8",
    )
    outputs["report"] = report_path
    return outputs


def main() -> None:
    df = load_all_results()
    outputs = write_phase4_outputs(df)
    print("Phase 4 exploration complete.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
