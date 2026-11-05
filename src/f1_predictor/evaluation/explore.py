from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd

from f1_predictor.paths import OUTPUT_DIR, RAW_DIR


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


def _retirement_mask(status: pd.Series) -> pd.Series:
    status = status.fillna("").astype(str).str.lower()
    finished_like = status.eq("finished") | status.str.fullmatch(r"\+\d+\s*lap(s)?", na=False)
    return ~finished_like


def build_quality_summary(df: pd.DataFrame) -> QualitySummary:
    """Belangrijkste kwaliteits- en analysecijfers voor fase 4."""
    grid_finish_spearman = df["grid"].corr(df["finish_position"], method="spearman")
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


def build_retirement_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Retirement-percentage per team."""
    df = df.copy()
    df["retired"] = _retirement_mask(df["status"])
    stats = df.groupby("constructor_id")["retired"].mean().reset_index()
    stats.columns = ["constructor_id", "retirement_rate"]
    stats = stats.sort_values("retirement_rate", ascending=False).reset_index(drop=True)
    stats["retirement_rate"] = (stats["retirement_rate"] * 100).round(2)
    return stats


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
    """Circuitgemiddelden, gerangschikt van snelste naar langzaamste gemiddelde finish."""
    series = df.groupby("circuit_id")["finish_position"].mean().reset_index()
    series.columns = ["circuit_id", "avg_finish_position"]
    series = series.sort_values("avg_finish_position").reset_index(drop=True)
    series["races_at_circuit"] = (
        df.groupby("circuit_id").size().reindex(series["circuit_id"]).to_numpy()
    )
    return series


def build_driver_circuit_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Laat zien of een coureur op bepaalde circuits structureel beter of slechter presteert."""
    all_races = df.groupby(["driver_id", "circuit_id"]).size().reset_index(name="races")
    driver_mean = df.groupby("driver_id")["finish_position"].mean().reset_index()
    driver_mean.columns = ["driver_id", "driver_mean_finish"]
    circuit_mean = df.groupby(["driver_id", "circuit_id"])["finish_position"].mean().reset_index()
    circuit_mean.columns = ["driver_id", "circuit_id", "circuit_mean_finish"]
    bias = circuit_mean.merge(driver_mean, on="driver_id", how="left")
    bias = bias.merge(all_races, on=["driver_id", "circuit_id"], how="left")
    bias["delta_vs_driver_mean"] = bias["circuit_mean_finish"] - bias["driver_mean_finish"]
    bias = bias[bias["races"] >= 2].sort_values("delta_vs_driver_mean").reset_index(drop=True)
    return bias


def build_report(
    quality: QualitySummary,
    retirement: pd.DataFrame,
    team_strength: pd.DataFrame,
    circuit_effects: pd.DataFrame,
    driver_bias: pd.DataFrame,
) -> str:
    """Markdown-rapport met de belangrijkste observaties uit fase 4."""
    top_retirement = retirement.head(5).copy()
    top_retirement["retirement_rate"] = top_retirement["retirement_rate"].map(
        lambda value: f"{value:.2f}%"
    )
    team_top = (
        team_strength.sort_values(["season", "avg_finish_position"])
        .groupby("season", as_index=False)
        .head(1)
        .sort_values("season")
        .head(3)
    )
    best_circuit = circuit_effects.head(5)[["circuit_id", "avg_finish_position"]].copy()
    best_driver = driver_bias.nsmallest(5, "delta_vs_driver_mean")[
        ["driver_id", "circuit_id", "delta_vs_driver_mean"]
    ].copy()
    worst_driver = driver_bias.nlargest(5, "delta_vs_driver_mean")[
        ["driver_id", "circuit_id", "delta_vs_driver_mean"]
    ].copy()

    retirement_rows = "\n".join(
        f"| {row['constructor_id']} | {row['retirement_rate']} |"
        for _, row in top_retirement.iterrows()
    )
    team_rows = "\n".join(
        f"| {int(row['season'])} | {row['constructor_id']} | {row['avg_finish_position']:.2f} |"
        for _, row in team_top.iterrows()
    )
    circuit_rows = "\n".join(
        f"| {row['circuit_id']} | {row['avg_finish_position']:.2f} |"
        for _, row in best_circuit.iterrows()
    )
    positive_bias_rows = "\n".join(
        f"| {row['driver_id']} | {row['circuit_id']} | {row['delta_vs_driver_mean']:.2f} |"
        for _, row in best_driver.iterrows()
    )
    negative_bias_rows = "\n".join(
        f"| {row['driver_id']} | {row['circuit_id']} | {row['delta_vs_driver_mean']:.2f} |"
        for _, row in worst_driver.iterrows()
    )
    range_rows = "\n".join(
        f"| {column} | {minimum:g} | {maximum:g} |"
        for column, (minimum, maximum) in quality["numeric_ranges"].items()
    )
    invalid_rows = "\n".join(
        f"| {check.replace('_', ' ')} | {count} |"
        for check, count in quality["invalid_domain_values"].items()
    )

    return (
        "# Fase 4: Data verkennen\n\n"
        "## Datasetkwaliteit\n\n"
        f"- Aantal rijen: {quality['rows']}\n"
        f"- Aantal seizoenen: {quality['seasons']}\n"
        f"- Aantal unieke races: {quality['races']}\n"
        "- Duplicates op (`season`, `round`, `driver_id`): "
        f"{quality['duplicate_driver_race_rows']}\n"
        f"- Ontbrekende waarden: {quality['missing_values']}\n"
        "- Spearman correlatie tussen grid en finishpositie: "
        f"{quality['grid_finish_spearman']:.4f}\n"
        f"- Gemiddelde finishpositie: {quality['avg_finish_position']:.2f}\n"
        f"- Mediaan finishpositie: {quality['median_finish_position']:.2f}\n\n"
        "## Bereiken en domeincontrole\n\n"
        "Numerieke bereiken zijn gecontroleerd op onmogelijke waarden. Een generieke "
        "IQR-regel is hier niet toegepast: startplaatsen, rondes, punten en rondetotalen "
        "zijn discrete waarden die per seizoen en racecontext verschillen. `grid = 0` "
        "en `laps = 0` zijn geldige pitlane-starten/niet gestarte of vroeg uitgevallen "
        "deelnemers en worden niet als fout aangemerkt.\n\n"
        "| Kolom | Minimum | Maximum |\n"
        "|---|---:|---:|\n"
        f"{range_rows}\n\n"
        "| Domeincontrole | Ongeldige rijen |\n"
        "|---|---:|\n"
        f"{invalid_rows}\n\n"
        "## Belangrijkste observaties\n\n"
        "1. De gridpositie is een sterke maar niet perfecte voorspeller van de "
        "eindklassering. De Spearman-correlatie van "
        f"{quality['grid_finish_spearman']:.4f} laat zien dat startpositie veel "
        "verklaart, maar dat race-ruis en uitvallers nog steeds belangrijk "
        "zijn.\n"
        "2. Er zijn in de ruwe dataset vrijwel geen duplicaten en slechts een "
        "klein aantal ontbrekende `quali_position`-waarden "
        f"({quality['missing_values'].get('quali_position', 0)}). Dat betekent "
        "dat het datamodel consistent is en geen grote data-quality-issues "
        "vertoont.\n"
        "3. De uitvalpercentages verschillen sterk tussen teams, wat een "
        "aanwijzing is dat team- en auto-kwaliteit een duidelijke invloed hebben "
        "op de kans op een DNF.\n"
        "4. Er zijn duidelijke circuitverschillen in gemiddelde finishpositie, "
        "maar de range is beperkt. Dat suggereert dat circuitkarakteristieken "
        "relevant zijn, maar niet zo dominant dat ze alle andere signalen "
        "volledig overschaduwen.\n"
        "5. Verschillende coureurs presteren structureel beter of slechter op "
        "bepaalde circuits. Dat is interessant voor de feature-engineering in "
        "fase 6, maar we moeten het met voldoende data-per-circuit "
        "controleren om overfitting te voorkomen.\n\n"
        "## Uitvalpercentage per team\n\n"
        "| Team | Uitvalpercentage |\n"
        "|---|---:|\n"
        f"{retirement_rows}\n\n"
        "## Teamsterkte per seizoen\n\n"
        "De tabel hieronder toont de gemiddelde bijdrage van elk team per jaar. "
        "Lagere gemiddelden betekenen gemiddeld beter presteren.\n\n"
        "| Seizoen | Team | Gemiddelde teamfinish |\n"
        "|---|---|---:|\n"
        f"{team_rows}\n\n"
        "## Circuit-effecten\n\n"
        "| Circuit | Gemiddelde finishpositie |\n"
        "|---|---:|\n"
        f"{circuit_rows}\n\n"
        "## Driver-circuit bias\n\n"
        "De onderstaande selectie toont de sterkste positieve en negatieve "
        "afwijkingen ten opzichte van de eigen gemiddelde finish van de coureur."
        "\n\n"
        "### Positieve bias (beter dan normaal)\n\n"
        "| Coureur | Circuit | Verschil ten opzichte van eigen gemiddelde |\n"
        "|---|---|---:|\n"
        f"{positive_bias_rows}\n\n"
        "### Negatieve bias (slechter dan normaal)\n\n"
        "| Coureur | Circuit | Verschil ten opzichte van eigen gemiddelde |\n"
        "|---|---|---:|\n"
        f"{negative_bias_rows}\n\n"
        "## Conclusie\n\n"
        "Fase 4 laat zien dat de dataset goed genoeg is om door te gaan naar "
        "fase 5 en fase 6. De startpositie is een sterke baseline, het "
        "team-/circuit-signaal is aanwezig, en er is voldoende data voor "
        "feature engineering zonder leakage. De volgende stap is het "
        "definiëren van baselines en backtest-metrieken, zodat het model "
        "eerlijk kan worden vergeleken met eenvoudige, tijdgeordende "
        "benchmarks.\n"
    )


def write_phase4_outputs(df: pd.DataFrame) -> dict[str, Path]:
    """Schrijf analyse-output bestanden voor fase 4."""
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = build_quality_summary(df)
    retirement = build_retirement_stats(df)
    team_strength = build_team_strength_by_season(df)
    circuit_effects = build_circuit_effects(df)
    driver_bias = build_driver_circuit_bias(df)

    summary_path = output_dir / "phase4_quality_summary.csv"
    retirement_path = output_dir / "phase4_retirement_by_team.csv"
    team_path = output_dir / "phase4_team_strength_by_season.csv"
    circuit_path = output_dir / "phase4_circuit_effects.csv"
    driver_bias_path = output_dir / "phase4_driver_circuit_bias.csv"

    pd.DataFrame([quality]).to_csv(summary_path, index=False)
    retirement.to_csv(retirement_path, index=False)
    team_strength.to_csv(team_path, index=False)
    circuit_effects.to_csv(circuit_path, index=False)
    driver_bias.to_csv(driver_bias_path, index=False)

    report_path = Path("docs") / "phase4_exploration.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(quality, retirement, team_strength, circuit_effects, driver_bias),
        encoding="utf-8",
    )

    return {
        "summary": summary_path,
        "retirement": retirement_path,
        "team_strength": team_path,
        "circuit_effects": circuit_path,
        "driver_bias": driver_bias_path,
        "report": report_path,
    }


def main() -> None:
    df = load_all_results()
    outputs = write_phase4_outputs(df)
    print("Phase 4 exploration complete.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
