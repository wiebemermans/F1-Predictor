"""De GitHub Actions-workflows passen bij de CLI en committen alleen vanaf main.

Een geplande workflow faalt pas op het moment dat hij draait (bv. een zaterdagavond); een
hernoemd commando of een verdwenen optie moet dus al hier opvallen.
"""

import re

import pytest
import typer
import yaml

from f1_predictor import cli
from f1_predictor.paths import ROOT

WORKFLOWS = ROOT / ".github" / "workflows"
AUTOMATION = ["predict.yml", "post-race.yml"]
# `uv run [uv-opties] f1 [globale opties] <commando> <argumenten>`
F1_CALL = re.compile(r"uv run (?:-\S+ )*f1 (?:-\S+ )*(?P<command>\w[\w-]*)(?P<args>[^\n;&|]*)")


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML volgt YAML 1.1 en leest de sleutel `on` als de boolean True.
    return workflow.get("on", workflow.get(True))


def _steps(workflow: dict) -> list[dict]:
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def test_workflows_only_call_existing_cli_commands_and_options():
    commands = typer.main.get_command(cli.app).commands
    calls = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for step in _steps(_workflow(path.name)):
            for match in F1_CALL.finditer(step.get("run", "")):
                calls.append(match["command"])
                command = commands.get(match["command"])
                assert command is not None, f"{path.name}: onbekend commando f1 {match[0]}"
                options = {option for param in command.params for option in param.opts}
                for flag in re.findall(r"(?<!\S)(--?[\w-]+)", match["args"]):
                    assert flag in options, f"{path.name}: onbekende optie {flag} in {match[0]}"

    assert {"fetch", "predict", "backtest", "evaluate"} <= set(calls)


@pytest.mark.parametrize("name", AUTOMATION)
def test_automation_workflows_are_scheduled_and_commit_only_from_main(name):
    workflow = _workflow(name)
    triggers = _triggers(workflow)

    assert {"schedule", "workflow_dispatch"} <= set(triggers)
    assert workflow["permissions"] == {"contents": "write"}
    assert "output-commit" in workflow["concurrency"]["group"]
    pushes = [step for step in _steps(workflow) if "git push" in step.get("run", "")]
    assert len(pushes) == 1
    assert pushes[0]["if"] == "github.ref == 'refs/heads/main'"
    assert "git add output/" in pushes[0]["run"]
