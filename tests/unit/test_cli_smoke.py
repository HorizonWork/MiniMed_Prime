"""Smoke tests for CLI entrypoint."""

from __future__ import annotations

from typer.testing import CliRunner

from minimed_rag.cli.main import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "minimed" in result.output.lower() or "biomedical" in result.output.lower()


def test_subcommand_help_ingest():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_normalize():
    result = runner.invoke(app, ["normalize", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_build_kg():
    result = runner.invoke(app, ["build-kg", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_build_indexes():
    result = runner.invoke(app, ["build-indexes", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_rag():
    result = runner.invoke(app, ["rag", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_generate_paths():
    result = runner.invoke(app, ["generate-paths", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_train_trm():
    result = runner.invoke(app, ["train-trm", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_evaluate():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
