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


def test_rag_has_query_subcommand():
    result = runner.invoke(app, ["rag", "--help"])
    assert result.exit_code == 0
    assert "query" in result.output


def test_rag_query_exposes_use_kg_flag():
    result = runner.invoke(app, ["rag", "query", "--help"])
    assert result.exit_code == 0
    assert "--use-kg" in result.output


def test_ingest_has_primekg_and_umls_subcommands():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "primekg" in result.output
    assert "umls" in result.output


def test_ingest_primekg_help_shows_options():
    result = runner.invoke(app, ["ingest", "primekg", "--help"])
    assert result.exit_code == 0
    assert "--release" in result.output
    assert "--force" in result.output
    assert "--limit" in result.output


def test_ingest_umls_help_shows_options():
    result = runner.invoke(app, ["ingest", "umls", "--help"])
    assert result.exit_code == 0
    assert "--release" in result.output
    assert "--skip-postgres" in result.output
    assert "--limit-mrconso" in result.output


def test_build_kg_has_apply_schema_and_crosswalk():
    result = runner.invoke(app, ["build-kg", "--help"])
    assert result.exit_code == 0
    assert "apply-schema" in result.output
    assert "crosswalk" in result.output


def test_subcommand_help_generate_paths():
    result = runner.invoke(app, ["generate-paths", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_train_trm():
    result = runner.invoke(app, ["train-trm", "--help"])
    assert result.exit_code == 0


def test_subcommand_help_evaluate():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
