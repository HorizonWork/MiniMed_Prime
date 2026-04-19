from __future__ import annotations

import scripts.setup_environment as setup_environment


def test_should_skip_pip_install_when_requirement_is_already_satisfied(monkeypatch) -> None:
    monkeypatch.setattr(setup_environment.KaggleEnv, "is_kaggle", staticmethod(lambda: False))

    should_skip, reason = setup_environment._should_skip_pip_install("packaging>=0")

    assert should_skip is True
    assert "already satisfied" in reason.lower()


def test_should_skip_risky_requirement_on_kaggle_safe_mode(monkeypatch) -> None:
    monkeypatch.setattr(setup_environment.KaggleEnv, "is_kaggle", staticmethod(lambda: True))
    monkeypatch.delenv(setup_environment.KAGGLE_PIP_MODE_ENV, raising=False)

    should_skip, reason = setup_environment._should_skip_pip_install("scispacy>=0.5.5,<0.7")

    assert should_skip is True
    assert "risky native requirement" in reason.lower()


def test_should_preserve_out_of_spec_installed_package_on_kaggle_safe_mode(monkeypatch) -> None:
    monkeypatch.setattr(setup_environment.KaggleEnv, "is_kaggle", staticmethod(lambda: True))
    monkeypatch.delenv(setup_environment.KAGGLE_PIP_MODE_ENV, raising=False)

    should_skip, reason = setup_environment._should_skip_pip_install("packaging<0")

    assert should_skip is True
    assert "preserving the base image package" in reason.lower()


def test_force_mode_allows_risky_requirement_install_on_kaggle(monkeypatch) -> None:
    monkeypatch.setattr(setup_environment.KaggleEnv, "is_kaggle", staticmethod(lambda: True))
    monkeypatch.setenv(setup_environment.KAGGLE_PIP_MODE_ENV, setup_environment.KAGGLE_PIP_MODE_FORCE)

    should_skip, reason = setup_environment._should_skip_pip_install("scispacy>=0.5.5,<0.7")

    assert should_skip is False
    assert reason == ""
