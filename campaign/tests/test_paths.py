"""Paths come from one module, never create anything, and never fall back."""

import pytest

from trojans import paths


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("TROJAN_DATA", str(root))
    return root


def test_env_var_sets_the_data_root(data):
    assert paths.data_root() == data.resolve()


def test_missing_data_root_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TROJAN_DATA", str(tmp_path / "not-mounted"))
    with pytest.raises(paths.DataRootMissing):
        paths.data_root()


def test_path_functions_create_nothing(data):
    paths.raw_dir("sbdb", "2026-09-15")
    paths.catalogue_dir("2026-09-15")
    paths.initial_dir("ic1")
    paths.chunk_dir("pilot_10myr", 3)
    paths.merged_dir("pilot_10myr")
    assert list(data.iterdir()) == []


def test_layout_under_the_data_root(data):
    assert paths.chunk_dir("pilot_10myr", 3) == data.resolve() / "runs" / "pilot_10myr" / "chunks" / "c03"
    assert paths.raw_dir("sbdb", "2026-09-15") == data.resolve() / "raw" / "sbdb" / "2026-09-15"


def test_reports_and_configs_stay_in_the_repo():
    assert paths.report_dir("smoke").parent == paths.CAMPAIGN_ROOT / "reports"
    assert paths.config_path("smoke") == paths.CAMPAIGN_ROOT / "configs" / "smoke.toml"


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", " spaced", ".hidden"])
def test_ids_cannot_escape_their_directory(data, bad):
    with pytest.raises(ValueError):
        paths.run_dir(bad)


def test_existing_output_dir_needs_resume(data):
    target = paths.run_dir("smoke")
    paths.make_output_dir(target)
    with pytest.raises(paths.OutputExists):
        paths.make_output_dir(target)
    assert paths.make_output_dir(target, resume=True) == target


def test_resume_requires_an_existing_dir(data):
    with pytest.raises(FileNotFoundError):
        paths.make_output_dir(paths.run_dir("never-started"), resume=True)
