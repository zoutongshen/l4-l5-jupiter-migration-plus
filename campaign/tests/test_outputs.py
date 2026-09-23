"""Manifest and provenance writers: a file is written whole or not at all."""

import json

import pandas as pd
import pytest

from trojans import outputs


def test_write_json_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    outputs.write_json(path, {"b": 1, "a": [2, 3]})
    assert json.loads(path.read_text()) == {"b": 1, "a": [2, 3]}
    assert not list(tmp_path.glob("*.tmp"))


def test_write_parquet_round_trips(tmp_path):
    path = tmp_path / "ladder.parquet"
    frame = pd.DataFrame({"spkid": [1, 2], "m_jup": [0.0, 1.0]})
    outputs.write_parquet(path, frame)
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_write_leaves_the_previous_table_readable(tmp_path, monkeypatch):
    """The reason this helper exists: a long run checkpoints into one file.

    A crash or an unplugged drive part-way through must cost the last few
    results, not every result so far.
    """
    path = tmp_path / "ladder.parquet"
    first = pd.DataFrame({"spkid": [1, 2], "m_jup": [0.0, 0.0]})
    outputs.write_parquet(path, first)

    def boom(self, *args, **kwargs):
        raise OSError("device not configured")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        outputs.write_parquet(path, pd.DataFrame({"spkid": [3], "m_jup": [1.0]}))

    monkeypatch.undo()
    pd.testing.assert_frame_equal(pd.read_parquet(path), first)
