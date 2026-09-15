import numpy as np
import pytest

from trojans import integrate, runs


def test_assign_chunks_is_seeded_balanced_and_subsettable():
    spkids = np.arange(20000001, 20000101)
    a = runs.assign_chunks(spkids, 8, seed=1)
    b = runs.assign_chunks(spkids, 8, seed=1)
    assert a.equals(b)
    assert sorted(a["spkid"]) == sorted(spkids)
    assert a["chunk"].value_counts().max() - a["chunk"].value_counts().min() <= 1
    sub = runs.assign_chunks(spkids, 2, seed=1, subset=10)
    assert len(sub) == 10 and set(sub["spkid"]) <= set(spkids)


def test_settings_from_config():
    config = {"run": {"t_end_yr": 1.0e6}, "integrator": {"check_every_steps": 16, "dt_scale": 0.5},
              "diagnostics": {"window_yr": 1.0e4}, "escape": {"r_max_au": 50.0}}
    s = integrate.settings_from_config(config, dt_yr=0.1186)
    assert s.dt_yr == pytest.approx(0.0593)
    assert s.check_interval_yr == pytest.approx(16 * 0.0593)
    assert s.checks_per_window * s.check_interval_yr == pytest.approx(1.0e4, rel=1e-3)
    assert s.n_windows == 100
    assert s.r_max_au == 50.0
