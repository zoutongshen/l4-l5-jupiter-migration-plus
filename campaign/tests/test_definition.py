"""Stage 7: map particles, labels, snapshot features and catalogue boxes."""

import numpy as np
import pandas as pd

from trojans import coorbital, constants, definition, diagnostics, integrate
from test_integrate import _system


def _jupiter_and_states():
    states, _ = _system({})
    return coorbital.jupiter_at_t0(states), states


def test_map_particles_have_the_sigma_they_were_drawn_with():
    jupiter, states = _jupiter_and_states()
    el = coorbital.draw_elements(coorbital.MapGrid(n=200, seed=1), jupiter)
    full = coorbital.states_with_particles(states, el)
    trojans = full[full["body"] == "trojan"]
    assert (trojans["spkid"].to_numpy() == el["spkid"].to_numpy()).all()
    assert len(full) == len(states) + 200
    back = diagnostics.elements(trojans[["x", "y", "z"]].to_numpy(), trojans[["vx", "vy", "vz"]].to_numpy(), constants.G)
    sigma = diagnostics.resonant_angle(back["lam"], jupiter["lam"])
    assert np.allclose(diagnostics.wrap180(sigma - el["sigma"]), 0.0, atol=1e-8)
    assert np.allclose(back["a"], el["a"], rtol=1e-10)


def test_features_recover_sigma_and_wrap_angles():
    jupiter, _ = _jupiter_and_states()
    el = coorbital.draw_elements(coorbital.MapGrid(n=100, seed=2), jupiter)
    f = definition.features(el, jupiter)
    assert np.allclose(diagnostics.wrap180(f["sigma"] - el["sigma"]), 0.0, atol=1e-9)
    assert np.allclose(f["da"], el["a"] - jupiter["a"])
    assert np.allclose(f["sin_sigma"] ** 2 + f["cos_sigma"] ** 2, 1.0)


def test_label_needs_every_window_in_one_swarm():
    L4, L5, HS = integrate.TADPOLE_L4, integrate.TADPOLE_L5, integrate.HORSESHOE
    classes = np.array([[L4, L5, L4, HS],
                        [L4, L5, L5, L4],
                        [L4, L5, L4, L4]])
    assert list(definition.label_from_classes(classes, 1)) == ["L4", "L5", "L4", "none"]
    assert list(definition.label_from_classes(classes, 3)) == ["L4", "L5", "none", "none"]


def test_box_label_takes_the_swarm_from_sigma():
    f = pd.DataFrame({"a": [5.2, 5.2, 5.2, 6.0], "e": [0.1, 0.1, 0.1, 0.1], "inc": [10.0] * 4,
                      "sigma": [60.0, -60.0, 5.0, 60.0]})
    assert list(definition.box_label(f, definition.box_jpl_tjn)) == ["L4", "L5", "L4", "none"]
    assert list(definition.box_label(f, definition.box_mpc_inferred)) == ["L4", "L5", "none", "none"]
