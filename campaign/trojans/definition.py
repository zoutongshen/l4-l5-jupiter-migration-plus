"""Labels, snapshot features and catalogue boxes for the Trojan-definition study (stage 7).

- **Label:** from the integration. An orbit is an L4 (L5) Trojan if every one of its
  first ``k`` windows is a tadpole of L4 (L5); anything else, including removal, is
  "none". ``k`` is the label's timescale, a choice stated with every result, never
  something the model sees.
- **Features:** from the t0 snapshot only. ``RAW`` is the six osculating elements as
  catalogues publish them. ``PHYSICAL`` replaces the angles by their values relative
  to Jupiter, wrapped through sin and cos, so 359 and 1 deg are neighbours.
- **Boxes:** the catalogue rules of ``docs/TROJAN_DEFINITION.md`` §1, applied to the
  same snapshot, so they can be scored on the same labels.
"""

import h5py
import numpy as np
import pandas as pd

from trojans import catalogue, diagnostics, integrate

LABELS = ("none", "L4", "L5")
RAW = ["a", "e", "inc", "Omega", "omega", "M"]
PHYSICAL = ["da", "e", "inc", "sin_sigma", "cos_sigma", "sin_dvarpi", "cos_dvarpi", "sin_dOmega", "cos_dOmega"]


# --- labels ----------------------------------------------------------------

def read_classes(windows_files) -> tuple[np.ndarray, np.ndarray]:
    """spkids and window classes (n_windows, n) from one or more windows.h5 files.

    Files must hold the same number of windows; chunks of one run do.
    """
    spkids, classes = [], []
    for path in windows_files:
        with h5py.File(path) as f:
            spkids.append(f["spkid"][:])
            classes.append(f["class"][:])
    n_windows = {c.shape[0] for c in classes}
    if len(n_windows) != 1:
        raise ValueError(f"window counts differ between files: {sorted(n_windows)}")
    return np.concatenate(spkids), np.concatenate(classes, axis=1)


def label_from_classes(classes: np.ndarray, k: int) -> np.ndarray:
    """"L4", "L5" or "none" per object from its first ``k`` windows."""
    if classes.shape[0] < k:
        raise ValueError(f"label needs {k} windows, run has {classes.shape[0]}")
    first = classes[:k]
    out = np.full(classes.shape[1], "none", dtype=object)
    out[(first == integrate.TADPOLE_L4).all(axis=0)] = "L4"
    out[(first == integrate.TADPOLE_L5).all(axis=0)] = "L5"
    return out


# --- features --------------------------------------------------------------

def features(elements: pd.DataFrame, jupiter: dict) -> pd.DataFrame:
    """Snapshot features of objects with heliocentric a, e, inc, Omega, omega, M (deg)."""
    el = elements
    varpi = np.mod(el["Omega"] + el["omega"], 360.0)
    lam = np.mod(varpi + el["M"], 360.0)
    sigma = diagnostics.resonant_angle(lam, jupiter["lam"])
    dvarpi = np.radians(diagnostics.wrap180(varpi - jupiter["varpi"]))
    dOmega = np.radians(diagnostics.wrap180(el["Omega"] - jupiter["Omega"]))
    out = el[RAW].copy()
    out["sigma"] = sigma
    out["da"] = el["a"] - jupiter["a"]
    out["sin_sigma"], out["cos_sigma"] = np.sin(np.radians(sigma)), np.cos(np.radians(sigma))
    out["sin_dvarpi"], out["cos_dvarpi"] = np.sin(dvarpi), np.cos(dvarpi)
    out["sin_dOmega"], out["cos_dOmega"] = np.sin(dOmega), np.cos(dOmega)
    return out


# --- catalogue boxes ---------------------------------------------------------

def box_jpl_tjn(f: pd.DataFrame) -> np.ndarray:
    """JPL SBDB class TJN: 4.6 < a < 5.5 au, e < 0.3. No swarm; we give it the sign of sigma."""
    return ((f["a"] > 4.6) & (f["a"] < 5.5) & (f["e"] < 0.3)).to_numpy()


def box_mpc_inferred(f: pd.DataFrame) -> np.ndarray:
    """The MPC list's envelope as inferred from its members (the cuts are unpublished)."""
    s = np.abs(f["sigma"])
    return ((f["a"] >= 4.952) & (f["a"] <= 5.450) & (f["e"] <= 0.297) & (f["inc"] <= 55.4)
            & (s >= 20.4) & (s <= 144.6)).to_numpy()


def box_v24(f: pd.DataFrame) -> np.ndarray:
    return catalogue.v24_selected(f["a"], f["e"], f["inc"], f["sigma"])


BOXES = {"JPL TJN": box_jpl_tjn, "MPC (inferred)": box_mpc_inferred, "V24 box": box_v24}


def box_label(f: pd.DataFrame, box) -> np.ndarray:
    """A box's answer as a label: inside the box, swarm from the sign of sigma."""
    inside = np.asarray(box(f), dtype=bool)
    out = np.full(len(f), "none", dtype=object)
    out[inside & (f["sigma"].to_numpy() > 0)] = "L4"
    out[inside & (f["sigma"].to_numpy() <= 0)] = "L5"
    return out
