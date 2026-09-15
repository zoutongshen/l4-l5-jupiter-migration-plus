"""Parse the raw downloads, select Trojans, and build the master table.

Selection follows Vokrouhlicky et al. 2024 (V24, Sec. 2): at a common epoch,
4.95 < a < 5.45 au, e < 0.6, I < 50 deg and 20 < |sigma| < 130 deg, with
sigma = lambda - lambda_J; L4 has sigma > 0. The JPL (class TJN) and MPC labels
are carried alongside for comparison.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trojans import constants, diagnostics

V24_A_RANGE_AU = (4.95, 5.45)
V24_E_MAX = 0.6
V24_I_MAX_DEG = 50.0
V24_SIGMA_RANGE_DEG = (20.0, 130.0)

COMPLETE_H_MAX = 13.6            # meeting 2026-09-15
DISISTO2013_LAST_NUMBER = 329008  # docs/MSc_PROJECT_PLAN.md Stage 1.4

_PAREN = re.compile(r"\(([^)]*)\)")


# --- JPL SBDB ---------------------------------------------------------------

def _sbdb_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    return pd.DataFrame(payload["data"], columns=payload["fields"])


def _provisional(full_name: str, pdes: str) -> str | None:
    """Principal provisional designation, e.g. '2013 BD8' (None for old-style names)."""
    if not re.fullmatch(r"\d+", pdes):
        return pdes
    match = _PAREN.search(full_name)
    return match.group(1) if match else None


def load_sbdb(raw_dir: Path) -> pd.DataFrame:
    """Union of the V24 superset query and JPL class TJN, one row per object."""
    superset = _sbdb_frame(raw_dir / "superset.json")
    tjn = _sbdb_frame(raw_dir / "class_tjn.json")
    df = pd.concat([superset, tjn]).drop_duplicates("spkid").reset_index(drop=True)

    out = pd.DataFrame({"spkid": df["spkid"].astype("int64")})
    out["full_name"] = df["full_name"].str.strip()
    out["pdes"] = df["pdes"]
    out["name"] = df["name"]
    numbered = df["pdes"].str.fullmatch(r"\d+")
    out["number"] = pd.to_numeric(df["pdes"].where(numbered), errors="coerce").astype("Int64")
    out["prov_des"] = [_provisional(f, p) for f, p in zip(df["full_name"], df["pdes"])]
    out["epoch_jd"] = df["epoch"].astype(float)
    for col in ["a", "e", "i", "om", "w", "ma", "H", "data_arc", "t_jup"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["n_obs_used"] = pd.to_numeric(df["n_obs_used"], errors="coerce").astype("Int64")
    out["condition_code"] = df["condition_code"]
    out["first_obs"] = df["first_obs"]
    out["sbdb_class"] = df["class"]
    out["jpl_tjn"] = out["sbdb_class"].eq("TJN")
    out["in_sbdb_superset"] = out["spkid"].isin(superset["spkid"].astype("int64"))
    return out


def sbdb_object_row(payload: dict) -> dict:
    """A load_sbdb-style row from one SBDB API object record."""
    obj, orbit = payload["object"], payload["orbit"]
    el = {e["name"]: e["value"] for e in orbit["elements"]}
    phys = {p["name"]: p["value"] for p in payload.get("phys_par", [])}
    pdes = obj["des"]
    fullname = obj["fullname"]
    return {
        "spkid": int(obj["spkid"]),
        "full_name": fullname,
        "pdes": pdes,
        "name": None,
        "number": int(pdes) if re.fullmatch(r"\d+", pdes) else None,
        "prov_des": _provisional(fullname, pdes),
        "epoch_jd": float(orbit["epoch"]),
        **{k: float(el[k]) for k in ["a", "e", "i", "om", "w", "ma"]},
        "H": float(phys["H"]) if "H" in phys else np.nan,
        "data_arc": float(orbit["data_arc"]) if orbit.get("data_arc") else np.nan,
        "t_jup": np.nan,
        "n_obs_used": orbit.get("n_obs_used"),
        "condition_code": orbit.get("condition_code"),
        "first_obs": orbit.get("first_obs"),
        "sbdb_class": obj["orbit_class"]["code"],
        "jpl_tjn": obj["orbit_class"]["code"] == "TJN",
        "in_sbdb_superset": False,
    }


def load_candidates(sbdb_raw: Path, mpc_raw: Path) -> pd.DataFrame:
    """Every object that any of the three definitions could call a Trojan.

    SBDB superset and class TJN, plus MPC-listed objects that SBDB knows under
    another designation (looked up one by one into sbdb_raw/mpc_only/).
    """
    sbdb = load_sbdb(sbdb_raw)
    mpc = load_mpc(mpc_raw)
    table, unmatched = attach_mpc(sbdb, mpc)
    extra = []
    for _, row in unmatched.iterrows():
        path = sbdb_raw / "mpc_only" / f"{row['prov_des'].replace(' ', '_')}.json"
        record = sbdb_object_row(json.loads(path.read_text()))
        hit = table.index[table["spkid"] == record["spkid"]]
        if len(hit):
            table.loc[hit, "mpc_trojan"] = True
            table.loc[hit, "mpc_ln"] = row["mpc_ln"]
        else:
            extra.append({**record, "mpc_trojan": True, "mpc_ln": row["mpc_ln"]})
    if extra:
        table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
        table["number"] = table["number"].astype("Int64")
        table["n_obs_used"] = pd.to_numeric(table["n_obs_used"]).astype("Int64")
    return table


# --- MPC Jupiter Trojan list -------------------------------------------------

def load_mpc(raw_dir: Path) -> pd.DataFrame:
    """The MPC list as number / provisional designation / L4-L5 label / H."""
    html = (raw_dir / "JupiterTrojans.html").read_text(encoding="latin-1")
    block = html[html.index("<pre>") + 5: html.index("</pre>")]
    lines = block.splitlines()
    header = next(line for line in lines if line.startswith("Designation"))
    c_prov, c_ln, c_q = header.index("Prov. Des."), header.index("Ln"), header.index("q  ")
    rows = []
    for line in lines:
        ln = line[c_ln:c_ln + 2]
        if ln not in ("L4", "L5"):
            continue
        number = re.search(r"\((\d+)\)", line[:c_prov])
        tail = line[c_ln + 2:].split()
        rows.append({
            "number": int(number.group(1)) if number else None,
            "prov_des": line[c_prov:c_ln].strip() or None,
            "mpc_ln": ln,
            "mpc_H": float(tail[2]),
        })
    df = pd.DataFrame(rows)
    df["number"] = df["number"].astype("Int64")
    return df


def attach_mpc(sbdb: pd.DataFrame, mpc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add mpc_trojan / mpc_ln to the SBDB table; return it and the unmatched MPC rows."""
    by_number = {int(n): i for i, n in mpc["number"].dropna().items()}
    by_prov = {p: i for i, p in mpc.loc[mpc["number"].isna(), "prov_des"].items()}
    matched = []
    for number, prov in zip(sbdb["number"], sbdb["prov_des"]):
        if pd.notna(number) and int(number) in by_number:
            matched.append(by_number[int(number)])
        elif prov in by_prov:
            matched.append(by_prov[prov])
        else:
            matched.append(None)
    out = sbdb.copy()
    idx = pd.array(matched, dtype="Int64")
    out["mpc_trojan"] = ~pd.isna(idx)
    out["mpc_ln"] = [mpc.at[i, "mpc_ln"] if pd.notna(i) else None for i in idx]
    unmatched = mpc.drop(index=[i for i in idx if pd.notna(i)])
    return out, unmatched


# --- Vokrouhlicky et al. 2024 catalogue -------------------------------------

def v24_key(number, prov_des) -> list[str]:
    """Candidate V24 designations for one object: number first, then provisional."""
    keys = []
    if pd.notna(number):
        keys.append(str(int(number)))
    if prov_des:
        keys.append(prov_des.replace(" ", "_"))
    return keys


def load_v24_proper(raw_dir: Path) -> pd.DataFrame:
    """V24 synthetic proper elements; greeks = L4, trojans = L5."""
    frames = []
    for subdir, swarm in [("greeks-hcm_20230405_GS", "L4"), ("trojans-hcm_20230405_GS", "L5")]:
        df = pd.read_csv(raw_dir / subdir / "follow.proper.pro", sep=r"\s+", header=None)
        frames.append(pd.DataFrame({
            "v24_desig": df[0].astype(str),
            "v24_swarm": swarm,
            "v24_H": df[1],
            "v24_a_p": df[2],
            "v24_e_p": df[3],
            "v24_sini_p": df[4],
        }))
    return pd.concat(frames, ignore_index=True)


def load_v24_families(raw_dir: Path) -> pd.DataFrame:
    """Family membership, one row per (object, family list)."""
    rows = []
    for subdir, swarm in [("trojans_L4_families_2023Apr", "L4"), ("trojans_L5_families_2023Apr", "L5")]:
        for path in sorted((raw_dir / subdir).glob("*_family.list_*")):
            family = path.name.split("_family.list_")[0]
            cut = re.match(r"(\d+)", path.name.split("_family.list_")[1]).group(1)
            for line in path.read_text().splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                number, second = line.split()[:2]
                desig = number if number != "?" else second
                rows.append({
                    "v24_desig": desig,
                    "v24_family": family,
                    "v24_family_swarm": swarm,
                    "v24_v_cut": int(cut),
                    "v24_interlopers_removed": path.name.endswith("WO_INTERLOPERS"),
                })
    return pd.DataFrame(rows)


def match_v24(table: pd.DataFrame, v24: pd.DataFrame) -> pd.Series:
    """Index into ``v24`` (by v24_desig) for each table row, or NA."""
    lookup = {d: i for i, d in v24["v24_desig"].items()}
    out = []
    for number, prov in zip(table["number"], table["prov_des"]):
        out.append(next((lookup[k] for k in v24_key(number, prov) if k in lookup), None))
    return pd.array(out, dtype="Int64")


# The nominal families of V24 Table 4, as (list stem, cutoff) -> published name.
# The directory also holds the broad 60 m/s Ennomos clan (388 members, including
# Deiphobus); Table 4's Ennomos is the 50 m/s "LOWI" list.
V24_TABLE4_LISTS = {
    ("3548_Eurybates", 40): "Eurybates",
    ("20961_Arkesiaos", 40): "Arkesilaos",
    ("624_Hektor", 40): "Hektor",
    ("2797_Teucer", 60): "Teucer",
    ("9799_Thronium", 20): "Thronium",
    ("4489_Dracius", 60): "Dracius",
    ("24534_2001CX27", 60): "2001 CX27",
    ("38599_1999XC210", 60): "1999 XC210",
    ("432306_2009SQ357", 40): "2009 SQ357",
    ("1867_Deiphobus", 60): "Deiphobus",
    ("4709_Ennomos_LOWI", 50): "Ennomos",
    ("247341_2001UV209", 55): "2001 UV209",
    ("261781_2006BG132", 60): "2006 BG132",
}


# --- Selection ---------------------------------------------------------------

def v24_selected(a, e, inc_deg, sigma_deg) -> np.ndarray:
    abs_sigma = np.abs(sigma_deg)
    return (
        (a > V24_A_RANGE_AU[0]) & (a < V24_A_RANGE_AU[1])
        & (e < V24_E_MAX) & (inc_deg < V24_I_MAX_DEG)
        & (abs_sigma > V24_SIGMA_RANGE_DEG[0]) & (abs_sigma < V24_SIGMA_RANGE_DEG[1])
    )


def swarm_from_sigma(sigma_deg) -> np.ndarray:
    return np.where(np.asarray(sigma_deg) > 0.0, "L4", "L5")


def modal_epoch(sbdb: pd.DataFrame) -> float:
    return float(sbdb["epoch_jd"].mode().iloc[0])


# --- States at the common epoch ---------------------------------------------

def _horizons_state(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    from trojans import fetch

    s = fetch.parse_horizons_vectors(path.read_bytes())
    pos = np.array([s["x"], s["y"], s["z"]])
    vel = np.array([s["vx"], s["vy"], s["vz"]]) * constants.JULIAN_YEAR_D  # au/day -> au/yr
    return pos, vel, s["jd_tdb"]


def giants_at_t0(horizons_raw: Path, t0: float) -> dict:
    """Heliocentric states (au, au/yr) of the giant-planet system barycentres."""
    out = {}
    for planet, target in constants.HORIZONS_ID.items():
        pos, vel, jd = _horizons_state(horizons_raw / "planets" / f"{target}.json")
        if jd != t0:
            raise ValueError(f"{planet} state is at JD {jd}, not t0 = {t0}")
        out[planet] = (pos, vel)
    return out


def planet_elements(pos, vel, planet: str) -> dict:
    """Heliocentric elements of a giant, taken about the Sun plus that planet (JPL convention)."""
    mu = constants.G * (1.0 + constants.GIANT_MASS[planet])
    return {k: float(v[0]) for k, v in diagnostics.elements(pos[None], vel[None], mu).items()}


def states_at_t0(candidates: pd.DataFrame, horizons_raw: Path, t0: float):
    """Heliocentric positions and velocities (au, au/yr) of all candidates at t0.

    Objects whose SBDB orbit is at t0 are converted from elements with mu = GM_sun,
    SBDB's convention; the rest use Horizons states.
    """
    n = len(candidates)
    pos, vel = np.empty((n, 3)), np.empty((n, 3))
    on_epoch = (candidates["epoch_jd"] == t0).to_numpy()
    c = candidates.loc[on_epoch]
    pos[on_epoch], vel[on_epoch] = diagnostics.cartesian(
        c["a"], c["e"], c["i"], c["om"], c["w"], c["ma"], constants.G
    )
    for row in np.flatnonzero(~on_epoch):
        spkid = int(candidates["spkid"].iloc[row])
        p, v, jd = _horizons_state(horizons_raw / "objects" / f"{spkid}.json")
        if jd != t0:
            raise ValueError(f"{spkid}: Horizons state at JD {jd}, not t0 = {t0}")
        pos[row], vel[row] = p, v
    source = np.where(on_epoch, "sbdb_elements", "horizons")
    return pos, vel, source


# --- Master table ------------------------------------------------------------

def _label_region(v24, jpl, mpc) -> list[str]:
    out = []
    for flags in zip(v24, jpl, mpc):
        names = [name for name, flag in zip(("V24", "JPL", "MPC"), flags) if flag]
        out.append("+".join(names) if names else "none")
    return out


def build_master_table(candidates, pos, vel, source, jupiter_pv, v24_proper, v24_families):
    """One row per candidate: identity, flags, elements and sigma at t0, labels, A/B/C, V24 matches.

    Returns the table and Jupiter's heliocentric elements at t0.
    """
    jupiter = planet_elements(*jupiter_pv, "jupiter")
    el = diagnostics.elements(pos, vel, constants.G)
    sigma = diagnostics.resonant_angle(el["lam"], jupiter["lam"])

    t = candidates[[
        "spkid", "full_name", "name", "number", "prov_des", "H", "condition_code", "data_arc",
        "n_obs_used", "first_obs", "t_jup", "sbdb_class", "epoch_jd", "jpl_tjn", "mpc_trojan", "mpc_ln",
    ]].copy().reset_index(drop=True)
    t["complete"] = t["H"] <= COMPLETE_H_MAX
    t["disisto2013"] = t["number"].notna() & (t["number"] <= DISISTO2013_LAST_NUMBER)
    t["disisto2013"] = t["disisto2013"].fillna(False).astype(bool)
    t["state_source"] = source
    for k, key in enumerate(["x", "y", "z"]):
        t[key] = pos[:, k]
        t["v" + key] = vel[:, k]
    for key in ["a", "e", "inc", "Omega", "omega", "M", "varpi", "lam"]:
        t[key] = el[key]
    t["sigma"] = sigma
    t["swarm"] = swarm_from_sigma(sigma)
    t["v24_selected"] = v24_selected(el["a"], el["e"], el["inc"], sigma)
    t["mpc_trojan"] = t["mpc_trojan"].fillna(False).astype(bool)
    t["label_region"] = _label_region(t["v24_selected"], t["jpl_tjn"], t["mpc_trojan"])
    t["in_run"] = t["v24_selected"] | t["jpl_tjn"] | t["mpc_trojan"]

    mass_ratio = constants.GIANT_MASS["jupiter"] / (1.0 + constants.GIANT_MASS["jupiter"])
    t["A"], t["B"], t["C"] = diagnostics.abc(el["a"], el["e"], el["inc"], el["Omega"], el["varpi"], sigma, jupiter, mass_ratio)

    hit = match_v24(t, v24_proper)
    for col in ["v24_desig", "v24_swarm", "v24_H", "v24_a_p", "v24_e_p", "v24_sini_p"]:
        t[col] = [v24_proper.at[i, col] if pd.notna(i) else None for i in hit]
    for col in ["v24_H", "v24_a_p", "v24_e_p", "v24_sini_p"]:
        t[col] = t[col].astype(float)

    keys = [v24_key(n, p) for n, p in zip(t["number"], t["prov_des"])]
    nominal = v24_families[[
        (f, cut) in V24_TABLE4_LISTS for f, cut in zip(v24_families["v24_family"], v24_families["v24_v_cut"])
    ]]
    fam_of = nominal.groupby("v24_desig").apply(
        lambda g: ";".join(sorted(V24_TABLE4_LISTS[(f, c)] for f, c in zip(g["v24_family"], g["v24_v_cut"])))
    ).to_dict()
    lists_of = v24_families.groupby("v24_desig").apply(
        lambda g: ";".join(sorted(f"{f}@{c}" for f, c in zip(g["v24_family"], g["v24_v_cut"])))
    ).to_dict()
    t["v24_family"] = [next((fam_of[k] for k in ks if k in fam_of), None) for ks in keys]
    t["v24_family_lists"] = [next((lists_of[k] for k in ks if k in lists_of), None) for ks in keys]
    return t, jupiter


def label_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Counts per label region and sigma-swarm, with MPC swarm disagreements."""
    t = table.copy()
    t["mpc_swarm_disagrees"] = t["mpc_trojan"] & t["mpc_ln"].notna() & (t["mpc_ln"] != t["swarm"])
    t["v24_catalogue_swarm_disagrees"] = t["v24_swarm"].notna() & (t["v24_swarm"] != t["swarm"])
    return (
        t.groupby(["label_region", "swarm"])
        .agg(n=("spkid", "size"),
             n_complete=("complete", "sum"),
             n_mpc_swarm_disagrees=("mpc_swarm_disagrees", "sum"),
             n_v24_catalogue_swarm_disagrees=("v24_catalogue_swarm_disagrees", "sum"),
             median_H=("H", "median"))
        .reset_index()
        .sort_values(["label_region", "swarm"])
    )
