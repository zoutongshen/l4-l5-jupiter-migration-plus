"""Download functions for JPL SBDB, JPL Horizons, the MPC and the V24 catalogue.

Each function returns what it downloaded. Nothing here writes to disk: the
calling script decides where the bytes go and records a manifest.
"""

import json
import re
import time
from dataclasses import dataclass, field

import requests

USER_AGENT = "L4-L5-trojans-campaign/0.1 (MSc research, Leiden)"
TIMEOUT_S = 120

SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
MPC_TROJANS_URL = "https://minorplanetcenter.net/iau/lists/JupiterTrojans.html"
V24_BASE_URL = "https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/"

# greeks = L4 swarm, trojans = L5 swarm (proper elements); plus family memberships.
V24_DIRECTORIES = (
    "greeks-hcm_20230405_GS",
    "trojans-hcm_20230405_GS",
    "trojans_L4_families_2023Apr",
    "trojans_L5_families_2023Apr",
)

SBDB_FIELDS = (
    "spkid", "full_name", "pdes", "name", "epoch", "a", "e", "i", "om", "w", "ma",
    "H", "class", "condition_code", "data_arc", "n_obs_used", "first_obs", "t_jup",
)

# Superset of V24's selection (4.95 < a < 5.45 au, e < 0.6, I < 50 deg); sigma is
# applied later, once every object has a state at the common epoch.
SBDB_SUPERSET_CONSTRAINTS = {"AND": ["a|RG|4.9|5.5", "e|LT|0.6", "i|LT|50"]}


@dataclass
class Download:
    url: str
    params: dict
    content: bytes
    fetched_utc: str
    extra: dict = field(default_factory=dict)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            response = requests.get(
                url, params=params, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}
            )
            if response.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def download(url: str, params: dict | None = None) -> Download:
    response = _get(url, params)
    return Download(url=url, params=params or {}, content=response.content, fetched_utc=_utc_now())


def sbdb_query(constraints: dict | None = None, sb_class: str | None = None) -> Download:
    """Asteroids matching element constraints and/or an SBDB orbit class."""
    params = {"fields": ",".join(SBDB_FIELDS), "sb-kind": "a", "full-prec": "1"}
    if constraints is not None:
        params["sb-cdata"] = json.dumps(constraints, separators=(",", ":"))
    if sb_class is not None:
        params["sb-class"] = sb_class
    result = download(SBDB_QUERY_URL, params)
    payload = json.loads(result.content)
    if int(payload["count"]) != len(payload["data"]):
        raise RuntimeError(f"SBDB returned {len(payload['data'])} of {payload['count']} rows")
    result.extra["count"] = int(payload["count"])
    return result


SBDB_OBJECT_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"


def sbdb_object(designation: str) -> Download:
    """One object's record from the SBDB API (resolves aliases to the current designation)."""
    return download(SBDB_OBJECT_URL, {"sstr": designation, "full-prec": "1", "phys-par": "1"})


def v24_listing(directory: str) -> list[str]:
    """File names in one directory of the V24 catalogue (Apache index page)."""
    html = _get(V24_BASE_URL + directory + "/").text
    names = re.findall(r'href="([^"?/][^"]*)"', html)
    return [n for n in names if not n.endswith("/")]


def horizons_vectors(command: str, jd_tdb: float) -> Download:
    """Heliocentric ecliptic-J2000 state of one target at one TDB epoch (au, au/day)."""
    params = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'500@10'",
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "ICRF",
        "TLIST": f"'{jd_tdb}'",
        "TLIST_TYPE": "JD",
        "TIME_TYPE": "TDB",
        "OUT_UNITS": "AU-D",
        "VEC_TABLE": "2",
        "CSV_FORMAT": "YES",
        "VEC_LABELS": "NO",
        "OBJ_DATA": "NO",
    }
    return download(HORIZONS_URL, params)


def parse_horizons_vectors(payload: bytes) -> dict:
    """Extract jd, x, y, z, vx, vy, vz from a Horizons VECTORS CSV response."""
    text = json.loads(payload)["result"]
    match = re.search(r"\$\$SOE\s*(.*?)\s*\$\$EOE", text, re.S)
    if match is None:
        raise ValueError("no $$SOE/$$EOE block in Horizons response:\n" + text[:2000])
    parts = [p.strip() for p in match.group(1).split(",")]
    jd = float(parts[0])
    x, y, z, vx, vy, vz = (float(v) for v in parts[2:8])
    return {"jd_tdb": jd, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}
