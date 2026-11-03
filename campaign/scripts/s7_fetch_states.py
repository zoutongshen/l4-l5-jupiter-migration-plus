"""Stage 7: Horizons states at t0 for the screen's objects to verify, into data/raw/horizons_screen/<date>/.

Selects from data/runs/<out-id>/screen_k<k>.parquet:
  unlabelled   no catalogue label, inside the rule's domain
  outside      a within the domain but e, I outside it (incl. retrograde)
  rejected     labelled but the rule says no, and not in the pilot (no pilot label)
Writes the raw responses (objects/<spkid>.json, manifest) and
data/runs/<out-id>/screen_targets_k<k>.parquet: spkid, group, heliocentric
elements and sigma at t0 from the Horizons state, ready for
s7_run_map.py --source targets. Objects Horizons cannot resolve are listed, not guessed.

    python campaign/scripts/s7_fetch_states.py --out-id definition_v1 --date 2026-10-27
"""

import argparse
import json

import numpy as np
import pandas as pd

from trojans import constants, coorbital, diagnostics, fetch, paths, runs
from trojans.outputs import Manifest, write_provenance

DAYS_PER_YEAR = 365.25


def comet_designation(full_name: str) -> str:
    """'42P/Neujmin 3' -> '42P'; 'P/2019 A6 (Lemmon-PANSTARRS)' -> 'P/2019 A6'."""
    name = full_name.split("(")[0].strip()
    return name.split("/")[0] if name[0].isdigit() else name


def targets(screen: pd.DataFrame) -> pd.DataFrame:
    dom = coorbital.MapGrid(n=0)
    groups = {
        "unlabelled": screen["in_domain"] & ~screen["labelled"],
        "outside": ~screen["in_domain"] & screen["a"].between(dom.a_min, dom.a_max),
        "rejected": screen["in_domain"] & screen["labelled"] & ~screen["rule_trojan"] & ~screen["labelled_sept"],
    }
    parts = [screen.loc[mask, ["spkid", "full_name", "kind"]].assign(group=g) for g, mask in groups.items()]
    return pd.concat(parts, ignore_index=True).drop_duplicates("spkid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-id", required=True)
    parser.add_argument("--date", required=True, help="directory date for the raw responses")
    parser.add_argument("--ic-id", default="2026-09-15_giants")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    states, ic = runs.load_ic(args.ic_id)
    t0 = ic["t0_jd_tdb"]
    jupiter = coorbital.jupiter_at_t0(states)
    run = paths.run_dir(args.out_id)
    todo = targets(pd.read_parquet(run / f"screen_k{args.k}.parquet"))

    directory = paths.make_output_dir(paths.raw_dir("horizons_screen", args.date), args.resume)
    write_provenance(directory)
    manifest = Manifest(directory)
    rows, failed = [], []
    for row in todo.itertuples():
        # A periodic comet's SPK-ID can match several apparitions; then ask for the
        # one closest to t0 (CAP) by designation.
        commands = [(f"objects/{row.spkid}.json", f"DES={row.spkid};")]
        if row.kind == "comet":
            commands.append((f"objects/{row.spkid}_cap.json", f"DES={comet_designation(row.full_name)};CAP;"))
        sv, reason = None, None
        for name, command in commands:
            if not manifest.has(name):
                try:
                    manifest.save_file(directory, name, fetch.horizons_vectors(command, t0), flush=False)
                except Exception as exc:          # network or HTTP failure: report, don't guess
                    reason = repr(exc)[:200]
                    continue
            try:
                sv = fetch.parse_horizons_vectors((directory / name).read_bytes())
                break
            except ValueError as exc:
                reason = str(exc)[:200]
        if sv is None:
            failed.append({"spkid": row.spkid, "full_name": row.full_name, "reason": reason})
            continue
        if sv["jd_tdb"] != t0:
            failed.append({"spkid": row.spkid, "full_name": row.full_name, "reason": f"state at {sv['jd_tdb']}"})
            continue
        pos = np.array([[sv["x"], sv["y"], sv["z"]]])
        vel = np.array([[sv["vx"], sv["vy"], sv["vz"]]]) * DAYS_PER_YEAR
        el = {k: float(v[0]) for k, v in diagnostics.elements(pos, vel, constants.G).items()}
        rows.append({"spkid": row.spkid, "full_name": row.full_name, "kind": row.kind, "group": row.group,
                     **{k: el[k] for k in ("a", "e", "inc", "Omega", "omega", "M")},
                     "sigma": float(diagnostics.resonant_angle(el["lam"], jupiter["lam"]))})
    manifest.flush()

    out = pd.DataFrame(rows)
    # Hyperbolic or NaN orbits at t0 cannot be integrated as ellipses from elements.
    bound = (out["e"] < 1) & (out["a"] > 0) & out["M"].notna()
    for r in out[~bound].itertuples():
        failed.append({"spkid": r.spkid, "full_name": r.full_name, "reason": "unbound at t0"})
    out[bound].to_parquet(run / f"screen_targets_k{args.k}.parquet", index=False)
    pd.DataFrame(failed, columns=["spkid", "full_name", "reason"]).to_csv(
        run / f"screen_targets_failed_k{args.k}.csv", index=False)
    print(f"{int(bound.sum())} targets with states at t0 ({out[bound]['group'].value_counts().to_dict()}); "
          f"{len(failed)} failed")


if __name__ == "__main__":
    main()
