"""Stage 1.1 (continued): Horizons states at the common epoch into data/raw/horizons/<date>/.

- planets/<id>.json   the four giant-planet system barycentres at t0
- objects/<spkid>.json every candidate Trojan whose SBDB orbit epoch is not t0
- epoch.json          t0 and how it was chosen

t0 is the most common SBDB orbit epoch among the candidates. States are
heliocentric, ecliptic J2000, in au and au/day. Requests run one at a time;
--resume skips anything already in the manifest.
"""

import argparse
import datetime

from trojans import catalogue, constants, fetch, paths
from trojans.outputs import Manifest, write_json, write_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=datetime.date.today().isoformat(),
                        help="date of the sbdb/mpc downloads to read, also used for this directory")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidates = catalogue.load_candidates(paths.raw_dir("sbdb", args.date), paths.raw_dir("mpc", args.date))
    t0 = catalogue.modal_epoch(candidates)
    off_epoch = candidates.loc[candidates["epoch_jd"] != t0, "spkid"].tolist()

    directory = paths.make_output_dir(paths.raw_dir("horizons", args.date), args.resume)
    write_provenance(directory)
    write_json(directory / "epoch.json", {
        "t0_jd_tdb": t0,
        "rule": "most common SBDB orbit epoch among candidates",
        "n_candidates": len(candidates),
        "n_off_epoch": len(off_epoch),
    })
    manifest = Manifest(directory)

    for planet, target in constants.HORIZONS_ID.items():
        filename = f"planets/{target}.json"
        if not manifest.has(filename):
            manifest.save_file(directory, filename, fetch.horizons_vectors(target, t0), note=f"{planet} system barycentre")
    print(f"t0 = JD {t0} TDB; giants done; {len(off_epoch)} off-epoch objects")

    todo = [s for s in off_epoch if not manifest.has(f"objects/{s}.json")]
    for n, spkid in enumerate(todo, 1):
        result = fetch.horizons_vectors(f"DES={spkid};", t0)
        fetch.parse_horizons_vectors(result.content)  # fail loudly on a bad response
        manifest.save_file(directory, f"objects/{spkid}.json", result, flush=(n % 25 == 0))
        if n % 100 == 0:
            print(f"  {n}/{len(todo)}")
    manifest.flush()
    print(f"done: {len(off_epoch)} object states at t0")


if __name__ == "__main__":
    main()
