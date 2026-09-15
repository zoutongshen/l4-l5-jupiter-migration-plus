"""Stage 1.1: download the catalogue inputs into data/raw/<source>/<date>/.

- sbdb: JPL SBDB asteroids in V24's element superset, everything JPL labels TJN,
        and individual records for MPC-listed objects not matched by designation
- mpc:  the MPC list of Jupiter Trojans, with its own L4/L5 labels
- v24:  a mirror of the Vokrouhlicky et al. 2024 catalogue (proper elements, families)

Horizons states are fetched separately (s1_fetch_horizons.py), once the common
epoch is known. Each directory gets manifest.json (URL, parameters, time, sha256)
and provenance.json. Rerun with --resume to fill in files a failed run missed.
"""

import argparse
import datetime

from trojans import catalogue, fetch, paths
from trojans.outputs import Manifest, write_provenance


def fetch_sbdb(date: str, resume: bool) -> None:
    directory = paths.make_output_dir(paths.raw_dir("sbdb", date), resume)
    manifest = Manifest(directory)
    queries = {
        "superset.json": dict(constraints=fetch.SBDB_SUPERSET_CONSTRAINTS),
        "class_tjn.json": dict(sb_class="TJN"),
    }
    for filename, query in queries.items():
        if manifest.has(filename):
            continue
        result = fetch.sbdb_query(**query)
        manifest.save_file(directory, filename, result)
        print(f"sbdb/{filename}: {result.extra['count']} objects")
    write_provenance(directory)


def fetch_mpc(date: str, resume: bool) -> None:
    directory = paths.make_output_dir(paths.raw_dir("mpc", date), resume)
    manifest = Manifest(directory)
    if not manifest.has("JupiterTrojans.html"):
        result = fetch.download(fetch.MPC_TROJANS_URL)
        manifest.save_file(directory, "JupiterTrojans.html", result)
        print(f"mpc/JupiterTrojans.html: {len(result.content) / 1e6:.1f} MB")
    write_provenance(directory)


def fetch_mpc_only(date: str) -> None:
    """SBDB records for MPC-listed objects not found in the SBDB queries by designation."""
    directory = paths.raw_dir("sbdb", date)
    sbdb = catalogue.load_sbdb(directory)
    mpc = catalogue.load_mpc(paths.raw_dir("mpc", date))
    _, unmatched = catalogue.attach_mpc(sbdb, mpc)
    manifest = Manifest(directory)
    for prov in unmatched["prov_des"]:
        filename = f"mpc_only/{prov.replace(' ', '_')}.json"
        if manifest.has(filename):
            continue
        manifest.save_file(directory, filename, fetch.sbdb_object(prov), note="MPC-listed, not matched by designation")
    print(f"sbdb/mpc_only: {len(unmatched)} MPC-listed objects looked up individually")


def fetch_v24(date: str, resume: bool) -> None:
    directory = paths.make_output_dir(paths.raw_dir("v24", date), resume)
    manifest = Manifest(directory)
    for subdir in fetch.V24_DIRECTORIES:
        for name in fetch.v24_listing(subdir):
            filename = f"{subdir}/{name}"
            if manifest.has(filename):
                continue
            result = fetch.download(fetch.V24_BASE_URL + filename)
            manifest.save_file(directory, filename, result)
            print(f"v24/{filename}: {len(result.content) / 1e6:.2f} MB")
    write_provenance(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=datetime.date.today().isoformat(),
                        help="fetch date used as the directory name (default: today)")
    parser.add_argument("--resume", action="store_true",
                        help="continue into existing directories, skipping files already fetched")
    args = parser.parse_args()

    fetch_sbdb(args.date, args.resume)
    fetch_mpc(args.date, args.resume)
    fetch_mpc_only(args.date)
    fetch_v24(args.date, args.resume)


if __name__ == "__main__":
    main()
