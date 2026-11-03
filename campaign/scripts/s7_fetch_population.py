"""Stage 7: every asteroid and comet orbit in JPL SBDB, into data/raw/sbdb_population/<date>/.

- asteroids.json  all asteroids, no element constraints
- comets.json     all comets (no H; mean anomaly from tp where SBDB gives none)

Used by s7_screen.py to apply the learned Trojan rule to the whole known
population, not only to the element box the pilot's catalogue was drawn from.

    python campaign/scripts/s7_fetch_population.py --date 2026-10-27
"""

import argparse
import datetime

from trojans import fetch, paths
from trojans.outputs import Manifest, write_provenance

ASTEROID_FIELDS = ("spkid", "full_name", "pdes", "name", "epoch", "a", "e", "i", "om", "w", "ma",
                   "H", "class", "condition_code", "data_arc", "n_obs_used", "first_obs")
COMET_FIELDS = ("spkid", "full_name", "pdes", "name", "epoch", "a", "e", "i", "om", "w", "ma", "tp",
                "class", "condition_code", "data_arc", "n_obs_used", "first_obs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    directory = paths.make_output_dir(paths.raw_dir("sbdb_population", args.date), args.resume)
    write_provenance(directory)
    manifest = Manifest(directory)
    for filename, kind, fields in (("asteroids.json", "a", ASTEROID_FIELDS), ("comets.json", "c", COMET_FIELDS)):
        if manifest.has(filename):
            continue
        result = fetch.sbdb_query(kind=kind, fields=fields)
        manifest.save_file(directory, filename, result)
        print(f"sbdb_population/{filename}: {result.extra['count']} objects, {len(result.content) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
