"""Stage 3: integrate one chunk of a prepared run (called by s3_launch_local.py or slurm/).

Reads data/runs/<run_id>/{config.toml,chunks.parquet} and the run's initial
conditions; writes data/runs/<run_id>/chunks/cNN/. Each session (start or
resume) appends provenance, CPU and wall time to sessions.jsonl.
"""

import argparse
import json
import time

import pandas as pd

from trojans import initial, integrate, paths, runs
from trojans.outputs import provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = paths.run_dir(args.run_id)
    config = runs.load_config_file(run_dir / "config.toml")
    states, ic = runs.load_ic(config["run"]["ic_id"])
    chunks = pd.read_parquet(run_dir / "chunks.parquet")
    spkids = chunks.loc[chunks["chunk"] == args.chunk, "spkid"].to_numpy()
    settings = integrate.settings_from_config(config, ic["dt_yr"])
    out = paths.chunk_dir(args.run_id, args.chunk)

    if args.resume:
        runner = integrate.ChunkRunner.resume(spkids, settings, ic["giants"], out)
    else:
        paths.make_output_dir(out)
        sim = initial.build_simulation(states, spkids, settings.dt_yr)
        runner = integrate.ChunkRunner.start(sim, spkids, settings, ic["giants"], out)

    session = {**provenance(), "resume": args.resume, "t_start_yr": runner.sim.t, "n_particles": len(spkids)}
    cpu0, wall0 = time.process_time(), time.time()
    runner.run(log=lambda message: print(f"[c{args.chunk:02d}] {message}", flush=True))
    session.update(t_end_yr=runner.sim.t, cpu_s=time.process_time() - cpu0, wall_s=time.time() - wall0,
                   n_alive=int((runner.events["removal_reason"] == integrate.ALIVE).sum()))
    with open(out / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session) + "\n")


if __name__ == "__main__":
    main()
