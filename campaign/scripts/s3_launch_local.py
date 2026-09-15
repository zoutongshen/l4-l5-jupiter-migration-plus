"""Stage 3: prepare a run from campaign/configs/<name>.toml and run its chunks locally.

A new run creates data/runs/<run_id>/ with a copy of the config, the chunk
assignment and provenance, then runs every chunk as its own process (logs in
data/runs/<run_id>/logs/). With --resume, finished chunks are skipped and
partial ones continue from their last checkpoint; the config must be unchanged.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from trojans import integrate, paths, runs
from trojans.outputs import write_provenance

RUN_CHUNK = Path(__file__).with_name("s3_run_chunk.py")


def prepare(name: str, run_id: str, resume: bool) -> dict:
    config_path = paths.config_path(name)
    run_dir = paths.run_dir(run_id)
    if resume:
        if (run_dir / "config.toml").read_bytes() != config_path.read_bytes():
            raise SystemExit(f"{config_path} differs from the copy in {run_dir}; refusing to resume")
        return runs.load_config_file(run_dir / "config.toml")

    paths.make_output_dir(run_dir)
    shutil.copyfile(config_path, run_dir / "config.toml")
    config = runs.load_config_file(run_dir / "config.toml")
    states, ic = runs.load_ic(config["run"]["ic_id"])
    spkids = states.loc[states["body"] == "trojan", "spkid"]
    chunks = runs.assign_chunks(spkids, config["run"]["n_chunks"], config["run"]["seed"], config["run"].get("subset"))
    chunks.to_parquet(run_dir / "chunks.parquet", index=False)
    integrate.write_settings(run_dir, integrate.settings_from_config(config, ic["dt_yr"]))
    write_provenance(run_dir)
    (run_dir / "logs").mkdir()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="config name in campaign/configs/ (without .toml)")
    parser.add_argument("--run-id", default=None, help="defaults to the config name")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="create the run directory and chunk assignment, run nothing (for slurm/)")
    args = parser.parse_args()
    run_id = args.run_id or args.config

    config = prepare(args.config, run_id, args.resume)
    if args.prepare_only:
        print(f"{run_id}: prepared {config['run']['n_chunks']} chunks in {paths.run_dir(run_id)}")
        return
    _, ic = runs.load_ic(config["run"]["ic_id"])
    n_windows = integrate.settings_from_config(config, ic["dt_yr"]).n_windows
    log_dir = paths.run_dir(run_id) / "logs"

    pending = []
    for chunk in range(config["run"]["n_chunks"]):
        status = runs.chunk_status(run_id, chunk, n_windows)
        if status != "done":
            pending.append((chunk, status == "partial"))
    print(f"{run_id}: {len(pending)} chunks to run with {args.workers} workers")

    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    active, failed = {}, []
    while pending or active:
        while pending and len(active) < args.workers:
            chunk, partial = pending.pop(0)
            cmd = [sys.executable, str(RUN_CHUNK), "--run-id", run_id, "--chunk", str(chunk)]
            if partial:
                cmd.append("--resume")
            log = open(log_dir / f"c{chunk:02d}.log", "a")
            active[chunk] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        time.sleep(5)
        for chunk, (proc, log) in list(active.items()):
            if proc.poll() is not None:
                log.close()
                del active[chunk]
                print(f"chunk {chunk:02d} exited with {proc.returncode}")
                if proc.returncode != 0:
                    failed.append(chunk)
    if failed:
        raise SystemExit(f"failed chunks: {failed}; see {log_dir}")
    print(f"{run_id}: all chunks done")


if __name__ == "__main__":
    main()
