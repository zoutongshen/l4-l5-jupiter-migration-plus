"""Stage 0.6: measure cost and check the sampling interval and time step.

  cost     short in-process runs: CPU per Trojan-Myr for chunk sizes 500 and 2,000
           and WHFast safe_mode 1 vs 0, with and without the runner's sampling.
           Scratch outputs in data/runs/benchmark_cost/.
  compare  reads the finished runs benchmark_k1, benchmark_k4, benchmark_k8 and
           benchmark_halfdt (1,000 Trojans, 1 Myr; launch them with
           s3_launch_local.py) and compares encounter events and final classes.

Results go to campaign/reports/benchmark/.
"""

import argparse
import json
import time

import h5py
import numpy as np
import pandas as pd

from trojans import initial, integrate, paths, runs
from trojans.outputs import write_json, write_provenance

IC_ID = "2026-09-15_giants"
COST_SPAN_YR = 2000.0


def cost() -> None:
    states, ic = runs.load_ic(IC_ID)
    spkids = runs.assign_chunks(states.loc[states["body"] == "trojan", "spkid"], 1, 20260915)["spkid"].to_numpy()
    scratch = paths.make_output_dir(paths.run_dir("benchmark_cost"))
    write_provenance(scratch)
    rows = []
    for n in (500, 2000):
        for safe_mode in (1, 0):
            settings = integrate.Settings(t_end_yr=COST_SPAN_YR, dt_yr=ic["dt_yr"], check_every_steps=8,
                                          window_yr=1000.0, checkpoint_every_windows=100, safe_mode=safe_mode)
            steps = settings.n_windows * settings.checks_per_window * settings.check_every_steps

            sim = initial.build_simulation(states, spkids[:n], ic["dt_yr"])
            sim.ri_whfast.safe_mode = safe_mode
            cpu = time.process_time()
            for _ in range(steps // settings.check_every_steps):
                sim.steps(settings.check_every_steps)
                if safe_mode == 0:
                    sim.synchronize()
            bare = time.process_time() - cpu

            sim = initial.build_simulation(states, spkids[:n], ic["dt_yr"])
            runner = integrate.ChunkRunner.start(sim, spkids[:n], settings, ic["giants"], scratch / f"n{n}_safe{safe_mode}")
            cpu = time.process_time()
            runner.run(log=lambda *_: None)
            full = time.process_time() - cpu

            trojan_myr = n * steps * ic["dt_yr"] / 1e6
            rows.append({"n_trojans": n, "safe_mode": safe_mode, "check_every_steps": 8,
                         "cpu_s_per_trojan_myr_integration": bare / trojan_myr,
                         "cpu_s_per_trojan_myr_with_sampling": full / trojan_myr,
                         "cpu_s_per_particle_step": bare / (n * steps)})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    report = paths.make_output_dir(paths.report_dir("benchmark"), resume=paths.report_dir("benchmark").exists())
    table.to_csv(report / "cost.csv", index=False, float_format="%.4g")
    print(table.to_string(index=False))


def _run_outputs(run_id: str):
    chunk = paths.chunk_dir(run_id, 0)
    events = pd.read_parquet(chunk / "events.parquet").set_index("spkid").sort_index()
    with h5py.File(chunk / "windows.h5") as f:
        order = np.argsort(f["spkid"][:])
        last = {"class": f["class"][-1][order], "C": f["C"][-1][order], "spkid": f["spkid"][:][order]}
    sessions = [json.loads(line) for line in (chunk / "sessions.jsonl").read_text().splitlines()]
    return events, last, sum(s["cpu_s"] for s in sessions)


def compare() -> None:
    k1, k1_last, k1_cpu = _run_outputs("benchmark_k1")
    summary = {"reference": "benchmark_k1", "n_trojans": len(k1)}
    for run_id in ("benchmark_k4", "benchmark_k8", "benchmark_halfdt"):
        ev, last, cpu = _run_outputs(run_id)
        rec_ref, rec = k1["t_hill_record"].notna(), ev["t_hill_record"].notna()
        rem_ref, rem = k1["removal_reason"] != "alive", ev["removal_reason"] != "alive"
        both = rec_ref & rec
        same_class = last["class"] == k1_last["class"]
        summary[run_id] = {
            "cpu_s": cpu,
            "n_hill_record": int(rec.sum()), "n_hill_record_ref": int(rec_ref.sum()),
            "hill_record_only_here": int((rec & ~rec_ref).sum()), "hill_record_only_ref": int((rec_ref & ~rec).sum()),
            "median_abs_dt_hill_record_yr": float(np.median(np.abs(ev.loc[both, "t_hill_record"] - k1.loc[both, "t_hill_record"]))) if both.any() else None,
            "n_removed": int(rem.sum()), "n_removed_ref": int(rem_ref.sum()),
            "removed_disagree": int((rem != rem_ref).sum()),
            "final_class_agreement": float(np.mean(same_class)),
            "final_C_median_rel_diff": float(np.nanmedian(np.abs(last["C"] - k1_last["C"]) / k1_last["C"])),
        }
    summary["benchmark_k1"] = {"cpu_s": k1_cpu, "n_hill_record": int(k1["t_hill_record"].notna().sum()),
                               "n_removed": int((k1["removal_reason"] != "alive").sum())}
    report = paths.make_output_dir(paths.report_dir("benchmark"), resume=paths.report_dir("benchmark").exists())
    write_json(report / "accuracy.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["cost", "compare"])
    args = parser.parse_args()
    {"cost": cost, "compare": compare}[args.mode]()


if __name__ == "__main__":
    main()
