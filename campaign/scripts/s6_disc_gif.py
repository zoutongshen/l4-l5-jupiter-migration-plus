"""Stage 6 animation: the fixed and the growing Jupiter side by side, one GIF.

The production runs keep only 10 kyr window means, so a subset of the same disc is
re-integrated here with frequent snapshots. The particles are massless, so each
one's orbit does not depend on which others share its simulation. Jupiter is
stepped exactly as in production (``sim.steps(check_every_steps)``, then the
forcing), so the re-run follows the production orbits, and each particle is
coloured by its production fate (s6_disc_report.py). The script prints how often
a production Trojan sits on its own swarm's side at the last frame, as a check.

Each row (one per arm) shows:
  left    the rotating frame: Jupiter fixed on the +x axis, L4 at +60 deg, L5 at -60 deg
  middle  resonant angle sigma = lambda - lambda_J against semimajor axis
  right   L4 and L5 tadpole counts per 10 kyr window in the full production run,
          with a cursor at the current time

    python campaign/scripts/s6_disc_gif.py record --workers 8
    python campaign/scripts/s6_disc_gif.py render
Writes data/runs/<run>/gif_tracks.npz and campaign/reports/disc_capture/disc_capture.gif.
"""

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from trojans import constants, diagnostics, disc, integrate, paths

RUNS = {"fixed": "disc_fixed", "grow": "disc_grow"}
N_PAIRS = 1500
SUBSET_SEED = 20261024
T_END_YR = 3.0e5
N_FRAMES = 150
HOLD_FRAMES = 20
FPS = 12

SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
OTHER = "#b9b8b0"
PLANET_COLOR = "#8a6d3b"
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def frame_times():
    return np.linspace(0.0, T_END_YR, N_FRAMES + 1)


def subset_rows(n_pairs_disc: int) -> np.ndarray:
    pairs = np.sort(np.random.default_rng(SUBSET_SEED).choice(n_pairs_disc, N_PAIRS, replace=False))
    return np.sort(np.concatenate([2 * pairs, 2 * pairs + 1]))


def record_task(task: dict) -> dict:
    settings = json.loads((paths.run_dir(task["run_id"]) / "settings.json").read_text())
    growth = disc.Growth(**settings["growth"])
    every = settings["settings"]["check_every_steps"]
    a_j = settings["a_jupiter_au"]
    ids = np.asarray(task["ids"], dtype=np.int64)
    sim = disc.build_simulation(np.asarray(task["pos"]), np.asarray(task["vel"]), ids, growth, a_j,
                                settings["settings"]["dt_yr"])
    forcing = disc.GrowthForcing(growth, a_j)

    class Holder:  # stands in for the ChunkRunner the forcing writes into
        pass

    holder = Holder()
    forcing(sim, holder)
    n = len(ids)
    times = frame_times()
    xy = np.full((len(times), n, 2), np.nan, np.float32)
    sig = np.full((len(times), n), np.nan, np.float32)
    a = np.full((len(times), n), np.nan, np.float32)
    mass = np.zeros(len(times))
    mu = constants.G * constants.CENTRAL_MASS
    for k, t in enumerate(times):
        while sim.t < t - 1e-9:
            sim.steps(every)
            forcing(sim, holder)
        xyz = np.empty((sim.N, 3))
        vxyz = np.empty((sim.N, 3))
        sim.serialize_particle_data(xyz=xyz, vxvyvz=vxyz)
        pos, vel = xyz[2:] - xyz[0], vxyz[2:] - vxyz[0]
        jx = xyz[1] - xyz[0]
        lam_j = np.arctan2(jx[1], jx[0])
        c, s = np.cos(-lam_j), np.sin(-lam_j)
        xy[k, :, 0] = c * pos[:, 0] - s * pos[:, 1]
        xy[k, :, 1] = s * pos[:, 0] + c * pos[:, 1]
        el = diagnostics.elements(pos, vel, mu)
        sig[k] = diagnostics.resonant_angle(el["lam"], np.degrees(lam_j))
        a[k] = el["a"]
        mass[k] = sim.particles[1].m
    return {"arm": task["arm"], "ids": ids, "xy": xy, "sigma": sig, "a": a, "mass": mass}


def record(workers: int) -> None:
    tasks = []
    for arm, run_id in RUNS.items():
        init = np.load(paths.run_dir(run_id) / "disc_initial.npz")
        rows = subset_rows(len(init["id"]) // 2)
        for part in np.array_split(rows.reshape(-1, 2), workers // len(RUNS)):
            r = part.ravel()
            tasks.append({"arm": arm, "run_id": run_id, "ids": init["id"][r].tolist(),
                          "pos": init["pos"][r].tolist(), "vel": init["vel"][r].tolist()})
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(record_task, tasks)
    for arm, run_id in RUNS.items():
        parts = [r for r in results if r["arm"] == arm]
        np.savez_compressed(paths.run_dir(run_id) / "gif_tracks.npz", t=frame_times(),
                            ids=np.concatenate([p["ids"] for p in parts]),
                            xy=np.concatenate([p["xy"] for p in parts], axis=1),
                            sigma=np.concatenate([p["sigma"] for p in parts], axis=1),
                            a=np.concatenate([p["a"] for p in parts], axis=1), mass=parts[0]["mass"])
        print(f"{arm}: {sum(len(p['ids']) for p in parts)} particles, {len(frame_times())} frames")


def load_arm(arm: str) -> dict:
    run_id = RUNS[arm]
    tracks = dict(np.load(paths.run_dir(run_id) / "gif_tracks.npz"))
    fates = pd.read_parquet(paths.report_dir(run_id) / "particles.parquet").set_index("id")
    events = pd.concat(pd.read_parquet(c / "events.parquet")
                       for c in sorted((paths.run_dir(run_id) / "chunks").iterdir())).set_index("spkid")
    ids = tracks["ids"]
    tracks["swarm"] = fates.loc[ids, "swarm"].to_numpy()
    tracks["t_removed"] = events.loc[ids, "t_removed"].to_numpy()
    tracks["formation"] = pd.read_csv(paths.report_dir(run_id) / "formation.csv")
    tracks["summary"] = json.loads((paths.report_dir(run_id) / "summary.json").read_text())
    last_sigma = tracks["sigma"][-1]
    for swarm, sign in (("L4", 1), ("L5", -1)):
        m = tracks["swarm"] == swarm
        print(f"check {arm} {swarm}: {np.mean(np.sign(last_sigma[m]) == sign):.3f} of production "
              f"{swarm} Trojans on the {swarm} side at {T_END_YR / 1e3:.0f} kyr")
    return tracks


def render() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    arms = {arm: load_arm(arm) for arm in RUNS}
    a_j = 5.2026
    m_final = constants.GIANT_MASS["jupiter"]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), gridspec_kw={"width_ratios": [1, 1.25, 1.05]})
    fig.subplots_adjust(left=0.05, right=0.985, top=0.79, bottom=0.07, wspace=0.28, hspace=0.5)
    fig.text(0.05, 0.965, "Trojan swarms from a mirror-symmetric disc, no migration", fontsize=15,
             fontweight="bold")
    fig.text(0.05, 0.932, "Same 3,000 disc particles in both rows, coloured by their fate in the 20,000-particle run",
             fontsize=10.5, color=TEXT_2)
    for x, label, color in ((0.05, "● L4 Trojan", SWARM_COLOR["L4"]), (0.125, "● L5 Trojan", SWARM_COLOR["L5"]),
                            (0.2, "● neither (removed particles vanish)", MUTED)):
        fig.text(x, 0.902, label, fontsize=10.5, color=color, fontweight="bold" if color != MUTED else "normal")
    clock = fig.text(0.985, 0.965, "", ha="right", fontsize=14, family="monospace")

    order = {}
    artists = {}
    for row, arm in enumerate(RUNS):
        tr = arms[arm]
        # Trojans drawn last, on top of the grey background population.
        order[arm] = np.argsort(np.where(tr["swarm"] == "", 0, 1), kind="stable")
        colors = np.array([SWARM_COLOR.get(s, OTHER) for s in tr["swarm"]])[order[arm]]
        sizes = np.where(tr["swarm"][order[arm]] == "", 3, 6)
        ax, bx, cx = axes[row]
        label = "Arm 0: Jupiter at full mass from t = 0" if arm == "fixed" else \
            "Arm 1: Jupiter grows 10 M⊕ → 1 M_J over 0.1 Myr"
        ax.annotate(label, (0, 1.15), xycoords="axes fraction", fontsize=12, fontweight="bold")

        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(a_j * np.cos(th), a_j * np.sin(th), color=GRID, lw=0.8, zorder=1)
        for ang, name in ((60, "L4"), (-60, "L5")):
            # Outside the swarm, so the dots never cover the label.
            px, py = 6.75 * np.cos(np.radians(ang)), 6.75 * np.sin(np.radians(ang))
            ax.text(px + 0.6, py, name, fontsize=11, color=SWARM_COLOR[name], fontweight="bold",
                    ha="left", va="center", zorder=6)
        ax.plot(0, 0, "o", color="#e0a000", ms=7, zorder=4)
        ax.set_xlim(-7.2, 7.2)
        ax.set_ylim(-7.2, 7.2)
        ax.set_aspect("equal")
        ax.set_xlabel("x rotating with Jupiter [au]")
        ax.set_ylabel("y [au]")
        ax.set_title("seen from above, rotating frame", loc="left", fontsize=10, color=TEXT_2)

        for v in (60, -60):
            bx.axvline(v, color=GRID, lw=1, zorder=1)
        bx.axhline(a_j, color=GRID, lw=1, zorder=1)
        bx.set_xlim(-180, 180)
        bx.set_ylim(4.75, 5.65)
        bx.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        bx.set_xlabel("σ = λ − λ_J [deg]   (L5 at −60, L4 at +60)")
        bx.set_ylabel("semimajor axis a [au]")
        bx.set_title("resonant angle against a", loc="left", fontsize=10, color=TEXT_2)

        f = tr["formation"]
        t_myr = f["t_end_yr"] / 1e6
        for swarm in ("L4", "L5"):
            cx.plot(t_myr, f[swarm], color=SWARM_COLOR[swarm], lw=2, label=swarm,
                    ls="-" if swarm == "L4" else (0, (4, 2)))
        cx.set_xlim(0, 1.2)
        cx.set_ylim(0, 3300)
        cx.set_xlabel("time [Myr]")
        cx.set_ylabel("tadpoles per 10 kyr window")
        cx.grid(axis="y", color=GRID, lw=0.6)
        cx.set_axisbelow(True)
        s = tr["summary"]
        cx.set_title(f"full run: Trojans N4/N5 = {s['N4']}/{s['N5']} = {s['N4/N5']:.2f} ± {s['err']:.2f}",
                     loc="left", fontsize=10, color=TEXT_2)
        cx.legend(frameon=False, loc="lower right", fontsize=10)
        cursor = cx.axvline(0, color=TEXT_2, lw=1)

        xy0 = tr["xy"][0][order[arm]]
        scat_xy = ax.scatter(xy0[:, 0], xy0[:, 1], s=sizes, c=colors, linewidths=0, zorder=3)
        scat_sa = bx.scatter(tr["sigma"][0][order[arm]], tr["a"][0][order[arm]], s=sizes, c=colors,
                             linewidths=0, zorder=3)
        jup, = ax.plot([a_j], [0], "o", color=PLANET_COLOR, ms=9, mec=SURFACE, mew=1.5, zorder=5)
        mass_text = ax.annotate("", (a_j, 0), xytext=(8, -14), textcoords="offset points", fontsize=9,
                                color=PLANET_COLOR)
        artists[arm] = (scat_xy, scat_sa, jup, mass_text, cursor)

    times = arms["fixed"]["t"]

    def draw(i):
        k = min(i, N_FRAMES)
        t = times[k]
        clock.set_text(f"t = {t / 1e3:6.1f} kyr")
        for arm, (scat_xy, scat_sa, jup, mass_text, cursor) in artists.items():
            tr = arms[arm]
            o = order[arm]
            gone = (np.isfinite(tr["t_removed"]) & (tr["t_removed"] <= t))[o]
            xy = tr["xy"][k][o].copy()
            xy[gone] = np.nan
            sig, a = tr["sigma"][k][o].copy(), tr["a"][k][o].copy()
            sig[gone] = np.nan
            scat_xy.set_offsets(xy)
            scat_sa.set_offsets(np.column_stack([sig, a]))
            frac = tr["mass"][k] / m_final
            jup.set_markersize(4 + 7 * frac ** (1 / 3))
            mass_text.set_text(f"{frac:.2f} M_J" if arm == "grow" else "1 M_J")
            cursor.set_xdata([t / 1e6, t / 1e6])
        return []

    out = paths.CAMPAIGN_ROOT / "reports" / "disc_capture"
    out.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=N_FRAMES + 1 + HOLD_FRAMES, blit=False)
    anim.save(out / "disc_capture.gif", writer=PillowWriter(fps=FPS), dpi=80)
    draw(N_FRAMES)
    fig.savefig(out / "disc_capture_last_frame.png", dpi=110)
    print(f"wrote {out / 'disc_capture.gif'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["record", "render"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "record":
        record(args.workers)
    else:
        render()


if __name__ == "__main__":
    main()
