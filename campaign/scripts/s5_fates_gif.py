"""Stage 5a animation: one GIF per escaper, from its escape to its end state.

The fate run keeps only coarse tracks, so each escaper is re-integrated exactly as in
s5_escaper_fates.py: the same replay to the hand-off, then the same TRACE simulation
stepped in the same blocks. The integration is deterministic, and the script checks
that each re-run ends when the fate run said it did.

Frames are log-spaced in time after escape (lifetimes run from 25 kyr to 15 Myr).
Each frame shows:
  left   top view of the escaper's osculating orbit, the giants and their orbits; the
         view zooms out as the orbit grows. Heliocentric inside 30 au, barycentric
         beyond, as in the fate run.
  right  perihelion q against aphelion Q, the path so far coloured by orbit class
         (classes from the fate run's tracks; black = co-orbital with Jupiter).

Writes campaign/reports/<out_id>/gif/<spkid>_<group>_<swarm>.gif.

    python campaign/scripts/s5_fates_gif.py --out-id escaper_fates_pilot --workers 8
"""

import argparse
import json
import multiprocessing as mp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from trojans import constants, paths

import s5_escaper_fates as fates

N_FRAMES = 150
HOLD_FRAMES = 18
FPS = 15
CLASS_COLOR = {"jfc": "#1baf7a", "centaur": "#eda100", "outer": "#4a3aa7", "trojan_like": "#0b0b0b"}
CLASS_LABEL = {"jfc": "Jupiter-family comet", "centaur": "Centaur", "outer": "beyond Neptune",
               "trojan_like": "co-orbital with Jupiter", "other": "other", "inner": "other"}
OTHER = "#b9b8b0"
SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
PLANET_COLOR = "#8a6d3b"
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PLANET_LETTER = ["J", "S", "U", "N"]


def fmt_time(t):
    if t < 1e3:
        return f"{t:,.0f} yr"
    if t < 1e6:
        return f"{t / 1e3:,.1f} kyr"
    return f"{t / 1e6:,.2f} Myr"


def snapshot(sim):
    ps = sim.particles
    sun = ps[0]
    r_sun = np.sqrt((ps[5].x - sun.x) ** 2 + (ps[5].y - sun.y) ** 2 + (ps[5].z - sun.z) ** 2)
    centre = sun if r_sun < fates.BARYCENTRIC_BEYOND_AU else sim.com(last=5)
    o = ps[5].orbit(primary=centre)
    return {
        "a": o.a, "e": o.e, "inc": o.inc, "Omega": o.Omega, "omega": o.omega,
        "x": ps[5].x - centre.x, "y": ps[5].y - centre.y, "r": r_sun,
        "planets": [(ps[i].x - centre.x, ps[i].y - centre.y) for i in range(1, 5)],
        "planet_a": [ps[i].orbit(primary=sun).a for i in range(1, 5)],
    }


def rerun(args):
    """Re-integrate one escaper and record snapshots at the frame times."""
    spkid, h, dt_yr, t_fate, frame_t = args
    sim = fates.fate_simulation(h, dt_yr)
    t0 = sim.t
    frames, k = [], 0
    import rebound
    while k < len(frame_t):
        try:
            sim.steps(fates.CHECK_EVERY_STEPS)
        except (rebound.Escape, rebound.Collision):
            break
        while k < len(frame_t) and sim.t - t0 >= frame_t[k]:
            frames.append({"t": sim.t - t0, **snapshot(sim)})
            k += 1
        if sim.t - t0 >= t_fate:
            break
    return spkid, frames, sim.t - t0


def orbit_curve(s, limit):
    """x, y of the osculating orbit (top view), clipped to the view for hyperbolae."""
    a, e = s["a"], s["e"]
    if e < 1:
        f = np.linspace(0, 2 * np.pi, 400)
    else:
        f_max = np.arccos(-1 / e) * 0.999
        f = np.linspace(-f_max, f_max, 400)
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(f))
    ok = (r > 0) & (r < 3 * limit)
    u = f[ok] + s["omega"]
    r = r[ok]
    cO, sO, ci = np.cos(s["Omega"]), np.sin(s["Omega"]), np.cos(s["inc"])
    x = r * (cO * np.cos(u) - sO * np.sin(u) * ci)
    y = r * (sO * np.cos(u) + cO * np.sin(u) * ci)
    return x, y


def view_limits(frames):
    lim = []
    for s in frames:
        Q = s["a"] * (1 + s["e"]) if s["e"] < 1 else s["r"]
        lim.append(np.clip(1.15 * max(Q, s["r"]), 7.0, 1200.0))
    lim = np.array(lim)
    # smooth in log space so the zoom doesn't jitter between frames
    log = np.log(lim)
    kernel = np.ones(7) / 7
    padded = np.r_[np.full(3, log[0]), log, np.full(3, log[-1])]
    # never smaller than the frame itself needs: the final ejection is fast
    return np.maximum(np.exp(np.convolve(padded, kernel, mode="valid")), lim)


def render(job):
    spkid, frames, meta, track, episodes, out_path = job
    t_track = track["t_after_escape_yr"].to_numpy()
    cls_track = track["cls"].to_numpy()

    def cls_at(t):
        for e in episodes:
            if e["t_start_after_escape_yr"] <= t <= e["t_start_after_escape_yr"] + e["duration_yr"]:
                return "trojan_like"
        i = max(0, np.searchsorted(t_track, t, side="right") - 1)
        return cls_track[i]

    classes = [cls_at(s["t"]) for s in frames]
    q = np.array([s["a"] * (1 - s["e"]) for s in frames])
    Q = np.array([s["a"] * (1 + s["e"]) if s["e"] < 1 else np.inf for s in frames])
    lims = view_limits(frames)

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.12, top=0.84, wspace=0.22)
    group = "Trojan" if meta["group"] == "trojan" else "Interloper"
    fig.text(0.03, 0.95, f"{spkid}", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.03 + 0.012 * len(str(spkid)) + 0.03, 0.95, f"{group} escaped from {meta['swarm']}",
             fontsize=12, color=SWARM_COLOR[meta["swarm"]], fontweight="bold")
    clock = fig.text(0.97, 0.95, "", ha="right", fontsize=12, color=TEXT, family="monospace")
    status = fig.text(0.97, 0.895, "", ha="right", fontsize=10, color=TEXT_2)

    def draw(i):
        i = min(i, len(frames) - 1)
        s, lim = frames[i], lims[i]
        ax.clear()
        bx.clear()
        # left: top view
        th = np.linspace(0, 2 * np.pi, 200)
        for a_p in s["planet_a"]:
            ax.plot(a_p * np.cos(th), a_p * np.sin(th), color=GRID, lw=0.8, zorder=1)
        for (px, py), letter in zip(s["planets"], PLANET_LETTER):
            ax.plot(px, py, "o", color=PLANET_COLOR, ms=4, zorder=3)
            if lim < 80 or letter in "UN":
                ax.annotate(letter, (px, py), xytext=(3, 3), textcoords="offset points", fontsize=7, color=PLANET_COLOR)
        ax.plot(0, 0, "o", color="#e0a000", ms=6, zorder=3)
        color = CLASS_COLOR.get(classes[i], OTHER)
        x, y = orbit_curve(s, lim)
        ax.plot(x, y, color=color, lw=1.6, zorder=2)
        ax.plot(s["x"], s["y"], "o", color=color, ms=6, mec=SURFACE, mew=1, zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"orbit seen from above (field ±{lim:,.0f} au)", loc="left", fontsize=9, color=TEXT_2)
        # right: perihelion vs aphelion
        for xv, label in [(s["planet_a"][0], "Jupiter"), (s["planet_a"][3], "Neptune")]:
            bx.axvline(xv, color=GRID, lw=1)
            bx.annotate(label, (xv, 2.6e3), xytext=(3, 0), textcoords="offset points", fontsize=7, color=MUTED,
                        rotation=90, va="top")
        bx.plot([0.1, 3e3], [0.1, 3e3], color=GRID, lw=1, ls=":")
        bx.annotate("q = Q (circular)", (12, 12), xytext=(-4, 4), textcoords="offset points", fontsize=7,
                    color=MUTED, ha="right")
        bx.axhline(2.2e3, color=GRID, lw=1)
        bx.annotate("unbound", (0.12, 2.2e3), xytext=(2, 3), textcoords="offset points", fontsize=7, color=MUTED)
        Qp = np.where(np.isfinite(Q[: i + 1]), Q[: i + 1], 2.2e3)
        cols = [CLASS_COLOR.get(c, OTHER) for c in classes[: i + 1]]
        bx.plot(q[: i + 1], Qp, color=GRID, lw=0.8, zorder=1)
        bx.scatter(q[: i + 1], Qp, c=cols, s=9, zorder=2, lw=0)
        bx.plot(q[i], Qp[i], "o", color=color, ms=9, mec=TEXT, mew=1, zorder=3)
        bx.set_xscale("log")
        bx.set_yscale("log")
        bx.set_xlim(0.1, 60)
        bx.set_ylim(3, 3e3)
        bx.set_xlabel("perihelion q [au]")
        bx.set_ylabel("aphelion Q [au]")
        bx.set_title("path so far, coloured by orbit class", loc="left", fontsize=9, color=TEXT_2)
        clock.set_text(f"t = {fmt_time(s['t'])} after escape")
        text = CLASS_LABEL.get(classes[i], "other")
        if i == len(frames) - 1:
            end = {"ejected": "ejected (hyperbolic)", "oort": "past 1000 au, still bound"}.get(meta["fate"], meta["fate"])
            text = f"end: {end}"
        status.set_text(text)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames) + HOLD_FRAMES)
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=80)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    parser.add_argument("--out-id", default="escaper_fates_pilot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", type=int, nargs="*", help="spkids (default: all followed escapers)")
    args = parser.parse_args()

    run = paths.run_dir(args.out_id)
    meta = {int(p.stem): json.loads(p.read_text()) for p in (run / "fates").glob("*.json")}
    targets = fates.escapers(args.run_id)
    targets = targets[targets["spkid"].isin(meta)]
    if args.only:
        targets = targets[targets["spkid"].isin(args.only)]
    settings = json.loads((paths.run_dir(args.run_id) / "settings.json").read_text())
    dt = settings["dt_yr"]

    handoff = fates.replay(args.run_id, targets, log=lambda m: None)
    jobs = []
    for s in targets["spkid"]:
        t_fate = meta[int(s)]["t_fate_after_escape_yr"]
        frame_t = np.logspace(np.log10(dt * fates.CHECK_EVERY_STEPS), np.log10(t_fate), N_FRAMES)
        jobs.append((int(s), handoff[int(s)], dt, t_fate, frame_t))

    out_dir = paths.report_dir(args.out_id) / "gif"
    paths.make_output_dir(out_dir, resume=out_dir.exists())
    with mp.get_context("spawn").Pool(args.workers) as pool:
        reruns = pool.map(rerun, jobs)
        render_jobs = []
        for spkid, frames, t_end in reruns:
            m = meta[spkid]
            drift = t_end - m["t_fate_after_escape_yr"]
            flag = "ok" if abs(drift) < 2 * dt * fates.CHECK_EVERY_STEPS else f"MISMATCH ({drift:+.3g} yr)"
            print(f"{spkid}: re-run ends {fmt_time(t_end)} after escape, fate run {fmt_time(m['t_fate_after_escape_yr'])}  {flag}")
            track = pd.read_parquet(run / "tracks" / f"{spkid}.parquet")
            name = f"{spkid}_{m['group']}_{m['swarm']}.gif"
            render_jobs.append((spkid, frames, m, track, m["episodes"], out_dir / name))
        for path in pool.imap_unordered(render, render_jobs):
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
