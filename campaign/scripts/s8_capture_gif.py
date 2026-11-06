"""Stage 8 animation: capture while Jupiter grows and migrates, from the stored snapshots.

Unlike s6_disc_gif.py nothing is re-integrated: every frame is a snapshot written
during the production run (snapshots.h5). Particles are coloured by their final
fate in s8_migration_report.py (run that first), or with --color-at-kyr T by the side
(sigma > 0: L4, sigma < 0: L5) of the co-orbitals at time T (for runs that end with
no Trojans, such as mig_g2_pirani). If campaign/reports/<run-id>/giants.npz exists
(scripts/s8_giants_replay.py), Saturn, Uranus and Neptune are drawn too.

Panels:
  left    rotating frame (Jupiter on +x), with Jupiter's current orbit
  middle  sigma = lambda - lambda_J against a / a_J(t)
  right   co-orbitals (|a - a_J| < 2.5 R_H) at sigma > 0 against sigma < 0 (Pirani's
          horseshoe-side excess), plus Jupiter's a(t) and nu(t)

    python campaign/scripts/s8_capture_gif.py --run-id mig_v2_pirani
Writes campaign/reports/<run-id>/capture.gif and capture_last_frame.png.
"""

import argparse
import json

import h5py
import numpy as np
import pandas as pd

from trojans import disc, paths

SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
OTHER = "#b9b8b0"
PLANET_COLOR = "#8a6d3b"
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load(run_id: str, max_particles: int, max_frames: int, seed: int = 1, color_at_yr=None):
    fates = pd.read_parquet(paths.report_dir(run_id) / "particles.parquet").set_index("id")
    chunks = sorted((paths.run_dir(run_id) / "chunks").iterdir())
    with h5py.File(chunks[0] / "snapshots.h5") as f:
        t = f["t"][:]
    frames = np.unique(np.linspace(0, len(t) - 1, min(max_frames, len(t))).round().astype(int))
    per_chunk = int(np.ceil(max_particles / len(chunks)))
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ("x", "y", "sigma", "a", "id")}
    for chunk in chunks:
        with h5py.File(chunk / "snapshots.h5") as f:
            ids = f["spkid"][:]
            # Keep every Trojan in the subset, then fill with others.
            swarm = fates.loc[ids, "swarm"].to_numpy()
            if color_at_yr is not None:
                j = int(np.argmin(np.abs(t - color_at_yr)))
                hill = (f["jupiter_m"][j] / 3) ** (1 / 3)
                co = np.abs(f["a"][j] / f["jupiter_a"][j] - 1) < 2.5 * hill
                sg = f["sigma"][j]
                swarm = np.where(co & (sg > 0), "L4", np.where(co & (sg < 0), "L5", ""))
            troj = np.flatnonzero(swarm != "")
            rest = rng.permutation(np.flatnonzero(swarm == ""))
            cols = np.sort(np.concatenate([troj, rest[:max(0, per_chunk - len(troj))]]))
            lam = np.radians(f["jupiter_lam"][:][frames])
            x, y = f["x"][frames][:, cols], f["y"][frames][:, cols]
            c, s = np.cos(-lam)[:, None], np.sin(-lam)[:, None]
            parts["x"].append(x * c - y * s)
            parts["y"].append(x * s + y * c)
            parts["sigma"].append(f["sigma"][frames][:, cols])
            parts["a"].append(f["a"][frames][:, cols] / f["jupiter_a"][:][frames][:, None])
            parts["id"].append(ids[cols])
            parts.setdefault("swarm", []).append(swarm[cols])
            jup_a, jup_m = f["jupiter_a"][:][frames], f["jupiter_m"][:][frames]
            jup_lam = lam
    tracks = {k: np.concatenate(v, axis=1 if k not in ("id", "swarm") else 0) for k, v in parts.items()}
    tracks.update(t=t[frames], jup_a=jup_a, jup_m=jup_m)
    giants_file = paths.report_dir(run_id) / "giants.npz"
    if giants_file.exists():
        g = np.load(giants_file)
        c, s = np.cos(-jup_lam)[:, None], np.sin(-jup_lam)[:, None]
        gx, gy = g["x"][frames], g["y"][frames]
        tracks.update(g_names=[str(n) for n in g["planets"]], g_x=gx * c - gy * s, g_y=gx * s + gy * c,
                      g_a=g["a"][frames], g_t=g["t"], g_a_all=g["a"])
    return tracks


GIANT_STYLE = {"saturn": ("#c9a227", "S"), "uranus": ("#4fb3bf", "U"), "neptune": ("#3b5bdb", "N")}


def render(run_id: str, max_particles: int, max_frames: int, fps: int, hold: int, color_at_kyr=None, suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    color_at_yr = None if color_at_kyr is None else 1e3 * color_at_kyr
    tr = load(run_id, max_particles, max_frames, color_at_yr=color_at_yr)
    has_giants = "g_names" in tr
    report = paths.report_dir(run_id)
    summary = json.loads((report / "summary.json").read_text())
    sides = pd.read_csv(report / "coorbital.csv")
    h = summary["history"]
    a_lim = 1.25 * max(h["a_start_au"], h["a_end_au"]) + 0.5
    if has_giants and "saturn" in tr["g_names"]:
        a_lim = max(a_lim, 1.1 * np.nanmax(tr["g_a"][:, tr["g_names"].index("saturn")]))

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.2, 1.1]})
    fig.subplots_adjust(left=0.05, right=0.94, top=0.78, bottom=0.11, wspace=0.3)
    title = (f"Jupiter {h['direction'].replace('_', ' ')}: {h['a_start_au']:.2f} → {h['a_end_au']:.2f} au, "
             f"{h['m0_earth']:.3g} M⊕ → 1 M_J")
    fig.text(0.05, 0.95, title, fontsize=15, fontweight="bold")
    n4, n5 = summary["N4"], summary["N5"]
    ratio = f"{summary['N4/N5']:.2f} ± {summary['err']:.2f}" if summary["N4/N5"] else "n/a"
    fig.text(0.05, 0.905, f"{run_id}: Trojans N4/N5 = {n4}/{n5} = {ratio}; "
             f"{len(tr['id'])} of {summary['n_disc']} particles shown, coloured by "
             + ("final fate" if color_at_kyr is None else f"co-orbital side at {color_at_kyr:g} kyr"),
             fontsize=10.5, color=TEXT_2)
    labels = (("● L4 Trojan", "● L5 Trojan") if color_at_kyr is None else
              (f"● L4 side at {color_at_kyr:g} kyr", f"● L5 side at {color_at_kyr:g} kyr"))
    xs = (0.05, 0.125, 0.2) if color_at_kyr is None else (0.05, 0.175, 0.30)
    for x, label, color in ((xs[0], labels[0], SWARM_COLOR["L4"]), (xs[1], labels[1], SWARM_COLOR["L5"]),
                            (xs[2], "● neither", MUTED)):
        fig.text(x, 0.865, label, fontsize=10.5, color=color, fontweight="bold" if color != MUTED else "normal")
    clock = fig.text(0.94, 0.95, "", ha="right", fontsize=14, family="monospace")

    order = np.argsort(np.where(tr["swarm"] == "", 0, 1), kind="stable")
    colors = np.array([SWARM_COLOR.get(s, OTHER) for s in tr["swarm"]])[order]
    sizes = np.where(tr["swarm"][order] == "", 3, 6)

    th = np.linspace(0, 2 * np.pi, 400)
    orbit, = ax.plot([], [], color=GRID, lw=0.8, zorder=1)
    ax.plot(0, 0, "o", color="#e0a000", ms=7, zorder=4)
    ax.set_xlim(-a_lim, a_lim)
    ax.set_ylim(-a_lim, a_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x rotating with Jupiter [au]")
    ax.set_ylabel("y [au]")
    ax.set_title("seen from above, rotating frame", loc="left", fontsize=10, color=TEXT_2)

    for v in (60, -60):
        bx.axvline(v, color=GRID, lw=1, zorder=1)
    bx.axhline(1.0, color=GRID, lw=1, zorder=1)
    bx.set_xlim(-180, 180)
    bx.set_ylim(0.85, 1.15)
    bx.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    bx.set_xlabel("σ = λ − λ_J [deg]   (L5 at −60, L4 at +60)")
    bx.set_ylabel("a / a_J(t)")
    bx.set_title("resonant angle against a", loc="left", fontsize=10, color=TEXT_2)

    t_myr = sides["t_yr"] / 1e6
    cx.plot(t_myr, sides["n_sigma_pos"], color=SWARM_COLOR["L4"], lw=2, label="σ > 0 (L4 side)")
    cx.plot(t_myr, sides["n_sigma_neg"], color=SWARM_COLOR["L5"], lw=2, ls=(0, (4, 2)), label="σ < 0 (L5 side)")
    cx.set_xlabel("time [Myr]")
    cx.set_ylabel("co-orbitals, |a − a_J| < 2.5 R_H")
    cx.grid(axis="y", color=GRID, lw=0.6)
    cx.set_axisbelow(True)
    cx.legend(frameon=False, loc="center right", fontsize=9)
    cx.set_title("co-orbital sides; Jupiter's a (right axis)", loc="left", fontsize=10, color=TEXT_2)
    ja = cx.twinx()
    ja.plot(t_myr, sides["jupiter_a"], color=PLANET_COLOR, lw=1.2)
    if has_giants and "saturn" in tr["g_names"]:
        sat = GIANT_STYLE["saturn"][0]
        ja.plot(tr["g_t"] / 1e6, tr["g_a_all"][:, tr["g_names"].index("saturn")], color=sat, lw=1.2, ls=(0, (3, 2)))
        ja.text(0.98, 0.93, "a_S (dashed)", transform=ja.transAxes, ha="right", va="top", fontsize=9, color=sat)
    ja.set_ylabel("a_J [au]", color=PLANET_COLOR)
    ja.tick_params(axis="y", colors=PLANET_COLOR)
    cursor = cx.axvline(0, color=TEXT_2, lw=1)

    scat_xy = ax.scatter(tr["x"][0][order], tr["y"][0][order], s=sizes, c=colors, linewidths=0, zorder=3)
    scat_sa = bx.scatter(tr["sigma"][0][order], tr["a"][0][order], s=sizes, c=colors, linewidths=0, zorder=3)
    jup, = ax.plot([], [], "o", color=PLANET_COLOR, ms=9, mec=SURFACE, mew=1.5, zorder=5)
    mass_text = ax.annotate("", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=9, color=PLANET_COLOR)
    giant_dots, giant_labels, giant_orbits = [], [], []
    if has_giants:
        for name in tr["g_names"][1:]:
            color, tag = GIANT_STYLE.get(name, (MUTED, name[0].upper()))
            giant_orbits.append(ax.plot([], [], color=color, lw=0.6, alpha=0.5, zorder=1)[0])
            giant_dots.append(ax.plot([], [], "o", color=color, ms=7, mec=SURFACE, mew=1.2, zorder=5)[0])
            giant_labels.append(ax.annotate("", (0, 0), xytext=(6, 6), textcoords="offset points",
                                            fontsize=9, color=color, annotation_clip=False))
    m_final = tr["jup_m"][-1]
    n_frames = len(tr["t"])

    def draw(i):
        k = min(i, n_frames - 1)
        t, a_j, m = tr["t"][k], tr["jup_a"][k], tr["jup_m"][k]
        nu_now = np.interp(t, sides["t_yr"], sides["nu"])
        clock.set_text(f"t = {t / 1e3:7.1f} kyr   ν = {nu_now:5.2f}")
        scat_xy.set_offsets(np.column_stack([tr["x"][k][order], tr["y"][k][order]]))
        scat_sa.set_offsets(np.column_stack([tr["sigma"][k][order], tr["a"][k][order]]))
        orbit.set_data(a_j * np.cos(th), a_j * np.sin(th))
        jup.set_data([a_j], [0.0])
        mass_text.xy = (a_j, 0.0)
        jup.set_markersize(4 + 7 * (m / m_final) ** (1 / 3))
        if has_giants:
            for g, name in enumerate(tr["g_names"][1:], start=1):
                tag = GIANT_STYLE.get(name, (MUTED, name[0].upper()))[1]
                gx, gy, ga = tr["g_x"][k, g], tr["g_y"][k, g], tr["g_a"][k, g]
                r = np.hypot(gx, gy)
                edge = r > 0.97 * a_lim                 # outside the panel: pin to the edge
                px, py = (gx, gy) if not edge else (0.97 * a_lim * gx / r, 0.97 * a_lim * gy / r)
                giant_dots[g - 1].set_data([px], [py])
                giant_dots[g - 1].set_marker(">" if edge else "o")
                giant_orbits[g - 1].set_data(ga * np.cos(th), ga * np.sin(th))
                giant_labels[g - 1].xy = (px, py)
                giant_labels[g - 1].set_text(f"{tag} {ga:.1f} au" if edge else tag)
        mass_text.set_text(f"{m / disc.M_EARTH:.3g} M⊕" if m < 0.1 * m_final else f"{m / m_final:.2f} M_J")
        cursor.set_xdata([t / 1e6, t / 1e6])
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames + hold, blit=False)
    anim.save(report / f"capture{suffix}.gif", writer=PillowWriter(fps=fps), dpi=80)
    draw(n_frames - 1)
    fig.savefig(report / f"capture{suffix}_last_frame.png", dpi=110)
    print(f"wrote {report / f'capture{suffix}.gif'} ({n_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-particles", type=int, default=3000)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold", type=int, default=20)
    parser.add_argument("--color-at-kyr", type=float, help="colour by co-orbital side at this time instead of final fate")
    parser.add_argument("--out-suffix", default="", help="write capture<suffix>.gif")
    args = parser.parse_args()
    render(args.run_id, args.max_particles, args.max_frames, args.fps, args.hold, args.color_at_kyr, args.out_suffix)


if __name__ == "__main__":
    main()
