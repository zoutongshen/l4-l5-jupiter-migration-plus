"""Pirani+2019a's growth tracks for the stage-9 N-body runs, with their verification.

Inputs: her curves extracted from the PDF vectors (s9_digitize_pirani.py -> configs/tracks/pirani19a/).
  Fig. 1  a(m) of J, S, U, N (nominal);  Fig. 3  a(m) of Jupiter at 8 rates;  Fig. 13  adot(m) of Jupiter.
Model: trojans/pirani_tracks.py (BLJ15 recipe on the Ida+2016 disc).

What is taken from her figures and what is modelled:
  path a(m)   her Fig. 1 / Fig. 3 curves, exactly.
  clock m(t)  the growth model run along her path (``integrate(..., path=...)``): pebbles (Ida+2016
              eqs. 59-61) to isolation, then Piso & Youdin contraction, then Machida runaway capped
              at 0.8 Mdot*. Calibrated: FLUX_FACTOR (pebble flux x Ida eq. 54), fitted to Jupiter's
              clock from her Figs. 1+13 and final mass; ISO_FACTOR (x BLJ15 eq. 16), so that the gas
              phase starts where her dashed lines start, for J and S together; TAU_S (pebble Stokes number,
              not given by her), chosen from the planets' positions in her Figs. 4/5 snapshots
              (one nominal run, 2.3-3.0 Myr): tau = 0.08 puts J and S within 9 kyr of every
              snapshot, against 32 kyr with Ida's eq. 47 tau (~0.14). Ice-giant start times
              (her "~2.70 Myr") inferred from their final masses.
  rate scan   her sect. 3.3 "contraction or expansion of the nominal track": a_k(t) = a_end,k +
              k (a_nom(t) - a_end,nom), same m(t) (verified on Figs. 3 and 13: k constant to 1%).
              k = 0.125 for her "x0.15" (Fig. 13 rate ratio 0.126, Fig. 8 Delta a = 1.7 au).
  Saturn in the rate scan: same rule on her nominal Saturn path, with the end chosen so the start
              is her Table 1 value (the paper gives no Saturn rate tracks).

Checks written to campaign/reports/pirani_tracks/: summary.json and verification.png, including
the full forward model (migration integrated, PBK11 torque scaled by TORQUE_SCALE) against Fig. 1.

    python campaign/scripts/s9_pirani_tracks.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trojans import pirani_tracks as pt  # noqa: E402

CAMPAIGN = Path(__file__).resolve().parents[1]
DIG = CAMPAIGN / "configs" / "tracks" / "pirani19a"
OUT_TRACKS = DIG
REPORT = CAMPAIGN / "reports" / "pirani_tracks"

FLUX_FACTOR = 4.3                      # x Ida+2016 eq. 54 (with TAU_S)
TAU_S = 0.08                           # pebble Stokes number (BLJ15's range 0.05-0.5); see docstring
ISO_FACTOR = 1.066
TORQUE_SCALE = 0.6                     # forward-model check only
START = {"jupiter": 2.31e6, "saturn": 2.56e6}     # her sect. 3.1; Jupiter's is refined by jupiter_start()
FINAL_MASS_FIG1 = {"jupiter": 306.0, "saturn": 113.0, "uranus": 18.1, "neptune": 14.4}
RATES = {0.125: "x0.15", 0.25: "x0.25", 0.5: "x0.50", 0.75: "x0.75", 1.0: "x1.00", 1.25: "x1.25",
         1.5: "x1.50", 2.0: "x2.00"}
TABLE1 = {0.0: (5.4, 8.6), 0.125: (7.1, 10.45), 0.25: (8.8, 11.9), 0.5: (11.7, 14.6), 0.75: (15.0, 18.0),
          1.0: (18.4, 21.3), 1.25: (22.2, 25.5), 1.5: (25.6, 28.9), 2.0: (32.3, 35.5)}
N_ROWS = 3000


def load(name, smooth=True):
    """Her a(m) curve. Smoothed: medians in log-m bins (0.25 dex below 2 M_E, 0.08 dex above),
    joined by a monotone (PCHIP) spline. Her plotted vertices step by ~0.01-0.03 au where a barely
    changes (m < 1 M_E), and the N-body forcing differentiates a(t). The largest deviation from
    her curve is recorded in summary.json (smoothing_max_dev_au)."""
    a, m, g = np.loadtxt(DIG / f"{name}.csv", delimiter=",", skiprows=1).T
    o = np.argsort(m)
    m, a, g = m[o], np.minimum.accumulate(a[o]), g[o]
    if not smooth:
        return m, a, g
    lm = np.log10(m)
    split = np.log10(2.0)                          # coarser below 2 M_E, where a barely moves
    edges = np.concatenate([np.arange(lm[0], split, 0.25), np.arange(split, lm[-1] + 0.08, 0.08)])
    nodes, vals = [lm[0]], [a[0]]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lm > lo) & (lm <= hi)
        if sel.sum() >= 2:
            nodes.append(float(np.median(lm[sel])))
            vals.append(float(np.median(a[sel])))
    nodes.append(lm[-1])
    vals.append(a[-1])
    nodes, idx = np.unique(nodes, return_index=True)
    vals = np.minimum.accumulate(np.array(vals)[idx])
    grid = np.linspace(lm[0], lm[-1], 600)
    a_s = PchipInterpolator(nodes, vals)(grid)
    return 10**grid, np.minimum.accumulate(a_s), np.interp(grid, lm, g)


def params(t0, m_seed, **kw):
    return pt.TrackParams(t0_yr=t0, m_seed_earth=m_seed, flux_factor=FLUX_FACTOR, iso_factor=ISO_FACTOR,
                          tau_s=TAU_S, dt_yr=50.0, **kw)


M_JOIN = 3.0          # Jupiter: below this mass the path is built from her Fig. 13 rate


def jupiter_path(r):
    """Jupiter's a(m): her Fig. 1 above M_JOIN; below it a(m) = a_her(M_JOIN) + int |adot_Fig13| dt
    along the clock r, so that adot(t) of the track is her Fig. 13 rate where her path is too flat
    to differentiate."""
    m, a, _ = load("fig1_jupiter")
    mr, vr, _ = np.loadtxt(DIG / "fig13_nominal.csv", delimiter=",", skiprows=1).T
    o = np.argsort(mr)
    mm, tt = np.maximum.accumulate(r["m"]), r["t"]
    lo = np.logspace(np.log10(m[0]), np.log10(M_JOIN), 200)
    t_lo = np.interp(lo, mm, tt)
    v = 10 ** np.interp(np.log10(lo), np.log10(mr[o]), np.log10(vr[o]), left=np.log10(vr[o][0])) / 1e6
    da = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t_lo))])
    a_join = float(np.interp(np.log10(M_JOIN), np.log10(m), a))
    a_lo = a_join + (da[-1] - da)
    hi = m > M_JOIN
    return np.concatenate([lo, m[hi]]), np.concatenate([a_lo, a[hi]])


HER_CLOCK = (3.0, 100.0)   # Jupiter: between these masses the clock is hers, |adot|_Fig13 / |da/dm|_Fig1


def her_window_clock(r, pm, pa, lo=HER_CLOCK[0], hi=HER_CLOCK[1], t_end=3.0e6):
    """Replace the model clock between ``lo`` and ``hi`` M_E by the one her Figs. 1 and 13 imply.

    Her path has two sharp kinks there (gas onset at 20.5 M_E; runaway once M_env = M_core,
    ~37 M_E); with the model's capped gas accretion the track's adot then dips to ~48 and
    spikes to ~130 au/Myr, while her Fig. 13 (42 vertices) is smooth. Her implied clock needs
    dm/dt up to ~1800 M_E/Myr just after isolation, ~3x BLJ15's 0.8 Mdot* cap: either her
    onset is not capped like that or Fig. 13 is too coarse to show a dip; we follow her figure.
    Above ``hi`` the model clock is stretched so that the track still ends at ``t_end``.
    """
    mm = np.maximum.accumulate(r["m"])
    keep = np.concatenate([[True], np.diff(mm) > 0])
    mm, tt = mm[keep], r["t"][keep]
    mr, vr, _ = np.loadtxt(DIG / "fig13_nominal.csv", delimiter=",", skiprows=1).T
    o = np.argsort(mr)
    G = np.logspace(np.log10(lo), np.log10(hi), 600)
    slope = np.abs(np.gradient(np.interp(G, pm, pa), G))
    v = 10 ** np.interp(np.log10(G), np.log10(mr[o]), np.log10(vr[o])) / 1e6
    t_lo, t_hi = np.interp(lo, mm, tt), np.interp(hi, mm, tt)
    t_win = t_lo + np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] / v[1:] + slope[:-1] / v[:-1]) * np.diff(G))])
    below, above = mm < lo, mm > hi
    stretch = (t_end - t_win[-1]) / (t_end - t_hi)
    m_new = np.concatenate([mm[below], G, mm[above]])
    t_new = np.concatenate([tt[below], t_win, t_win[-1] + (tt[above] - t_hi) * stretch])
    out = dict(r)
    out.update(model_t=tt, model_m=mm, t=t_new, m=m_new, a=np.interp(np.log10(m_new), np.log10(pm), pa),
               her_clock_window=(lo, hi), late_stretch=float(stretch), window_kyr_model=float((t_hi - t_lo) / 1e3),
               window_kyr_her=float((t_win[-1] - t_lo) / 1e3))
    return out


def clock(name, t0):
    m, a, _ = load(f"fig1_{name}")
    r = pt.integrate(params(t0, m[0]), path=(m, a))
    if name == "jupiter":                          # second pass along the Fig.-13-based low-mass path
        pm, pa = jupiter_path(r)
        r = pt.integrate(params(t0, pm[0]), path=(pm, pa))
        r = her_window_clock(r, pm, pa)
        r["path"] = (pm, pa)
    return r


def jupiter_start():
    """Jupiter's start (her "t ~ 2.31 Myr") from her Fig. 4 snapshots at 2.5 and 2.6 Myr, when the other
    giants are still seeds, so her Jupiter is on its track: the mean timing offset there is set to zero."""
    snap = np.loadtxt(DIG / "fig4_planets.csv", delimiter=",", skiprows=1)
    clean = [(t, a) for t, a in zip(snap[:, 0], snap[:, 1]) if t in (2.5, 2.6)]

    def offset(t0):
        r = clock("jupiter", t0)
        return np.mean([r["t"][np.argmax(r["a"] < a)] / 1e6 - t for t, a in clean])
    return brentq(offset, 2.25e6, 2.35e6, xtol=200.0)


def ice_start(name):
    m, a, _ = load(f"fig1_{name}")
    f = lambda t0: pt.integrate(params(t0, m[0]), path=(m, a))["m"][-1] - FINAL_MASS_FIG1[name]
    return brentq(f, 2.6e6, 2.95e6, xtol=200.0)


def table(t, a, m, t_end=3.0e6):
    """Rows (t, a, m) from t = 0 (seed parked at its start a) to the end of the gas, N_ROWS samples."""
    grid = np.concatenate([[0.0], np.linspace(t[0], t_end, N_ROWS - 1)])
    return grid, np.interp(grid, t, a), np.exp(np.interp(grid, t, np.log(m)))


def write(name, t, a, m, note):
    path = OUT_TRACKS / f"track_{name}.csv"
    with open(path, "w") as f:
        f.write(f"t_yr,a_au,m_earth  # Pirani+2019a reproduction: {note}\n")
        np.savetxt(f, np.column_stack([t, a, m]), delimiter=",", fmt="%.10g")
    return path.name


def her_clock():
    """Jupiter's clock from her Figs. 1 + 13 alone, t(m) = int da / adot on 0.2-100 M_E."""
    mJ, aJ, _ = load("fig1_jupiter")
    mr, vr, _ = np.loadtxt(DIG / "fig13_nominal.csv", delimiter=",", skiprows=1).T
    o = np.argsort(mr)
    grid = np.logspace(np.log10(0.2), 2, 300)
    ag = np.interp(np.log10(grid), np.log10(mJ), aJ)
    vg = 10 ** np.interp(np.log10(grid), np.log10(mr[o]), np.log10(vr[o]))
    t = np.concatenate([[0.0], np.cumsum(-np.diff(ag) / (0.5 * (vg[1:] + vg[:-1])) * 1e6)])
    return grid, t, vg


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    summary = {"alpha": pt.gasdisc.ALPHA, "flux_factor": FLUX_FACTOR, "tau_s": TAU_S, "iso_factor": ISO_FACTOR,
               "tracks": {}, "checks": {}}
    clocks = {}
    starts = dict(START)
    starts["jupiter"] = jupiter_start()
    for n in ("uranus", "neptune"):
        starts[n] = ice_start(n)
    for n in ("jupiter", "saturn", "uranus", "neptune"):
        mr_, ar_, _ = load(f"fig1_{n}", smooth=False)
        ms_, as_, _ = load(f"fig1_{n}")
        summary["checks"].setdefault("smoothing_max_dev_au", {})[n] = float(
            np.max(np.abs(np.interp(np.log10(mr_), np.log10(ms_), as_) - ar_)))
        r = clock(n, starts[n])
        clocks[n] = r
        if n == "jupiter":
            summary["checks"]["jupiter_her_clock_window"] = {
                "masses_earth": r["her_clock_window"], "model_kyr": r["window_kyr_model"],
                "her_kyr": r["window_kyr_her"], "late_stretch": r["late_stretch"]}
        summary["checks"][n] = {"t_start_myr": starts[n] / 1e6, "m_3myr_model": float(r["m"][-1]),
                                "m_3myr_fig1": FINAL_MASS_FIG1[n], "a_end": float(r["a"][-1]),
                                "m_isolation_model": r["m_iso_reached"],
                                "m_gas_start_fig1": float(load(f"fig1_{n}")[0][load(f"fig1_{n}")[2] == 1].min())
                                if (load(f"fig1_{n}")[2] == 1).any() else None}
        t, a, m = table(r["t"], r["a"], r["m"])
        summary["tracks"][f"nominal_{n}"] = write(f"nominal_{n}", t, a, m, f"nominal {n}, Fig. 1 path, modelled clock")

    # Jupiter's clock against her own (Figs. 1 + 13)
    gm, th, vh = her_clock()
    rj = clocks["jupiter"]
    tm = np.interp(gm, rj["m"], rj["t"])
    i5 = np.argmin(abs(gm - 5))
    off = (tm - tm[i5]) - (th - th[i5])
    summary["checks"]["jupiter_clock_offset_kyr"] = {f"{M:g}": float(np.interp(M, gm, off) / 1e3)
                                                    for M in (1, 2, 10, 20, 30, 50, 100)}
    tmod = np.interp(gm, rj["model_m"], rj["model_t"])
    off_model = (tmod - tmod[i5]) - (th - th[i5])
    summary["checks"]["jupiter_model_clock_offset_kyr"] = {f"{M:g}": float(np.interp(M, gm, off_model) / 1e3)
                                                          for M in (1, 2, 10, 20, 30, 50, 100)}

    # rate scan and in situ: Jupiter (her Fig. 3 ends) and Saturn (Table 1 starts)
    tJ, aJ, mJ = table(rj["t"], rj["a"], rj["m"])
    rs = clocks["saturn"]
    tS, aS, mS = table(rs["t"], rs["a"], rs["m"])
    aJ_end, aS_end = aJ[-1], aS[-1]
    for k, lab in RATES.items():
        ends = load(f"fig3_{lab}")[1].min()
        a_k = ends + k * (aJ - aJ_end)
        summary["tracks"][f"rate_{lab}_jupiter"] = write(f"rate_{lab}_jupiter", tJ, a_k, mJ,
                                                        f"rate k = {k} (her {lab}), Jupiter")
        aS0 = TABLE1[k][1]
        endS = aS0 - k * (aS[1] - aS_end)
        summary["tracks"][f"rate_{lab}_saturn"] = write(f"rate_{lab}_saturn", tS, endS + k * (aS - aS_end), mS,
                                                       f"rate k = {k} (her {lab}), Saturn, start = Table 1")
        summary["checks"].setdefault("rate_starts", {})[lab] = {"jupiter": float(a_k[1]), "table1_j": TABLE1[k][0],
                                                              "saturn": float(endS + k * (aS[1] - aS_end)),
                                                              "table1_s": aS0}
    summary["tracks"]["insitu_jupiter"] = write("insitu_jupiter", tJ, np.full_like(tJ, 5.4), mJ, "in situ Jupiter, 5.4 au")
    summary["tracks"]["insitu_saturn"] = write("insitu_saturn", tS, np.full_like(tS, 8.6), mS, "in situ Saturn, 8.6 au")
    a_out = aJ[-1] + aJ[1] - aJ                       # 5.08 -> 18.4 au, her sect. 4.2 outward test
    summary["tracks"]["outward_jupiter"] = write("outward_jupiter", tJ, a_out, mJ, "outward Jupiter (nominal mirrored)")

    # forward model (migration integrated): does the physics alone land on her path?
    fwd = {}
    for n in ("jupiter", "saturn", "uranus", "neptune"):
        m0, a0, _ = load(f"fig1_{n}")
        fwd[n] = pt.integrate(pt.TrackParams(t0_yr=starts[n], a0_au=float(a0[0]), m_seed_earth=float(m0[0]),
                                             flux_factor=FLUX_FACTOR, iso_factor=ISO_FACTOR, tau_s=TAU_S,
                                             torque_scale=TORQUE_SCALE, dt_yr=100.0))
        summary["checks"].setdefault("forward_model", {})[n] = {"a_end": float(fwd[n]["a"][-1]),
                                                             "a_end_fig1": float(load(f"fig1_{n}")[1].min()),
                                                             "m_end": float(fwd[n]["m"][-1])}
    summary["checks"]["snapshots"] = snapshot_check(summary["tracks"])
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    figure(clocks, fwd, gm, th, vh, starts)
    wobble_figure(clocks["jupiter"])
    print(json.dumps(summary["checks"], indent=1, default=float))


def figure(clocks, fwd, gm, th, vh, starts):
    cols = {"jupiter": "#d62728", "saturn": "#ff7f0e", "uranus": "#2ca02c", "neptune": "#1f77b4"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for n, c in cols.items():
        m, a, g = load(f"fig1_{n}", smooth=False)
        ax[0, 0].plot(a, m, color=c, lw=4, alpha=0.3, label=f"{n}: her Fig. 1")
        ax[0, 0].plot(fwd[n]["a"], fwd[n]["m"], color=c, lw=1.2, ls="--")
        r = clocks[n]
        ax[1, 0].plot((r["t"] - 2.3e6) / 1e3, r["m"], color=c, lw=1.5, label=f"{n} (start {starts[n] / 1e6:.3f} Myr)")
        ax[1, 0].axhline(FINAL_MASS_FIG1[n], color=c, lw=0.6, ls=":")
    ax[0, 0].plot([], [], "k--", lw=1.2, label="forward model (migration integrated)")
    ax[0, 0].set(yscale="log", xlabel="a [au]", ylabel="m [M_E]", xlim=(0, 30), ylim=(1e-2, 1e3),
                 title="(a) Paths: her Fig. 1 (thick), physics alone (dashed)")
    ax[0, 0].legend(fontsize=8)
    rj = clocks["jupiter"]
    tm = np.interp(gm, rj["m"], rj["t"])
    i5 = np.argmin(abs(gm - 5))
    ax[0, 1].plot(gm, (th - th[i5]) / 1e3, color="#d62728", lw=4, alpha=0.3, label="her clock: Figs. 1 + 13")
    tmod = np.interp(gm, rj["model_m"], rj["model_t"])
    ax[0, 1].plot(gm, (tmod - tmod[i5]) / 1e3, color="k", lw=1.2, label="growth model alone, along her path")
    ax[0, 1].plot(gm, (tm - tm[i5]) / 1e3, color="#1f77b4", lw=1.0, ls="--",
                  label=f"N-body track (hers on {HER_CLOCK[0]:g}-{HER_CLOCK[1]:g} M_E)")
    ax[0, 1].set(xscale="log", xlabel="m_J [M_E]", ylabel="t - t(5 M_E) [kyr]", title="(b) Jupiter's clock")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].set(yscale="log", xlabel="t - 2.3 Myr [kyr]", ylabel="m [M_E]", ylim=(1e-2, 1e3),
                 title="(c) N-body track clocks (dotted: her final masses)")
    ax[1, 0].legend(fontsize=8)
    v_model = np.abs(np.gradient(np.interp(gm, rj["m"], rj["a"]), np.interp(gm, rj["m"], rj["t"]))) * 1e6
    for lab, k, c in [("nominal", 1.0, "#1f77b4"), ("x0.25", 0.25, "#2ca02c"), ("x0.15", 0.125, "#d62728"),
                      ("x2.00", 2.0, "#9467bd")]:
        mr, vr, _ = np.loadtxt(DIG / f"fig13_{lab}.csv", delimiter=",", skiprows=1).T
        o = np.argsort(mr)
        ax[1, 1].plot(mr[o], vr[o], color=c, lw=4, alpha=0.3, label=f"her Fig. 13 {lab}")
        ax[1, 1].plot(gm, k * v_model, color=c, lw=1.2)
    ax[1, 1].set(xscale="log", yscale="log", xlabel="m_J [M_E]", ylabel="|da/dt| [au/Myr]", xlim=(0.1, 100),
                 ylim=(0.1, 1000), title="(d) Rate: her Fig. 13 (thick), tracks (thin)")
    ax[1, 1].legend(fontsize=8)
    fig.savefig(REPORT / "verification.png", dpi=140)


def snapshot_check(tracks):
    """Tracks against the planet dots of her Figs. 4/5 (one nominal simulation): position and timing."""
    snap = np.loadtxt(DIG / "fig4_planets.csv", delimiter=",", skiprows=1)
    names = ["jupiter", "saturn", "uranus", "neptune"]
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    cols = {"jupiter": "#d62728", "saturn": "#ff7f0e", "uranus": "#2ca02c", "neptune": "#1f77b4"}
    for k, n in enumerate(names):
        t, a, m = np.loadtxt(OUT_TRACKS / tracks[f"nominal_{n}"], delimiter=",", skiprows=1).T
        rows = []
        for tm, ah in zip(snap[:, 0], snap[:, k + 1]):
            ours = float(np.interp(tm * 1e6, t, a))
            moving = a[0] - 0.3 > ah > a[-1] + 0.1
            dt = float((t[np.argmax(a < ah)] / 1e6 - tm) * 1e3) if moving else None
            rows.append({"t_myr": tm, "her_a": ah, "our_a": ours, "da": ours - ah, "timing_kyr": dt})
        out[n] = rows
        ax.plot(t / 1e6, a, color=cols[n], lw=1.3, label=f"{n} track")
        ax.plot(snap[:, 0], snap[:, k + 1], "o", color=cols[n], mec="k", ms=6)
    ax.plot([], [], "ko", label="her Fig. 4 snapshots")
    ax.set(xlim=(2.25, 3.02), xlabel="t [Myr]", ylabel="a [au]", title="Tracks vs the planets in her Fig. 4")
    ax.legend(fontsize=8)
    fig.savefig(REPORT / "snapshots.png", dpi=140)
    return out


def wobble_figure(r):
    """Why the tracks' |adot| wobbles at 20-40 M_E: adot = (da/dm)(dm/dt)."""
    pm, pa = r["path"]
    mraw, araw, _ = load("fig1_jupiter", smooth=False)
    mr, vr, _ = np.loadtxt(DIG / "fig13_nominal.csv", delimiter=",", skiprows=1).T
    o = np.argsort(mr)
    G = np.logspace(np.log10(8), np.log10(80), 200)
    slope = np.abs(np.gradient(np.interp(G, pm, pa), G))
    slope_raw = np.abs(np.gradient(np.interp(G, mraw, araw), G))
    mm = np.maximum.accumulate(r["m"])
    mdot = np.gradient(G, np.interp(G, mm, r["t"])) * 1e6
    v13 = 10 ** np.interp(np.log10(G), np.log10(mr[o]), np.log10(vr[o]))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    ax[0].plot(G, slope_raw, color="#d62728", lw=3, alpha=0.4, label="her Fig. 1 path")
    ax[0].plot(G, slope, "k", lw=1, label="track path (smoothed)")
    ax[0].set(xscale="log", xlabel="m_J [M_E]", ylabel="|da/dm| [au/M_E]", title="(a) Path slope: two kinks in her path")
    ax[1].plot(G, v13 / slope_raw, color="#d62728", lw=3, alpha=0.4, label="hers implied: |adot|_Fig13 / |da/dm|_Fig1")
    ax[1].plot(G, mdot, "k", lw=1, label="model clock")
    ax[1].axhline(0.8 * float(pt.gasdisc.mdot(2.55e6)) / pt.M_EARTH * 1e6, color="0.5", ls=":", lw=1,
                  label="0.8 Mdot* at 2.55 Myr (BLJ15 cap)")
    ax[1].set(xscale="log", yscale="log", xlabel="m_J [M_E]", ylabel="dm/dt [M_E/Myr]", title="(b) Growth rate")
    ax[2].plot(mr[o], vr[o], "o-", color="#1f77b4", ms=3, lw=3, alpha=0.4, label="her Fig. 13 (42 vertices)")
    ax[2].plot(G, slope * mdot, "k", lw=1, label="track: |da/dm| dm/dt")
    ax[2].set(xscale="log", xlim=(8, 80), ylim=(0, 140), xlabel="m_J [M_E]", ylabel="|adot| [au/Myr]",
              title="(c) Migration rate")
    for a_ in ax:
        a_.axvspan(20.5, 24, color="0.9", zorder=0)
        a_.axvspan(35, 42, color="0.9", zorder=0)
        a_.legend(fontsize=7.5)
    fig.savefig(REPORT / "wobble.png", dpi=140)


if __name__ == "__main__":
    main()
