"""Group-meeting figures for stage 8 (Blocks C-D): rates, gap mass, counts, CDFs, ratios.

Reads campaign/reports/<run>/ (particles.parquet, coorbital.csv, reweighted.csv), the
Jupiter track headers in configs/tracks/, and the observed catalogue (complete sample,
H <= 13.6, V24 families removed, proper I). Writes PNGs and one combined PDF:

    python campaign/scripts/s8_summary_figures.py --out updates/11102026
"""

import argparse
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from trojans import paths  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "data" / "catalogue" / "2026-09-15_rebuilt" / "master_table.parquet"

L4C, L5C = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NU_COLOR = {0.225: "#86b6ef", 0.45: "#3987e5", 0.65: "#1c5cab", 0.9: "#0d366b"}   # ordinal blue ramp
GAP_MARKER = {12.5: "v", 25.0: "o", 50.0: "^"}
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
I_EDGES = [0, 10, 20, 30, 90]
I_LABELS = ["0–10°", "10–20°", "20–30°", ">30°"]
OBS_RATIO = np.array([2.15, 1.38, 0.93, 1.12])
OBS_ERR = np.array([0.14, 0.09, 0.08, 0.21])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})

# run id -> short label, disc description
RUNS = {
    "mig_c0b_cold_zone": ("C0b", "cold"), "mig_c1_nu045_cold": ("C1", "cold"),
    "mig_c2_nu09_cold": ("C2", "cold"), "mig_d2_mgap12": ("D2", "cold"),
    "mig_d3_mgap50_nu0225": ("D3", "cold"), "mig_d6_c0b_hot": ("D6", "e0≤0.05"),
    "mig_d5_c1_hot": ("D5", "e0≤0.05"), "mig_d1_c2_hot": ("D1", "e0≤0.05"),
    "mig_d9_nu065": ("D9", "e0≤0.1"), "mig_d8_c1_warm": ("D8", "e0≤0.3"),
    "mig_d7_c2_warm": ("D7", "e0≤0.3"),
}
I_RUNS = ["mig_d6_c0b_hot", "mig_d5_c1_hot", "mig_d8_c1_warm", "mig_d9_nu065", "mig_d1_c2_hot", "mig_d7_c2_warm"]


def track_params(run_id):
    cfg = tomllib.loads((CONFIGS / f"{run_id}.toml").read_text())
    path = CONFIGS / cfg["jupiter"]["track_csv"]
    head = path.read_text().splitlines()[0].split("#", 1)[1].split()
    p = {k: float(v) for k, v in (kv.split("=") for kv in head if "=" in kv and kv.split("=")[1][0].isdigit())}
    tr = pd.read_csv(path, skiprows=1, names=["t", "a", "m"])
    return p, tr


def nu_of_m(p, m):
    return p["nu_I"] / (1 + (np.asarray(m) / p["m_gap_earth"]) ** p["gap_power"])


def trojans(run_id):
    t = pd.read_parquet(paths.report_dir(run_id) / "particles.parquet")
    return t[t["swarm"].isin(["L4", "L5"])]


def ratio(n4, n5):
    n4, n5 = np.asarray(n4, float), np.asarray(n5, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = n4 / n5
        return r, r * np.sqrt(1 / n4 + 1 / n5)


def binned_ratio(t, col="inc", edges=I_EDGES):
    b = pd.cut(t[col], edges, right=False)
    n4 = t[t.swarm == "L4"].groupby(b[t.swarm == "L4"], observed=False).size().to_numpy()
    n5 = t[t.swarm == "L5"].groupby(b[t.swarm == "L5"], observed=False).size().to_numpy()
    return (*ratio(n4, n5), n4, n5)


def observed():
    m = pd.read_parquet(CATALOGUE, columns=["swarm", "v24_selected", "complete", "v24_family", "label_region",
                                            "v24_sini_p"])
    s = m[m.v24_selected & m.complete & m.v24_family.isna() & (m.label_region != "V24") & m.v24_sini_p.notna()]
    return pd.DataFrame({"swarm": s.swarm, "inc": np.degrees(np.arcsin(s.v24_sini_p.clip(-1, 1)))})


def label(run_id, p):
    return f"{RUNS[run_id][0]}  ν_I {p['nu_I']:g}, M_gap {p['m_gap_earth']:g}, {RUNS[run_id][1]}"


# ---------------------------------------------------------------- figures

def fig_tracks():
    runs = ["mig_c0b_cold_zone", "mig_d3_mgap50_nu0225", "mig_c1_nu045_cold", "mig_d9_nu065",
            "mig_c2_nu09_cold", "mig_d2_mgap12"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    m = np.logspace(-2, np.log10(318), 400)
    for r in runs:
        p, tr = track_params(r)
        c, ls = NU_COLOR[p["nu_I"]], {25.0: "-", 12.5: ":", 50.0: "--"}[p["m_gap_earth"]]
        lab = f"{RUNS[r][0]}: ν_I {p['nu_I']:g}, M_gap {p['m_gap_earth']:g}"
        ax1.plot(m, nu_of_m(p, m), color=c, ls=ls, label=lab)
        ax2.plot(tr.t / 1e3, tr.a, color=c, ls=ls)
        ax3.plot(tr.t / 1e3, tr.m, color=c, ls=ls)
    ax1.axvline(20, color=MUTED, lw=1)
    ax1.text(21, 0.93, "20 M⊕ (end of\npebble phase)", color=TEXT_2, fontsize=9, va="top")
    ax1.axhline(0.17, color=MUTED, lw=1, ls="--")
    ax1.text(0.012, 0.125, "Pirani v_lim = 0.17", color=TEXT_2, fontsize=9)
    ax1.set(xscale="log", xlabel="Jupiter mass [M⊕]", ylabel="ν = |ȧ| / (qΩa)", ylim=(0, 1.0),
            title="(a) Migration rate vs mass: ν = ν_I / (1 + (m/M_gap)³)")
    ax2.legend(*ax1.get_legend_handles_labels(), fontsize=8.5, loc="upper right")
    ax2.set(xlabel="t [kyr]", ylabel="a_J [au]", title="(b) Jupiter's distance (all end at 5.4 au)")
    ax3.set(xlabel="t [kyr]", ylabel="m_J [M⊕]", yscale="log", title="(c) Jupiter's mass (same in all runs)")
    ax3.axhline(20, color=MUTED, lw=1)
    return fig


def fig_ratio_vs_nu():
    """Common subset of every run: e0 < 0.05, i0 < 15 deg (overall and e0 < 0.025)."""
    rows = []
    for r in RUNS:
        p, _ = track_params(r)
        t = trojans(r)
        for name, sel in [("e0 < 0.05, i0 < 15°", (t.e0 < 0.05) & (t.i0 < 15)),
                          ("e0 < 0.025, i0 < 15°", (t.e0 < 0.025) & (t.i0 < 15))]:
            g = t[sel]
            rr, ee = ratio((g.swarm == "L4").sum(), (g.swarm == "L5").sum())
            rows.append(dict(run=RUNS[r][0], sub=name, nu_I=p["nu_I"], m_gap=p["m_gap_earth"],
                             nu10=nu_of_m(p, 10), nu20=nu_of_m(p, 20), nu30=nu_of_m(p, 30), r=rr, e=ee))
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharey="row", constrained_layout=True)
    for i, sub in enumerate(d["sub"].unique()):
        s = d[d["sub"] == sub]
        for j, (col, xl) in enumerate([("nu_I", "ν_I (type I rate, disc)"), ("nu10", "ν at 10 M⊕"),
                                       ("nu20", "ν at 20 M⊕"), ("nu30", "ν at 30 M⊕")]):
            ax = axes[i, j]
            for _, q in s.iterrows():
                ax.errorbar(q[col], q.r, yerr=q.e, fmt=GAP_MARKER[q.m_gap], ms=8, color=NU_COLOR[q.nu_I],
                            mec=SURFACE, mew=1, elinewidth=1, capsize=0)
                if q.run in ("D2", "D3"):
                    ax.annotate(q.run, (q[col], q.r), xytext=(6, -3), textcoords="offset points", fontsize=8.5,
                                color=TEXT_2)
            ax.axhline(1, color=MUTED, lw=1)
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(f"N4/N5\n{sub}")
            if i == 0:
                ax.set_title(["(a) by ν_I", "(b) by ν(10 M⊕)", "(c) by ν(20 M⊕)", "(d) by ν(30 M⊕)"][j])
    for lab, mk in [("M_gap 12.5", "v"), ("M_gap 25", "o"), ("M_gap 50", "^")]:
        axes[0, 0].plot([], [], mk, color=TEXT_2, ms=7, ls="none", label=lab)
    for nu, c in NU_COLOR.items():
        axes[0, 0].plot([], [], "s", color=c, ms=7, ls="none", label=f"ν_I {nu:g}")
    axes[0, 0].legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("Cold-start excess vs migration rate, all 11 runs on their common subset", x=0.01, ha="left",
                 fontsize=12.5)
    return fig, d


def fig_counts(obs):
    runs = ["mig_d6_c0b_hot", "mig_d8_c1_warm", "mig_d7_c2_warm"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    edges = np.arange(0, 47.5, 2.5)
    for ax, (name, t) in zip(axes, [("Observed (proper I, complete, no families)", obs)] +
                             [(label(r, track_params(r)[0]), trojans(r)) for r in runs]):
        for sw, c in [("L4", L4C), ("L5", L5C)]:
            h = np.histogram(t.loc[t.swarm == sw, "inc"], edges)[0]
            ax.step(edges[:-1], h, where="post", color=c, label=f"{sw} (n = {(t.swarm == sw).sum():,})")
        ax.set(xlabel="final I [°]" if ax is not axes[0] else "proper I [°]", title=name, ylim=(0, None))
        ax.title.set_fontsize(9.5)
        ax.legend(fontsize=8.5)
    axes[0].set_ylabel("objects per 2.5°")
    fig.suptitle("Actual counts by inclination (simulated discs are uniform in i0 to 40°)", x=0.01, ha="left",
                 fontsize=12.5)
    return fig


def fig_cdf(obs):
    runs = ["mig_d6_c0b_hot", "mig_d8_c1_warm", "mig_d9_nu065", "mig_d7_c2_warm"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), sharey=True, constrained_layout=True)
    for ax, (name, t) in zip(axes, [("Observed", obs)] + [(RUNS[r][0] + f"  ν_I {track_params(r)[0]['nu_I']:g}, "
                                                             + RUNS[r][1], trojans(r)) for r in runs]):
        x = {}
        for sw, c in [("L4", L4C), ("L5", L5C)]:
            v = np.sort(t.loc[t.swarm == sw, "inc"].to_numpy())
            x[sw] = v
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=sw)
        grid = np.linspace(0, 50, 1001)
        dmax = np.max(np.abs(np.searchsorted(x["L4"], grid, "right") / len(x["L4"])
                             - np.searchsorted(x["L5"], grid, "right") / len(x["L5"])))
        med4, med5 = np.median(x["L4"]), np.median(x["L5"])
        ax.text(0.97, 0.05, f"KS D = {dmax:.2f}\nmedian L4 {med4:.1f}°\nmedian L5 {med5:.1f}°", ha="right",
                transform=ax.transAxes, fontsize=8.5, color=TEXT_2)
        ax.set(xlabel="I [°]", xlim=(0, 45), title=name)
        ax.title.set_fontsize(9.5)
    axes[0].set_ylabel("CDF")
    axes[0].legend(loc="upper left")
    fig.suptitle("Inclination CDFs per swarm. Observed: L4 colder than L5. Fast migration: L4 hotter.",
                 x=0.01, ha="left", fontsize=12.5)
    return fig


def fig_ratio_by_I():
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(4)
    ax.errorbar(x, OBS_RATIO, OBS_ERR, fmt="D", ms=8, color=TEXT, lw=2.4, capsize=0, label="Observed (proper I)",
                zorder=5)
    ax.plot(x, OBS_RATIO, color=TEXT, lw=2.4, zorder=5)
    for k, r in enumerate(I_RUNS):
        p, _ = track_params(r)
        rr, ee, _, _ = binned_ratio(trojans(r))
        ls = "--" if RUNS[r][1] == "e0≤0.3" else "-"
        off = (k - 2.5) * 0.05
        ax.errorbar(x + off, rr, ee, fmt="o", ms=5, color=NU_COLOR[p["nu_I"]], ls=ls, capsize=0, elinewidth=1,
                    label=label(r, p))
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xticks(x, I_LABELS)
    ax.set(xlabel="final I (simulated) / proper I (observed)", ylabel="N4/N5",
           title="(a) Ratio by inclination: no run is cold-only")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    for r, c in [("mig_d8_c1_warm", NU_COLOR[0.45]), ("mig_d7_c2_warm", NU_COLOR[0.9])]:
        w = pd.read_csv(paths.report_dir(r) / "reweighted.csv")
        for _, row in w.iterrows():
            vals = [float(row[b].split("±")[0]) for b in ["0-10", "10-20", "20-30", ">30"]]
            axr.plot(x, vals, color=c, lw=1.2 if row.model != "uniform" else 2.4,
                     alpha=0.55 if row.model != "uniform" else 1)
        axr.plot([], [], color=c, label=f"{RUNS[r][0]} (ν_I {track_params(r)[0]['nu_I']:g}): 7 disc models")
    axr.errorbar(x, OBS_RATIO, OBS_ERR, fmt="D-", ms=8, color=TEXT, lw=2.4, capsize=0, label="Observed")
    axr.axhline(1, color=MUTED, lw=1)
    axr.set_xticks(x, I_LABELS)
    axr.set(xlabel="final I", title="(b) Reweighted to physical discs (Rayleigh, cold+hot mixes)")
    axr.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    return fig


def fig_maps():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    e_edges, i_edges = [0, 0.05, 0.1, 0.2, 0.3001], [0, 10, 20, 30, 40.01]
    for ax, r in zip(axes, ["mig_d8_c1_warm", "mig_d7_c2_warm"]):
        t = trojans(r)
        be = pd.cut(t.e0, e_edges, right=False, labels=False)
        bi = pd.cut(t.i0, i_edges, right=False, labels=False)
        R, N = np.full((4, 4), np.nan), {}
        for a in range(4):
            for b in range(4):
                g = t[(be == a) & (bi == b)]
                N[a, b] = ((g.swarm == "L4").sum(), (g.swarm == "L5").sum())
                if min(N[a, b]) > 0:
                    R[a, b] = N[a, b][0] / N[a, b][1]
        im = ax.imshow(np.log2(R), cmap=DIV, vmin=-1.3, vmax=1.3, origin="lower", aspect="auto")
        for a in range(4):
            for b in range(4):
                txt = f"{R[a, b]:.2f}" if np.isfinite(R[a, b]) else "–"
                ax.text(b, a + 0.08, txt, ha="center", va="center", fontsize=10.5, color=TEXT)
                ax.text(b, a - 0.25, f"{N[a, b][0]} / {N[a, b][1]}", ha="center", va="center", fontsize=8,
                        color=TEXT_2)
        ax.set_xticks(range(4), ["0–10", "10–20", "20–30", "30–40"])
        ax.set_yticks(range(4), ["0–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.3"])
        ax.grid(False)
        p, _ = track_params(r)
        ax.set(xlabel="i0 [°]", ylabel="e0", title=f"{RUNS[r][0]}: ν_I {p['nu_I']:g}, N4/N5 by starting (e0, i0)")
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_ticks(np.log2([0.5, 0.7, 1, 1.4, 2]), labels=["0.5", "0.7", "1", "1.4", "2"])
    cb.set_label("N4/N5  (blue = L4 excess); small: N4 / N5")
    return fig


def fig_buildup():
    runs = ["mig_c0b_cold_zone", "mig_d3_mgap50_nu0225", "mig_d2_mgap12", "mig_c1_nu045_cold", "mig_c2_nu09_cold"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    for r in runs:
        p, _ = track_params(r)
        c = pd.read_csv(paths.report_dir(r) / "coorbital.csv")
        c = c[c.jupiter_m_earth > 0.05]
        ls = {25.0: "-", 12.5: ":", 50.0: "--"}[p["m_gap_earth"]]
        lab = f"{RUNS[r][0]}: ν_I {p['nu_I']:g}, M_gap {p['m_gap_earth']:g}"
        ax1.plot(c.jupiter_m_earth, c["pos/neg"], color=NU_COLOR[p["nu_I"]], ls=ls, label=lab)
        ax2.plot(c.t_yr / 1e3, c["pos/neg"], color=NU_COLOR[p["nu_I"]], ls=ls)
    ax1.axvline(20, color=MUTED, lw=1)
    ax1.axhline(1, color=MUTED, lw=1)
    ax1.set(xscale="log", xlabel="Jupiter mass [M⊕]", ylabel="co-orbital L4-side / L5-side",
            title="(a) When the asymmetry builds, by Jupiter's mass")
    ax1.legend(fontsize=8.5, loc="upper left")
    ax2.axhline(1, color=MUTED, lw=1)
    ax2.axvline(690, color=MUTED, lw=1)
    ax2.text(700, ax2.get_ylim()[1] * 0.98, "gas gone,\ngiants free", fontsize=8.5, color=TEXT_2, va="top")
    ax2.set(xlabel="t [kyr]", title="(b) Same, by time: frozen after the gas phase")
    return fig


def fig_frozen():
    pairs = [("mig_c2_nu09_cold", "mig_d4_c2_long", "C2 → D4"), ("mig_d7_c2_warm", "mig_d10_d7_long", "D7 → D10"),
             ("mig_d8_c1_warm", "mig_d11_d8_long", "D8 → D11")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
    x = np.arange(4)
    for ax, (s, l, name) in zip(axes, pairs):
        for r, mk, lab in [(s, "o", "1.2 Myr"), (l, "s", "10.7 Myr")]:
            rr, ee, n4, n5 = binned_ratio(trojans(r))
            few = (n4 < 10) | (n5 < 10)
            rr, ee = np.where(few, np.nan, rr), np.where(few, np.nan, ee)
            ax.errorbar(x + (0.08 if mk == "s" else -0.08), rr, ee, fmt=mk + "-", ms=6, capsize=0, elinewidth=1,
                        color=NU_COLOR[track_params(s)[0]["nu_I"]], mfc=SURFACE if mk == "s" else None,
                        label=f"{lab}: {n4.sum()} / {n5.sum()}")
        ax.axhline(1, color=MUTED, lw=1)
        ax.set_xticks(x, I_LABELS)
        ax.set(xlabel="final I", title=f"{name}: giants free for 10 Myr")
        ax.legend(fontsize=8.5, title="time: N4 / N5", title_fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("N4/N5")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "updates" / "11102026")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    obs = observed()
    figs = [("01_tracks_nu_mgap", fig_tracks())]
    f, table = fig_ratio_vs_nu()
    table.to_csv(out / "02_ratio_vs_nu.csv", index=False)
    figs += [("02_ratio_vs_nu", f), ("03_buildup_vs_mass", fig_buildup()), ("04_counts_by_I", fig_counts(obs)),
             ("05_cdf_by_I", fig_cdf(obs)), ("06_ratio_by_I", fig_ratio_by_I()), ("07_e0_i0_maps", fig_maps()),
             ("08_frozen_10myr", fig_frozen())]
    with PdfPages(out / "stage8_figures.pdf") as pdf:
        for name, fig in figs:
            fig.savefig(out / f"{name}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main()
