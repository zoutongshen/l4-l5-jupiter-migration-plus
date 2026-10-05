"""Stage 3: do L4 and L5 differ in e, I and D, and does that explain L5's higher escape rate?

Two questions, one script. Reads a merged run and the catalogue, writes to
campaign/reports/<run_id>/:

  fig_ecdf.png           empirical CDFs of L4 and L5 in proper e, proper I and libration
                         half-amplitude D, for three samples, with the KS distance marked
  ks_tests.csv           KS and Anderson-Darling (k-sample) for each sample and variable,
                         proper and osculating
  fig_composition.png    the L5/L4 escape ratio observed, and the ratio expected if each
                         swarm escaped at the pooled rate for its own mix of (e, D, I)
  ic_reweighting.csv     the numbers behind fig_composition (left panel)
  fig_cold_l4.png        L4 and L5 counts per proper-inclination bin and their ratio
                         (complete sample, families removed)
  ratio_by_inclination.csv  N4/N5 in bins of proper inclination, complete sample with and
                         without families: where in inclination the number excess sits
  escape_split.csv       escapes split into objects every label calls a Trojan and objects
                         only V24's element box admits (not JPL TJN, not on the MPC list;
                         comet-like, T_J 2.3-2.8, e > 0.3) (fig_composition, right panel)

The KS test compares two cumulative distributions: its statistic is the largest vertical
gap between the two empirical CDFs, which is why the CDFs, not histograms, are plotted.
Anderson-Darling weighs the tails more, where escapers live; its p-value comes from
permutations.

Samples
  well determined  V24-selected, U <= 4 (docs/CATALOGUE_DEFINITIONS.md)
  complete         V24-selected, H <= 13.6: every orbit is good and discovery bias is least
  complete, bg     the same without V24 family members
Variables
  e_p, I_p   V24 synthetic proper elements (I_p = arcsin sin I_p), where V24 has the object
  e, inc     osculating heliocentric at t0
  D          libration half-amplitude, (max sigma - min sigma) / 2 over the first 10 kyr
             window of the run (as s3_explore.py); ~180 deg means already circulating

Composition test (indirect standardisation). Cut (e, D, I) into strata; the pooled L4+L5
escape fraction in each stratum, applied to each swarm's own counts, gives the escapes
that swarm would have if the two swarms escaped identically at fixed (e, D, I). If the
expected ratio L5/L4 matches the observed one, L5 escapes more only because it starts
in different places. Intervals come from bootstrapping objects.

    python campaign/scripts/s3_distributions.py --run-id pilot_10myr
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from trojans import paths

SWARMS = ("L4", "L5")
CATALOGUE_ID = "2026-09-15"
N_PERMUTATIONS = 999
N_BOOTSTRAP = 2000
SEED = 20260930

# (column, label, x-limits for the plot)
PANELS = [("e_p", "proper eccentricity e_p", (0, 0.25)),
          ("I_p", "proper inclination I_p [deg]", (0, 45)),
          ("D", "libration half-amplitude D [deg]", (0, 50))]
KS_VARIABLES = ["e_p", "I_p", "D", "e", "inc"]
STRATA = {
    "e, D": {"e": [0, 0.1, 0.15, 0.25, 1], "D": [0, 20, 30, 60, 181]},
    "I": {"inc": [0, 10, 20, 30, 90]},
    "e, D, I": {"e": [0, 0.1, 0.15, 0.25, 1], "D": [0, 20, 30, 60, 181], "inc": [0, 20, 30, 90]},
}

# Reference palette (dataviz skill), as in s3_report.py.
SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
SURFACE, TEXT, TEXT_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def load(run_id):
    merged = paths.merged_dir(run_id)
    ev = pd.read_parquet(merged / "events.parquet")
    with h5py.File(merged / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(ev["spkid"])
        n = f["n_samples"][:][:, order]
        smin = f["sigma_min"][:][:, order]
        smax = f["sigma_max"][:][:, order]
    sampled = n > 0
    first = np.argmax(sampled, axis=0)
    cols = np.arange(n.shape[1])
    d0 = np.where(sampled.any(axis=0), (smax[first, cols] - smin[first, cols]) / 2, np.nan)

    master = pd.read_parquet(paths.catalogue_dir(CATALOGUE_ID) / "master_table.parquet",
                             columns=["spkid", "v24_e_p", "v24_sini_p"])
    t = pd.DataFrame({
        "spkid": ev["spkid"], "swarm": ev["swarm"], "H": ev["H"], "e": ev["e"], "inc": ev["inc"], "D": d0,
        "v24": ev["v24_selected"], "complete": ev["complete"], "family": ev["v24_family"].notna(),
        "v24_only": ev["label_region"] == "V24",
        "well_determined": pd.to_numeric(ev["condition_code"], errors="coerce") <= 4,
        "esc": ev["t_hill_record"].notna(),
    }).merge(master, on="spkid", how="left")
    t["e_p"] = t["v24_e_p"]
    t["I_p"] = np.degrees(np.arcsin(t["v24_sini_p"].clip(-1, 1)))
    return t[t["v24"]].reset_index(drop=True)


def sample_masks(t):
    return {
        "well determined (U ≤ 4)": t["well_determined"],
        "complete (H ≤ 13.6)": t["complete"],
        "complete, families removed": t["complete"] & ~t["family"],
    }


def ks_row(a, b, rng):
    ks = stats.ks_2samp(a, b)
    ad = stats.anderson_ksamp([a, b], variant="midrank", method=stats.PermutationMethod(n_resamples=N_PERMUTATIONS, random_state=rng))
    return {"n_L4": len(a), "n_L5": len(b), "median_L4": np.median(a), "median_L5": np.median(b),
            "ks_D": ks.statistic, "ks_p": ks.pvalue, "ks_at": ks.statistic_location,
            "ks_sign": ks.statistic_sign, "ad_stat": ad.statistic, "ad_p": ad.pvalue}


def ks_table(t, rng) -> pd.DataFrame:
    rows = []
    for name, mask in sample_masks(t).items():
        s = t[mask]
        for var in KS_VARIABLES:
            a = s.loc[s["swarm"] == "L4", var].dropna().to_numpy()
            b = s.loc[s["swarm"] == "L5", var].dropna().to_numpy()
            rows.append({"sample": name, "variable": var, **ks_row(a, b, rng)})
    return pd.DataFrame(rows)


def _p_text(p, floor=None):
    if floor is not None and p <= floor:
        return f"< {floor:.0e}".replace("e-0", "e-")
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.0e}".replace("e-0", "e-")


def fig_ecdf(t, ks, out):
    masks = sample_masks(t)
    fig, axes = plt.subplots(len(masks), len(PANELS), figsize=(10.5, 8.4), constrained_layout=True)
    for r, (name, mask) in enumerate(masks.items()):
        s = t[mask]
        for c, (var, label, xlim) in enumerate(PANELS):
            ax = axes[r, c]
            row = ks[(ks["sample"] == name) & (ks["variable"] == var)].iloc[0]
            for sw in SWARMS:
                x = np.sort(s.loc[s["swarm"] == sw, var].dropna().to_numpy())
                y = np.arange(1, x.size + 1) / x.size
                ax.step(np.r_[0, x], np.r_[0, y], where="post", color=SWARM_COLOR[sw], lw=1.8,
                        label=f"{sw} (n = {x.size:,})")
            # The KS distance: the largest vertical gap between the two curves.
            at = row["ks_at"]
            y4 = np.searchsorted(np.sort(s.loc[s["swarm"] == "L4", var].dropna()), at, side="right") / row["n_L4"]
            y5 = np.searchsorted(np.sort(s.loc[s["swarm"] == "L5", var].dropna()), at, side="right") / row["n_L5"]
            ax.plot([at, at], [y4, y5], color=TEXT, lw=2.4, solid_capstyle="butt")
            ax.annotate(f"KS D = {row['ks_D']:.3f}\np(KS) = {_p_text(row['ks_p'])}\n"
                        f"p(AD) = {_p_text(row['ad_p'], floor=1 / (N_PERMUTATIONS + 1))}",
                        (0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=8, color=TEXT_2)
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1.02)
            if r == len(masks) - 1:
                ax.set_xlabel(label)
            if c == 0:
                ax.set_ylabel(f"{name}\ncumulative fraction")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("L4 vs L5 starting distributions: empirical CDFs; the black bar is the KS distance",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(out / "fig_ecdf.png", dpi=200)
    plt.close(fig)


def stratum(t, edges: dict) -> pd.Series:
    parts = [pd.cut(t[col], bins, right=False, labels=False, include_lowest=True).astype("Int64").astype(str)
             for col, bins in edges.items()]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def expected_ratio(t, key):
    """Observed and composition-expected escape fractions per swarm, pooled rates per stratum."""
    rate = t.groupby(key)["esc"].mean()
    exp = t.assign(p=key.map(rate)).groupby("swarm")["p"].sum()
    obs = t.groupby("swarm")["esc"].sum()
    n = t.groupby("swarm").size()
    return {
        "obs_f_L4": obs["L4"] / n["L4"], "obs_f_L5": obs["L5"] / n["L5"],
        "exp_f_L4": exp["L4"] / n["L4"], "exp_f_L5": exp["L5"] / n["L5"],
        "obs_ratio": (obs["L5"] / n["L5"]) / (obs["L4"] / n["L4"]),
        "exp_ratio": (exp["L5"] / n["L5"]) / (exp["L4"] / n["L4"]),
        "obs_k_L4": int(obs["L4"]), "obs_k_L5": int(obs["L5"]), "exp_k_L4": exp["L4"], "exp_k_L5": exp["L5"],
    }


def composition_table(t, rng) -> pd.DataFrame:
    s = t[t["well_determined"]].dropna(subset=["e", "D", "inc"]).reset_index(drop=True)
    rows = []
    for name, edges in STRATA.items():
        key = stratum(s, edges)
        point = expected_ratio(s, key)
        boots = {"obs_ratio": [], "exp_ratio": []}
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, len(s), len(s))
            b = s.iloc[idx]
            r = expected_ratio(b.reset_index(drop=True), key.iloc[idx].reset_index(drop=True))
            for k in boots:
                boots[k].append(r[k])
        row = {"strata": name, "n_strata": key.nunique(), **point}
        for k, v in boots.items():
            v = np.array(v)
            v = v[np.isfinite(v)]
            row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(v, [16, 84])
        # Share of L5's excess that composition accounts for, on a log scale.
        row["explained_share"] = np.log(point["exp_ratio"]) / np.log(point["obs_ratio"])
        rows.append(row)
    return pd.DataFrame(rows)


def wilson(k, n, z=1.0):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def escape_split(t) -> pd.DataFrame:
    """Escapes from real Trojans and from V24-box-only interlopers, per swarm (well-determined sample)."""
    s = t[t["well_determined"]]
    n_swarm = s.groupby("swarm").size()
    rows = []
    for group, mask in [("Trojan by V24 and JPL/MPC", ~s["v24_only"]), ("V24 box only (interloper)", s["v24_only"])]:
        for sw in SWARMS:
            g = s[mask & (s["swarm"] == sw)]
            k, n = int(g["esc"].sum()), len(g)
            f, lo, hi = wilson(k, n)
            rows.append({"group": group, "swarm": sw, "n": n, "k": k, "f": f, "lo": lo, "hi": hi,
                         "median_e": g["e"].median(), "per_10k_of_swarm": 1e4 * k / n_swarm[sw]})
    return pd.DataFrame(rows)


def ratio_by_inclination(t) -> pd.DataFrame:
    """N4/N5 per proper-inclination bin in the complete sample (Poisson errors)."""
    rows = []
    for name, mask in [("complete", t["complete"]), ("complete, families removed", t["complete"] & ~t["family"])]:
        s = t[mask & ~t["v24_only"] & t["I_p"].notna()]
        for lo, hi in [(0, 15), (15, 90), (0, 10), (10, 20), (20, 30), (30, 90)]:
            g = s[(s["I_p"] >= lo) & (s["I_p"] < hi)]
            n4, n5 = int((g["swarm"] == "L4").sum()), int((g["swarm"] == "L5").sum())
            r = n4 / n5
            rows.append({"sample": name, "I_lo": lo, "I_hi": hi, "n_L4": n4, "n_L5": n5,
                         "ratio": r, "err": r * np.sqrt(1 / n4 + 1 / n5)})
    return pd.DataFrame(rows)


def fig_cold_l4(t, out):
    s = t[t["complete"] & ~t["family"] & ~t["v24_only"] & t["I_p"].notna()]
    fine = np.arange(0, 47.5, 2.5)
    coarse = np.array([0, 5, 10, 15, 20, 25, 30, 47.5])
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, height_ratios=[3, 2],
                                   constrained_layout=True)
    h = {sw: np.histogram(s.loc[s["swarm"] == sw, "I_p"], fine)[0] for sw in SWARMS}
    top.fill_between(fine[:-1], h["L5"], np.maximum(h["L4"], h["L5"]), step="post", color=SWARM_COLOR["L4"],
                     alpha=0.18, lw=0, label="L4 excess")
    for sw in SWARMS:
        top.step(fine[:-1], h[sw], where="post", color=SWARM_COLOR[sw], lw=2,
                 label=f"{sw} (n = {int((s['swarm'] == sw).sum()):,})")
    top.set_ylabel("objects per 2.5°")
    top.set_title("Complete sample (H ≤ 13.6), families removed: where the extra L4 Trojans are", loc="left")
    top.legend(loc="upper right")
    top.set_ylim(bottom=0)

    c4 = np.histogram(s.loc[s["swarm"] == "L4", "I_p"], coarse)[0]
    c5 = np.histogram(s.loc[s["swarm"] == "L5", "I_p"], coarse)[0]
    r = c4 / c5
    err = r * np.sqrt(1 / c4 + 1 / c5)
    mid = (coarse[:-1] + coarse[1:]) / 2
    bot.errorbar(mid, r, yerr=err, xerr=np.diff(coarse) / 2, fmt="o", ms=6, color=TEXT_2, elinewidth=1.2, capsize=0)
    bot.axhline(1, color=BASELINE, lw=1.2, ls="--")
    overall = (s["swarm"] == "L4").sum() / (s["swarm"] == "L5").sum()
    bot.axhline(overall, color=SWARM_COLOR["L4"], lw=1, ls=":")
    bot.annotate(f"all inclinations: {overall:.2f}", (46, overall), xytext=(0, 4), textcoords="offset points",
                 ha="right", color=TEXT_2, fontsize=8)
    bot.annotate("equal", (46, 1), xytext=(0, -11), textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    bot.set_ylabel("N4 / N5")
    bot.set_xlabel("proper inclination I_p [deg] (V24)")
    bot.set_xlim(0, 47.5)
    bot.set_ylim(0, max(r + err) * 1.1)
    fig.savefig(out / "fig_cold_l4.png", dpi=200)
    plt.close(fig)


def fig_composition(comp, split, out):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), width_ratios=[1.25, 1], constrained_layout=True)
    labels = ["observed"] + [f"expected from {s}" for s in comp["strata"]]
    y = np.arange(len(labels))[::-1]
    obs = comp.iloc[0]
    vals = [obs["obs_ratio"]] + list(comp["exp_ratio"])
    lo = [obs["obs_ratio_lo"]] + list(comp["exp_ratio_lo"])
    hi = [obs["obs_ratio_hi"]] + list(comp["exp_ratio_hi"])
    for yi, v, l, h, lab in zip(y, vals, lo, hi, labels):
        color = TEXT if lab == "observed" else TEXT_2
        ax.plot([l, h], [yi, yi], color=color, lw=1.6)
        ax.plot(v, yi, "o", ms=8, color=color, mfc=color if lab == "observed" else SURFACE, mew=1.6)
        ax.annotate(f"{v:.2f}", (h, yi), xytext=(6, 0), textcoords="offset points", va="center",
                    color=TEXT_2, fontsize=8)
    ax.axvline(1, color=BASELINE, lw=1.2, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("escape ratio L5 / L4 over 10 Myr (well-determined sample; bar = 68% bootstrap interval)")
    ax.set_title("How much of L5's extra escape is explained by where its members start?", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(hi) * 1.15)

    # Right: where each swarm's escapes come from, per 10,000 members.
    shades = {"Trojan by V24 and JPL/MPC": 1.0, "V24 box only (interloper)": 0.45}
    for yi, sw in zip([1, 0], SWARMS):
        left = 0.0
        for group, alpha in shades.items():
            r = split[(split["group"] == group) & (split["swarm"] == sw)].iloc[0]
            w = r["per_10k_of_swarm"]
            ax2.barh(yi, w, left=left, height=0.55, color=SWARM_COLOR[sw], alpha=alpha,
                     edgecolor=SURFACE, linewidth=2)
            ax2.annotate(f"{r['k']} of {r['n']:,}", (left + w / 2, yi - 0.3), ha="center", va="top", fontsize=8,
                         color=TEXT_2)
            left += w
        ax2.annotate(f"{left:.0f}", (left, yi), xytext=(5, 0), textcoords="offset points", va="center",
                     color=TEXT_2, fontsize=8)
    ax2.set_yticks([1, 0], SWARMS)
    ax2.set_ylim(-0.75, 1.4)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("escapers per 10,000 members in 10 Myr")
    ax2.set_title("dark = Trojan by all labels · light = V24 box only", loc="left")
    fig.savefig(out / "fig_composition.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    t = load(args.run_id)
    out = paths.report_dir(args.run_id)

    ks = ks_table(t, rng)
    ks.to_csv(out / "ks_tests.csv", index=False, float_format="%.5g")
    fig_ecdf(t, ks, out)

    comp = composition_table(t, rng)
    comp.to_csv(out / "ic_reweighting.csv", index=False, float_format="%.5g")
    fig_cold_l4(t, out)
    by_inc = ratio_by_inclination(t)
    by_inc.to_csv(out / "ratio_by_inclination.csv", index=False, float_format="%.4g")
    print(by_inc.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    split = escape_split(t)
    split.to_csv(out / "escape_split.csv", index=False, float_format="%.5g")
    fig_composition(comp, split, out)

    print(ks.drop(columns=["ks_sign"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(split.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
