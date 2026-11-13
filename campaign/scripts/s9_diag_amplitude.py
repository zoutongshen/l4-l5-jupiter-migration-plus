"""Stage 9 diagnostic (run on haagsebeek: python campaign/scripts/s9_diag_amplitude.py <run_id> ...). Libration amplitude of the Trojans present at 3.0 Myr (gas gone), from sigma over the 3.0-3.5 Myr snapshots
(10 kyr apart, ~70 libration periods, so the samples cover the cycle): A = max |sigma - sigma_eq|, sigma_eq = +-60.
Split by side and by survival to 5 Myr."""
import sys, numpy as np, h5py
sys.path.insert(0, "campaign")
from trojans import disc, paths
for run in sys.argv[1:]:
    rows = []
    for ch in sorted((paths.run_dir(run) / "chunks").iterdir()):
        with h5py.File(ch / "snapshots.h5", "r") as f:
            t = f["t"][:]
            i0 = int(np.abs(t - 3.0e6).argmin()); sel = np.flatnonzero((t >= 3.0e6 - 1) & (t <= 3.5e6 + 1))
            aj, mj = f["jupiter_a"][i0], f["jupiter_m"][i0]; rh = disc.hill_radius(aj, mj)
            a, e, s = f["a"][i0], f["e"][i0], f["sigma"][i0]
            tr = np.isfinite(a) & (a * (1 - e) >= aj - 2.5 * rh) & (a * (1 + e) <= aj + 2.5 * rh)
            S = np.array([f["sigma"][k] for k in sel])
            aend = f["a"][len(t) - 1]
            for j in np.flatnonzero(tr):
                side = 1 if s[j] > 0 else -1
                sg = S[:, j][np.isfinite(S[:, j])]
                amp = np.max(np.abs(sg - 60 * side)) if len(sg) else np.nan
                flips = np.any(np.sign(sg) != side)
                rows.append((side, amp, flips, np.isfinite(aend[j]), e[j]))
    r = np.array(rows, dtype=float)
    print(f"\n== {run}: Trojans at 3.0 Myr ==")
    for side, lab in ((1, "L4"), (-1, "L5")):
        m = r[:, 0] == side
        A = r[m, 1]; surv = r[m, 3] == 1
        print(f" {lab}: N={m.sum():4d}  amplitude median {np.nanmedian(A):5.1f} deg, quartiles {np.nanpercentile(A,25):5.1f}/{np.nanpercentile(A,75):5.1f};"
              f"  >60 deg (near separatrix): {(A > 60).sum():3d};  side-flippers (horseshoe): {int(r[m,2].sum()):3d};  e median {np.median(r[m,4]):.3f}")
        print(f"      survive to 5 Myr: {surv.sum()}/{m.sum()}  | amplitude median of survivors {np.nanmedian(A[surv]):5.1f}, of the lost {np.nanmedian(A[~surv]) if (~surv).any() else float('nan'):5.1f}")
