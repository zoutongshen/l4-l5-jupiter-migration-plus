"""Stage 9 diagnostic (run on haagsebeek: python campaign/scripts/s9_diag_escape.py <run_id> ...). Where do L5 (and L4) Trojans go during 2.40-3.0 Myr, and who refills L5? States per snapshot:
T4/T5 = Trojan (q,Q in 2.5 RH) with sigma >0/<0; C = co-orbital (|a-aJ| < 2.5 RH) but not Trojan;
O = alive, not co-orbital; X = removed. Also prints alpha = |adot|/(2 a n mu) of Jupiter at each time."""
import sys, numpy as np, h5py
sys.path.insert(0, "campaign")
from trojans import disc, paths
TIMES = [2.40, 2.45, 2.50, 2.55, 2.60, 2.70, 2.80, 2.90, 3.00, 5.00]
ME = disc.M_EARTH

def states(run):
    root = paths.run_dir(run)
    ini = np.load(root / "disc_initial.npz"); a0 = dict(zip(ini["id"], ini["a0"]))
    S, A0, J = [], [], {}
    for ch in sorted((root / "chunks").iterdir()):
        with h5py.File(ch / "snapshots.h5", "r") as f:
            tt = f["t"][:]; ids = f["spkid"][:]
            st = np.empty((len(ids), len(TIMES)), dtype="<U1")
            for k, T in enumerate(TIMES):
                i = int(np.abs(tt - T * 1e6).argmin())
                aj, mj = f["jupiter_a"][i], f["jupiter_m"][i]; rh = disc.hill_radius(aj, mj)
                a, e, s = f["a"][i], f["e"][i], f["sigma"][i]
                alive = np.isfinite(a)
                tr = alive & (a * (1 - e) >= aj - 2.5 * rh) & (a * (1 + e) <= aj + 2.5 * rh)
                co = alive & (np.abs(a - aj) < 2.5 * rh)
                st[:, k] = np.where(~alive, "X", np.where(tr & (s > 0), "4", np.where(tr & (s < 0), "5", np.where(co, "C", "O"))))
                J[T] = (aj, mj)
            S.append(st); A0.append([a0[i] for i in ids])
    return np.vstack(S), np.concatenate(A0), J

for run in sys.argv[1:]:
    S, A0, J = states(run)
    print(f"\n===== {run} =====")
    print("t      " + "  ".join(f"{T:5.2f}" for T in TIMES))
    for lab in "45CX":
        print(f"N_{lab}    " + "  ".join(f"{(S[:, k] == lab).sum():5d}" for k in range(len(TIMES))))
    for side in "54":
        m = S[:, 0] == side
        print(f"\n  L{side} Trojans at 2.40 Myr: {m.sum()}")
        print("   t     " + "  ".join(f"{T:5.2f}" for T in TIMES[1:]))
        for lab, name in (("5", "in L5"), ("4", "in L4"), ("C", "co-orb, not Trojan"), ("O", "left co-orbital"), ("X", "removed")):
            print(f"   {name:19s}" + "  ".join(f"{(S[m, k] == lab).sum():5d}" for k in range(1, len(TIMES))))
        ever_left = m & np.any(S[:, 1:6] != side, axis=1)
        back = ever_left & (S[:, TIMES.index(3.00)] == side)
        print(f"   left L{side} at some point by 2.70 Myr: {ever_left.sum()}, of which back in L{side} at 3.0 Myr: {back.sum()}")
    end5 = S[:, -1] == "5"
    print(f"\n  L5 Trojans at 5 Myr: {end5.sum()}; their state at 2.40 Myr: " +
          ", ".join(f"{lab}:{(S[end5, 0] == lab).sum()}" for lab in "45CO"))
    print(f"   ... at 2.60 Myr: " + ", ".join(f"{lab}:{(S[end5, TIMES.index(2.60)] == lab).sum()}" for lab in "45CO"))
    print(f"   origin a0 of 5 Myr L5: median {np.median(A0[end5]):.2f}, outside 17-19.5 au: {((A0[end5] < 17) | (A0[end5] > 19.5)).sum()}")
    end4 = S[:, -1] == "4"
    print(f"  L4 Trojans at 5 Myr: {end4.sum()}; state at 2.40: " + ", ".join(f"{lab}:{(S[end4, 0] == lab).sum()}" for lab in "45CO"))
