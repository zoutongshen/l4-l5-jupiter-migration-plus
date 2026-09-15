"""Validate migration trajectories + plot the 4 batch scenarios.

(1) Jupiter a(t) vs the analytic a0*exp(-t/tau) for each scenario  -> validation
(2) Trojan L4/L5 count evolution                                   -> asymmetry
(3) Final top-down (x-y) Trojan snapshot                           -> structure
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from simulation import compute_jupiter_orbit, compute_trojan_counts
from simulation.config import TrojanCriteria

LOST = 1.0e8
OUT = 'results/png'
os.makedirs(OUT, exist_ok=True)

SCENARIOS = [
    ('inward s42',   'inward_50kyr_seed42',   'tab:blue'),
    ('inward s1042', 'inward_50kyr_seed1042', 'tab:cyan'),
    ('outward s42',  'outward_50kyr_seed42',  'tab:red'),
    ('insitu s42',   'insitu_50kyr_seed42',   'tab:green'),
]

def load(name):
    with open(f'results/pkl/batch/{name}_n10000.pkl', 'rb') as f:
        return pickle.load(f)

def jupiter_a_series(d, jidx):
    mp, mv, mm = np.asarray(d['massive_pos']), np.asarray(d['massive_vel']), np.asarray(d['massive_mass'])
    return np.array([compute_jupiter_orbit(mp[i], mv[i], mm, jupiter_index=jidx)['a']
                     for i in range(mp.shape[0])])

# ---------------------------------------------------------------- validation
print("=== Migration trajectory validation (Jupiter semimajor axis) ===")
data = {}
for label, name, color in SCENARIOS:
    d = load(name)
    cfg = d['config']
    jidx = cfg.get('jupiter_index', 4) if isinstance(cfg, dict) else 4
    t = np.asarray(d['times'], float); t_rel = t - t[0]
    a = jupiter_a_series(d, jidx)
    a0 = a[0]
    # analytic exponential using the tau actually implied by initial/target/duration
    mig = cfg.get('migration') if isinstance(cfg, dict) else None
    tau = None
    if isinstance(mig, dict):
        tau = mig.get('tau_a')
    if tau is None or tau == 0:
        # fall back: derive from endpoints (insitu -> flat)
        if abs(a[-1] - a0) > 1e-4:
            tau = t_rel[-1] / np.log(a0 / a[-1])
        else:
            tau = np.inf
    a_analytic = a0 * np.exp(-t_rel / tau) if np.isfinite(tau) else np.full_like(t_rel, a0)
    rel = np.abs(a - a_analytic) / a_analytic
    data[label] = dict(t_rel=t_rel, a=a, a_analytic=a_analytic, color=color, d=d, jidx=jidx)
    print(f"  {label:13s}: a {a0:.4f} -> {a[-1]:.4f} AU  (tau_a={tau:.0f} yr)  "
          f"max|rel err| vs exp = {rel.max()*100:.3f}%")

# ---------------------------------------------------------------- plot 1: a(t)
fig, ax = plt.subplots(figsize=(8, 5))
for label in data:
    D = data[label]
    ax.plot(D['t_rel']/1e3, D['a'], color=D['color'], lw=2, label=label)
    ax.plot(D['t_rel']/1e3, D['a_analytic'], color=D['color'], lw=1, ls='--', alpha=0.6)
ax.axhline(5.2, color='k', lw=0.7, ls=':', alpha=0.5)
ax.set_xlabel('time since migration start [kyr]')
ax.set_ylabel('Jupiter semimajor axis a [AU]')
ax.set_title('Jupiter migration trajectories (solid=sim, dashed=analytic exp)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/migration_trajectories_4scenarios.png', dpi=130)
print(f"\nsaved {OUT}/migration_trajectories_4scenarios.png")

# ---------------------------------------------------------------- plot 2: L4/L5
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, (label, name, color) in zip(axes.flat, SCENARIOS):
    d = data[label]['d']; t = data[label]['t_rel']/1e3
    l4 = np.asarray(d['l4_counts']); l5 = np.asarray(d['l5_counts'])
    ax.plot(t, l4, 'tab:orange', lw=1.8, label='L4 (leading)')
    ax.plot(t, l5, 'tab:purple', lw=1.8, label='L5 (trailing)')
    ax.set_title(f"{label}   (final L4/L5 = {l4[-1]}/{l5[-1]})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_ylabel('Trojan count')
for ax in axes[-1]:
    ax.set_xlabel('time [kyr]')
fig.suptitle('L4 / L5 Trojan population evolution', y=1.00)
fig.tight_layout(); fig.savefig(f'{OUT}/trojan_evolution_4scenarios.png', dpi=130)
print(f"saved {OUT}/trojan_evolution_4scenarios.png")

# ---------------------------------------------------------------- plot 3: top-down
crit = TrojanCriteria()
def classify_xy(mp, pp, jidx):
    """Replicate compute_trojan_counts per-particle -> boolean L4/L5 masks."""
    jx, jy = mp[jidx, 0], mp[jidx, 1]
    r_j = np.hypot(jx, jy); lam_j = np.arctan2(jy, jx)
    r = np.hypot(pp[:, 0], pp[:, 1])
    lam = np.arctan2(pp[:, 1], pp[:, 0])
    dlam = np.arctan2(np.sin(lam - lam_j), np.cos(lam - lam_j))
    coorb = np.abs(r / r_j - 1) < crit.coorbital_tolerance
    hw = np.deg2rad(crit.sector_half_width)
    isL4 = coorb & (np.abs(dlam - np.deg2rad(crit.l4_angle)) < hw)
    isL5 = coorb & (np.abs(dlam - np.deg2rad(crit.l5_angle)) < hw)
    return isL4, isL5, r_j, lam_j

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, (label, name, color) in zip(axes.flat, SCENARIOS):
    D = data[label]; d = D['d']; jidx = D['jidx']
    mp = np.asarray(d['massive_pos'])[-1]
    pp_all = np.asarray(d['planetesimal_pos'])[-1]
    alive = np.abs(pp_all[:, 0]) < LOST/2
    pp = pp_all[alive]
    isL4, isL5, r_j, lam_j = classify_xy(mp, pp, jidx)
    jx, jy = mp[jidx, 0], mp[jidx, 1]
    other = ~(isL4 | isL5)
    ax.scatter(pp[other, 0], pp[other, 1], s=2, c='lightgray', alpha=0.4, label='other')
    ax.scatter(pp[isL4, 0], pp[isL4, 1], s=10, c='tab:orange', alpha=0.9, label=f'L4 ({isL4.sum()})')
    ax.scatter(pp[isL5, 0], pp[isL5, 1], s=10, c='tab:purple', alpha=0.9, label=f'L5 ({isL5.sum()})')
    ax.scatter([0], [0], c='gold', s=220, marker='*', zorder=6, edgecolor='k')
    ax.scatter([jx], [jy], c='brown', s=130, marker='o', zorder=6, edgecolor='k', label='Jupiter')
    # co-orbital annulus + L4/L5 sector centre lines
    th = np.linspace(0, 2*np.pi, 200)
    for fac in (1-crit.coorbital_tolerance, 1+crit.coorbital_tolerance):
        ax.plot(r_j*fac*np.cos(th), r_j*fac*np.sin(th), color='gray', lw=0.6, ls=':', alpha=0.6)
    for sign, col in [(+1, 'tab:orange'), (-1, 'tab:purple')]:
        la = lam_j + sign*np.deg2rad(60)
        ax.plot([0, r_j*np.cos(la)], [0, r_j*np.sin(la)], color=col, lw=1.0, ls='--', alpha=0.6)
    ax.set_title(f"{label}  —  L4={isL4.sum()}, L5={isL5.sum()}  (r_J={r_j:.2f} AU)")
    ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x [AU]'); ax.set_ylabel('y [AU]'); ax.legend(fontsize=8, loc='upper right')
fig.suptitle('Final top-down (x-y) — L4 (orange) / L5 (purple) Trojans highlighted', y=1.00)
fig.tight_layout(); fig.savefig(f'{OUT}/trojan_topdown_4scenarios.png', dpi=130)
print(f"saved {OUT}/trojan_topdown_4scenarios.png")
print("DONE")
