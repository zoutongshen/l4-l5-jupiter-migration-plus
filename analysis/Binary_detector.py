"""
Trojan Binary Analysis
======================

Analyze the Jupiter Trojan populations from a simulation to look for potential
binary systems in L4 and L5, and compare the two clouds.

Works on the NumPy-array pickle format written by the `simulation` package
(keys: massive_pos, planetesimal_pos, planetesimal_vel, ...). No REBOUND/AMUSE
needed to load -- just numpy + scipy.

Run from the analysis/ directory:
    conda activate trojan-rb
    python Binary_detector.py
"""

# %% [markdown]
# # Trojan Binary Analysis
# Find close, gravitationally bound pairs in L4 / L5.

# %%  Setup
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from simulation import classify_planetesimals

# Gravitational constant in AU^3 / (Msun * yr^2)
G = 4.0 * np.pi ** 2

print('Libraries loaded successfully!')

# %%  Load a simulation result
FILENAME = 'results/pkl/jupiter_migration_50kyr_inward.pkl'

with open(FILENAME, 'rb') as f:
    data = pickle.load(f)

times = data['times']
massive_pos = data['massive_pos']
planetesimal_pos = data['planetesimal_pos']
planetesimal_vel = data['planetesimal_vel']

print('=' * 60)
print(f'Loaded {FILENAME}')
print('=' * 60)
print(f"  Snapshots: {len(times)}  ({times[0]:.1f} - {times[-1]:.1f} yr)")
print(f"  Massive bodies: {massive_pos.shape[1]}")
print(f"  Planetesimals: {planetesimal_pos.shape[1]}")


# %%  Identify Trojans in the final snapshot
JUPITER_INDEX = data.get('config', {}).get('jupiter_index', 4)


def identify_trojans(massive_pos_snap, planetesimal_pos_snap, jupiter_index=JUPITER_INDEX):
    """Return (l4_indices, l5_indices) using the shared classifier."""
    c = classify_planetesimals(massive_pos_snap, planetesimal_pos_snap,
                               jupiter_index=jupiter_index)
    return np.array(c['L4'], dtype=int), np.array(c['L5'], dtype=int)


final_massive = massive_pos[-1]
final_pos = planetesimal_pos[-1]
final_vel = planetesimal_vel[-1]

l4_idx, l5_idx = identify_trojans(final_massive, final_pos)
print('=' * 60)
print('Trojan Identification (Final Snapshot)')
print('=' * 60)
print(f'L4 Trojans: {len(l4_idx)}')
print(f'L5 Trojans: {len(l5_idx)}')


# %%  Binary detection algorithm
def find_binaries(positions, velocities, sun_mass=1.0, max_hill_radii=5.0,
                  m_planetesimal=1e-10):
    """
    Find potential binary pairs among particles (positions in AU, velocities
    in AU/yr). Returns a list of dicts with separation, eccentricity, bound flag.
    """
    n = len(positions)
    if n < 2:
        return []

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    binaries = []
    for i in range(n):
        for j in range(i + 1, n):
            r_ij = np.linalg.norm(positions[j] - positions[i])

            a_avg = 0.5 * (np.linalg.norm(positions[i]) + np.linalg.norm(positions[j]))
            R_hill = a_avg * (2 * m_planetesimal / (3 * sun_mass)) ** (1 / 3)
            sep_hill = r_ij / R_hill

            if sep_hill >= max_hill_radii:
                continue

            v_rel = velocities[j] - velocities[i]
            r_rel = positions[j] - positions[i]
            v_rel_mag = np.linalg.norm(v_rel)

            m_total = 2 * m_planetesimal
            E_specific = 0.5 * v_rel_mag ** 2 - G * m_total / r_ij
            is_bound = E_specific < 0

            if is_bound:
                h = np.cross(r_rel, v_rel)
                h_mag = np.linalg.norm(h)
                a_mutual = -G * m_total / (2 * E_specific)
                e = np.sqrt(max(0.0, 1 + 2 * E_specific * h_mag ** 2 / (G * m_total) ** 2))
            else:
                a_mutual, e = None, None

            binaries.append({
                'i': i, 'j': j,
                'separation_AU': r_ij,
                'separation_hill': sep_hill,
                'rel_velocity': v_rel_mag,
                'eccentricity': e,
                'semi_major_axis': a_mutual,
                'bound': is_bound,
            })
    return binaries


print('Binary detection function defined!')

# %%  Search for binaries in L4 and L5 (final snapshot)
l4_pos, l4_vel = final_pos[l4_idx], final_vel[l4_idx]
l5_pos, l5_vel = final_pos[l5_idx], final_vel[l5_idx]

l4_binaries = find_binaries(l4_pos, l4_vel, max_hill_radii=5.0)
l5_binaries = find_binaries(l5_pos, l5_vel, max_hill_radii=5.0)
l4_bound = [b for b in l4_binaries if b['bound']]
l5_bound = [b for b in l5_binaries if b['bound']]

print('=' * 60)
print('Binary Search Results (Final Snapshot)')
print('=' * 60)
for label, particles, bound in [('L4', l4_pos, l4_bound), ('L5', l5_pos, l5_bound)]:
    print(f'\n{label} Region ({len(particles)} particles):')
    print(f'  Gravitationally bound pairs: {len(bound)}')
    if bound:
        seps = [b['separation_hill'] for b in bound]
        ecc = [b['eccentricity'] for b in bound if b['eccentricity'] is not None]
        print(f'  Separation range: {min(seps):.2f} - {max(seps):.2f} R_Hill')
        if ecc:
            print(f'  Eccentricity range: {min(ecc):.3f} - {max(ecc):.3f}')

# %% [markdown]
# ## Scan all simulation files for binaries

# %%  Batch scan over results/pkl/*.pkl
pkl_files = sorted(glob.glob('results/pkl/*.pkl'))
print('=' * 80)
print(f'BINARY SEARCH ACROSS {len(pkl_files)} FILES')
print('=' * 80)

summary_rows = []
for filename in pkl_files:
    try:
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        if 'planetesimal_pos' not in d:
            print(f'  Skipping {filename} (old/unknown format)')
            continue

        mpos = d['massive_pos'][-1]
        ppos = d['planetesimal_pos'][-1]
        pvel = d['planetesimal_vel'][-1]
        jidx = d.get('config', {}).get('jupiter_index', 4)

        li4, li5 = identify_trojans(mpos, ppos, jupiter_index=jidx)
        b4 = [b for b in find_binaries(ppos[li4], pvel[li4]) if b['bound']] if len(li4) >= 2 else []
        b5 = [b for b in find_binaries(ppos[li5], pvel[li5]) if b['bound']] if len(li5) >= 2 else []

        summary_rows.append((filename.split('/')[-1], ppos.shape[0],
                             len(li4), len(li5), len(b4), len(b5)))
    except Exception as e:
        print(f'  Error on {filename}: {e}')

print(f"\n{'Filename':<45} {'Particles':>10} {'L4':>6} {'L5':>6} {'L4Bin':>7} {'L5Bin':>7}")
print('-' * 80)
for fname, npart, n4, n5, nb4, nb5 in summary_rows:
    fname_disp = (fname[:43] + '..') if len(fname) > 45 else fname
    print(f"{fname_disp:<45} {npart:>10} {n4:>6} {n5:>6} {nb4:>7} {nb5:>7}")
print('-' * 80)
files_with_binaries = sum(1 for r in summary_rows if r[4] > 0 or r[5] > 0)
print(f'\nFiles with bound binaries: {files_with_binaries}/{len(summary_rows)}')
