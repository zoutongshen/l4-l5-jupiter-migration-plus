"""
Generate a 3D animated GIF from a simulation pickle.

Works on the NumPy-array pickle format written by the `simulation` package
(keys: times, massive_pos, massive_names, planetesimal_pos). No REBOUND/AMUSE
needed -- just numpy + matplotlib.

Usage:
    conda activate trojan-rb
    python generate_3d_gif.py results/pkl/jupiter_migration_50kyr_inward.pkl \
        --output simulation_3d.gif --fps 10 --step 1
"""

import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


PLANET_COLORS = {
    'Sun': 'yellow', 'Venus': 'orange', 'Earth': 'blue', 'Mars': 'red',
    'Jupiter': 'brown', 'Saturn': 'gold', 'Uranus': 'cyan', 'Neptune': 'darkblue',
}


def load_data(filename):
    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Generate 3D GIF from simulation pickle.')
    parser.add_argument('input_file', type=str, help='Path to input .pkl file')
    parser.add_argument('--output', type=str, default='simulation_3d.gif', help='Output GIF filename')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--step', type=int, default=1, help='Step size for frames (skip frames)')
    args = parser.parse_args()

    data = load_data(args.input_file)
    times = data['times']
    massive_pos = np.asarray(data['massive_pos'])          # (n_snap, n_massive, 3)
    names = data.get('massive_names', [])
    planetesimal_pos = np.asarray(data['planetesimal_pos'])  # (n_snap, n_p, 3)
    n_frames = len(times)
    print(f"Loaded {n_frames} snapshots.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Axis limits from the initial massive-body extent (with headroom).
    max_range = np.max(np.abs(massive_pos[0])) * 1.5
    if max_range < 6:
        max_range = 6
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_xlabel('X [AU]'); ax.set_ylabel('Y [AU]'); ax.set_zlabel('Z [AU]')

    if len(names) == massive_pos.shape[1]:
        colors = [PLANET_COLORS.get(n, 'black') for n in names]
    else:
        colors = ['yellow'] + ['blue'] * (massive_pos.shape[1] - 1)

    scat_planets = ax.scatter([], [], [], c=colors, s=50, depthshade=False, label='Massive Bodies')
    scat_planetesimals = ax.scatter([], [], [], c='gray', s=1, alpha=0.5, label='Planetesimals')
    title_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    ax.legend()

    def update(frame):
        m = massive_pos[frame]
        p = planetesimal_pos[frame]
        title_text.set_text(f'Time: {times[frame]:.1f} yr')
        scat_planets._offsets3d = (m[:, 0], m[:, 1], m[:, 2])
        if len(p) > 0:
            scat_planetesimals._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        return scat_planets, scat_planetesimals, title_text

    frames = range(0, n_frames, args.step)
    anim = FuncAnimation(fig, update, frames=frames, blit=False)

    print(f"Saving animation to {args.output}...")
    anim.save(args.output, writer=PillowWriter(fps=args.fps))
    print("Done!")


if __name__ == "__main__":
    main()
