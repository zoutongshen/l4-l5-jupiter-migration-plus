"""Animate the six orbital elements of a Jupiter Trojan, one GIF per element.

Each GIF varies one element and holds the others fixed:
  a      semi-major axis, swept across V24's box 4.95-5.45 au
  e      eccentricity, swept 0-0.3 (the MPC/JPL cut)
  i      inclination, swept 0-45 deg (V24 cuts at 50)
  Omega  longitude of the ascending node, one full turn
  omega  argument of perihelion, one full turn
  lambda mean longitude (lambda = M + varpi), with sigma = lambda - lambda_J

Jupiter's orbit (a_J = 5.20 au, circular, I = 0) is the reference throughout.
The panels are schematic: in the lambda panel the libration of sigma is sped up
about 12-fold so that one swing fits in one orbit.

    python docs/figures/make_element_gifs.py
writes docs/figures/elements/<name>.gif.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = Path(__file__).resolve().parent / "elements"
A_J = 5.20
N_FRAMES = 72
FPS = 18
SIZE_IN, DPI = 4.6, 100

BG = "#f5f7f9"
INK = "#17202b"
MUTED = "#7b8693"
RULE = "#c9d1db"
TROJAN = "#16707f"
SUN = "#d08a2c"
ACCENT = "#b5651d"
L5 = "#8a3a6b"

plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.edgecolor": BG,
                     "font.family": "DejaVu Sans"})


def pingpong(lo, hi, n=N_FRAMES):
    s = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    return lo + (hi - lo) * s


def ellipse_xy(a, e, omega=0.0, n=400):
    """Orbit in its own plane, Sun at the focus, perihelion at angle omega."""
    E = np.linspace(0, 2 * np.pi, n)
    x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E)
    c, s = np.cos(omega), np.sin(omega)
    return c * x - s * y, s * x + c * y


def rotate3d(x, y, inc, node):
    """Orbital-plane (x, y) to ecliptic xyz: tilt by inc about x, then turn by node about z."""
    y2, z2 = y * np.cos(inc), y * np.sin(inc)
    c, s = np.cos(node), np.sin(node)
    return c * x - s * y2, s * x + c * y2, z2


def new_2d(title):
    fig, ax = plt.subplots(figsize=(SIZE_IN, SIZE_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-7.2, 7.2)
    ax.set_ylim(-7.2, 7.2)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 0.90)
    fig.text(0.04, 0.95, title, fontsize=14, fontweight="bold", va="center")
    return fig, ax


def new_3d(title):
    fig = plt.figure(figsize=(SIZE_IN, SIZE_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_facecolor(BG)
    ax.set_axis_off()
    lim = 5.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim * 0.6, lim * 0.6)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=20, azim=-58)
    fig.subplots_adjust(-0.12, -0.12, 1.12, 1.0)
    fig.text(0.04, 0.95, title, fontsize=14, fontweight="bold", va="center")
    return fig, ax


def draw_sun(ax):
    ax.plot(0, 0, "o", color=SUN, ms=11, zorder=5)


def jupiter_orbit(ax):
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(A_J * np.cos(t), A_J * np.sin(t), color=MUTED, lw=1, ls=(0, (4, 3)))


def save(anim, fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / f"{name}.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {OUT / name}.gif")


def gif_a():
    fig, ax = new_2d("a — semi-major axis")
    t = np.linspace(0, 2 * np.pi, 300)
    for r0, r1 in [(4.95, 5.45)]:
        ax.fill(np.r_[r1 * np.cos(t), r0 * np.cos(t[::-1])], np.r_[r1 * np.sin(t), r0 * np.sin(t[::-1])],
                color=TROJAN, alpha=0.10, lw=0)
    jupiter_orbit(ax)
    draw_sun(ax)
    ax.text(0, -6.75, "shaded: V24 box 4.95 < a < 5.45 au   dashed: Jupiter, 5.20 au",
            ha="center", fontsize=8.5, color=MUTED)
    e = 0.05
    (orb,) = ax.plot([], [], color=TROJAN, lw=2.2)
    (arrow,) = ax.plot([], [], color=ACCENT, lw=1.6)
    (ctr,) = ax.plot([], [], "+", color=ACCENT, ms=9)
    label = ax.text(0, 6.3, "", ha="center", fontsize=12, color=ACCENT, fontweight="bold")
    note = fig.text(0.04, 0.885, "half the long axis of the ellipse: sets the period", fontsize=9.5, color=MUTED)
    a_vals = pingpong(4.95, 5.45)

    def frame(k):
        a = a_vals[k]
        x, y = ellipse_xy(a, e)
        orb.set_data(x, y)
        cx = -a * e
        arrow.set_data([cx, cx + a], [0, 0])
        ctr.set_data([cx], [0])
        label.set_text(f"a = {a:.2f} au   P = {a**1.5:.2f} yr")
        return orb, arrow, ctr, label, note

    save(FuncAnimation(fig, frame, N_FRAMES), fig, "a_semimajor_axis")


def gif_e():
    fig, ax = new_2d("e — eccentricity")
    jupiter_orbit(ax)
    draw_sun(ax)
    fig.text(0.04, 0.885, "how stretched the ellipse is; Sun sits at one focus", fontsize=9.5, color=MUTED)
    ax.text(0, -6.75, "swept 0 → 0.3 (MPC and JPL cut at e < 0.3)", ha="center", fontsize=8.5, color=MUTED)
    a = A_J
    (orb,) = ax.plot([], [], color=TROJAN, lw=2.2)
    (qp,) = ax.plot([], [], "o", color=ACCENT, ms=7)
    (Qp,) = ax.plot([], [], "o", color=L5, ms=7)
    qt = ax.text(0, 0, "", color=ACCENT, fontsize=10, ha="center")
    Qt = ax.text(0, 0, "", color=L5, fontsize=10, ha="center")
    label = ax.text(0, 1.0, "", ha="center", fontsize=12, color=INK, fontweight="bold")
    e_vals = pingpong(0.0, 0.3)

    def frame(k):
        e = e_vals[k]
        x, y = ellipse_xy(a, e)
        orb.set_data(x, y)
        q, Q = a * (1 - e), a * (1 + e)
        qp.set_data([q], [0])
        Qp.set_data([-Q], [0])
        qt.set_position((q, -0.9))
        qt.set_text(f"q = {q:.2f}")
        Qt.set_position((-Q, -0.9))
        Qt.set_text(f"Q = {Q:.2f}")
        label.set_text(f"e = {e:.2f}")
        return orb, qp, Qp, qt, Qt, label

    save(FuncAnimation(fig, frame, N_FRAMES), fig, "e_eccentricity")


def reference_plane(ax, r=5.6):
    t = np.linspace(0, 2 * np.pi, 120)
    ax.plot_trisurf(np.r_[0, r * np.cos(t)], np.r_[0, r * np.sin(t)], np.zeros(121),
                    color=RULE, alpha=0.35, linewidth=0, zorder=0)
    ax.plot(A_J * np.cos(t), A_J * np.sin(t), 0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.scatter([0], [0], [0], color=SUN, s=70, zorder=6, depthshade=False)


def gif_i():
    fig, ax = new_3d("i — inclination")
    fig.text(0.04, 0.885, "tilt of the orbit against Jupiter's orbital plane", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.04, "swept 0° → 45°  (V24 cuts at 50°; cold L4 excess sits below 15°)",
             ha="center", fontsize=8.5, color=MUTED)
    reference_plane(ax)
    ax.plot([0, 0], [0, 0], [0, 3.2], color=MUTED, lw=1.2)
    i_vals = pingpong(0.0, np.radians(45))
    E = np.linspace(0, 2 * np.pi, 300)
    arts = {}

    def frame(k):
        for v in arts.values():
            v.remove()
        arts.clear()
        inc = i_vals[k]
        x, y, z = rotate3d(A_J * np.cos(E), A_J * np.sin(E), inc, 0.0)
        arts["orb"] = ax.plot(x, y, z, color=TROJAN, lw=2.2, zorder=3)[0]
        arts["node"] = ax.plot([-A_J, A_J], [0, 0], [0, 0], color=INK, lw=0.8, ls=":", zorder=2)[0]
        nx, ny, nz = rotate3d(np.array([0.0]), np.array([0.0]), inc, 0.0)
        n = np.array([0, -np.sin(inc), np.cos(inc)]) * 3.2
        arts["normal"] = ax.plot([0, n[0]], [0, n[1]], [0, n[2]], color=ACCENT, lw=1.8, zorder=4)[0]
        th = np.linspace(0, inc, 30)
        arts["arc"] = ax.plot(0 * th, -1.6 * np.sin(th), 1.6 * np.cos(th), color=ACCENT, lw=1.4, zorder=4)[0]
        arts["lab"] = fig.text(0.5, 0.11, f"i = {np.degrees(inc):4.1f}°", ha="center", fontsize=12,
                               color=ACCENT, fontweight="bold")
        return list(arts.values())

    save(FuncAnimation(fig, frame, N_FRAMES), fig, "i_inclination")


def gif_node():
    fig, ax = new_3d("Ω — longitude of ascending node")
    fig.text(0.04, 0.885, "where the orbit rises through the reference plane,\nmeasured from the reference direction",
             fontsize=9.5, color=MUTED, va="top")
    fig.text(0.5, 0.04, "i held at 25°; Ω turned through 360°", ha="center", fontsize=8.5, color=MUTED)
    reference_plane(ax)
    ax.plot([0, 5.4], [0, 0], [0, 0], color=INK, lw=1.4)
    ax.text(5.6, 0, 0, "ref. dir.", fontsize=9, color=INK)
    inc = np.radians(25)
    E = np.linspace(0, 2 * np.pi, 300)
    nodes = np.radians(np.linspace(0, 360, N_FRAMES, endpoint=False))
    arts = {}

    def frame(k):
        for v in arts.values():
            v.remove()
        arts.clear()
        node = nodes[k]
        x, y, z = rotate3d(A_J * np.cos(E), A_J * np.sin(E), inc, node)
        up = z >= 0
        arts["orb_up"] = ax.plot(np.where(up, x, np.nan), np.where(up, y, np.nan), np.where(up, z, np.nan),
                                 color=TROJAN, lw=2.2, zorder=3)[0]
        arts["orb_dn"] = ax.plot(np.where(~up, x, np.nan), np.where(~up, y, np.nan), np.where(~up, z, np.nan),
                                 color=TROJAN, lw=1.6, ls=(0, (3, 2)), zorder=2)[0]
        c, s = np.cos(node), np.sin(node)
        arts["line"] = ax.plot([0, A_J * c], [0, A_J * s], [0, 0], color=ACCENT, lw=1.6, zorder=4)[0]
        arts["pt"] = ax.scatter([A_J * c], [A_J * s], [0], color=ACCENT, s=40, zorder=6, depthshade=False)
        th = np.linspace(0, node, 60)
        arts["arc"] = ax.plot(2.0 * np.cos(th), 2.0 * np.sin(th), 0 * th, color=ACCENT, lw=1.4, zorder=4)[0]
        arts["lab"] = fig.text(0.5, 0.11, f"Ω = {np.degrees(node):5.1f}°", ha="center", fontsize=12,
                               color=ACCENT, fontweight="bold")
        return list(arts.values())

    save(FuncAnimation(fig, frame, N_FRAMES), fig, "Omega_ascending_node")


def gif_peri():
    fig, ax = new_2d("ω — argument of perihelion")
    fig.text(0.04, 0.885, "angle in the orbit plane from the ascending node to perihelion",
             fontsize=9.5, color=MUTED)
    ax.text(0, -6.75, "seen face-on to the orbit, e = 0.25 exaggerated;  ϖ = Ω + ω",
            ha="center", fontsize=8.5, color=MUTED)
    ax.plot([-6.6, 6.6], [0, 0], color=INK, lw=0.9, ls=":")
    ax.text(6.6, 0.3, "line of nodes", ha="right", fontsize=9, color=INK)
    draw_sun(ax)
    a, e = A_J, 0.25
    (orb,) = ax.plot([], [], color=TROJAN, lw=2.2)
    (peri,) = ax.plot([], [], "o", color=ACCENT, ms=7)
    (ray,) = ax.plot([], [], color=ACCENT, lw=1.4)
    (arc,) = ax.plot([], [], color=ACCENT, lw=1.4)
    label = ax.text(0, 6.3, "", ha="center", fontsize=12, color=ACCENT, fontweight="bold")
    oms = np.radians(np.linspace(0, 360, N_FRAMES, endpoint=False))

    def frame(k):
        om = oms[k]
        x, y = ellipse_xy(a, e, om)
        orb.set_data(x, y)
        q = a * (1 - e)
        peri.set_data([q * np.cos(om)], [q * np.sin(om)])
        ray.set_data([0, q * np.cos(om)], [0, q * np.sin(om)])
        th = np.linspace(0, om, 60)
        arc.set_data(1.5 * np.cos(th), 1.5 * np.sin(th))
        label.set_text(f"ω = {np.degrees(om):5.1f}°")
        return orb, peri, ray, arc, label

    save(FuncAnimation(fig, frame, N_FRAMES), fig, "omega_perihelion")


def gif_lambda():
    fig, ax = new_2d(r"λ — mean longitude, and σ = λ − $\lambda_J$")
    fig.text(0.04, 0.885, "where the body is along its orbit, λ = M + ϖ;\nfor a Trojan, σ is its angle ahead of Jupiter",
             fontsize=9.5, color=MUTED, va="top")
    ax.text(0, -6.95, "σ libration sped up ~12× for display; L4 leads Jupiter by ~60°",
            ha="center", fontsize=8.5, color=MUTED)
    jupiter_orbit(ax)
    draw_sun(ax)
    ax.plot([0, 6.6], [0, 0], color=INK, lw=0.9, ls=":")
    ax.text(6.6, 0.3, "ref. dir.", ha="right", fontsize=9, color=INK)
    (jup,) = ax.plot([], [], "o", color=INK, ms=10)
    (tro,) = ax.plot([], [], "o", color=TROJAN, ms=8)
    (lam_arc,) = ax.plot([], [], color=MUTED, lw=1.2)
    (sig_arc,) = ax.plot([], [], color=ACCENT, lw=2.0)
    (rj,) = ax.plot([], [], color=INK, lw=0.8)
    (rt,) = ax.plot([], [], color=TROJAN, lw=0.8)
    label = ax.text(0, -6.0, "", ha="center", fontsize=11.5, color=INK, fontweight="bold")
    lam_j = np.radians(np.linspace(0, 360, N_FRAMES, endpoint=False))
    sigma = np.radians(60 + 18 * np.sin(2 * np.pi * np.arange(N_FRAMES) / N_FRAMES))

    def frame(k):
        lj, sg = lam_j[k], sigma[k]
        lt = lj + sg
        jup.set_data([A_J * np.cos(lj)], [A_J * np.sin(lj)])
        tro.set_data([A_J * np.cos(lt)], [A_J * np.sin(lt)])
        rj.set_data([0, A_J * np.cos(lj)], [0, A_J * np.sin(lj)])
        rt.set_data([0, A_J * np.cos(lt)], [0, A_J * np.sin(lt)])
        th = np.linspace(0, lt % (2 * np.pi), 120)
        lam_arc.set_data(1.2 * np.cos(th), 1.2 * np.sin(th))
        th = np.linspace(lj, lt, 40)
        sig_arc.set_data(3.6 * np.cos(th), 3.6 * np.sin(th))
        label.set_text(f"λ = {np.degrees(lt) % 360:5.1f}°   $\\lambda_J$ = {np.degrees(lj) % 360:5.1f}°   "
                       f"σ = {np.degrees(sg):+5.1f}°")
        return jup, tro, rj, rt, lam_arc, sig_arc, label

    save(FuncAnimation(fig, frame, N_FRAMES), fig, "lambda_mean_longitude")


if __name__ == "__main__":
    gif_a()
    gif_e()
    gif_i()
    gif_node()
    gif_peri()
    gif_lambda()
