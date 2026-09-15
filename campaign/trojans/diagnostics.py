"""Vectorised orbital elements, the resonant angle, and V24's A, B, C.

All angles are in degrees unless a name says otherwise. Positions and velocities
are heliocentric; ``mu`` is G times the mass the orbit is taken around.
"""

import numpy as np


def wrap180(angle_deg):
    """Wrap to (-180, 180]."""
    wrapped = np.mod(np.asarray(angle_deg) + 180.0, 360.0) - 180.0
    return np.where(wrapped == -180.0, 180.0, wrapped)


def elements(pos, vel, mu):
    """Osculating elements for arrays of states, shape (..., 3).

    Returns a dict of arrays: a, e, inc, Omega, omega, M, lam (mean longitude),
    varpi (longitude of pericentre). Angles in degrees; M and lam are NaN for
    unbound orbits.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    r = np.linalg.norm(pos, axis=-1)
    v2 = np.sum(vel * vel, axis=-1)

    h = np.cross(pos, vel)
    hn = np.linalg.norm(h, axis=-1)
    hhat = h / hn[..., None]
    inc = np.arccos(np.clip(hhat[..., 2], -1.0, 1.0))

    # Ascending node direction; x-axis when the orbit lies in the reference plane.
    node = np.stack([-h[..., 1], h[..., 0], np.zeros_like(r)], axis=-1)
    nn = np.linalg.norm(node, axis=-1)
    flat = nn < 1e-15 * hn
    nhat = np.where(flat[..., None], [1.0, 0.0, 0.0], node / np.where(flat, 1.0, nn)[..., None])
    Omega = np.arctan2(nhat[..., 1], nhat[..., 0])

    evec = np.cross(vel, h) / mu - pos / r[..., None]
    e = np.linalg.norm(evec, axis=-1)
    a = -mu / (2.0 * (0.5 * v2 - mu / r))

    def angle_from_node(vec):
        return np.arctan2(np.sum(np.cross(nhat, vec) * hhat, axis=-1), np.sum(nhat * vec, axis=-1))

    theta = Omega + angle_from_node(pos)          # true longitude
    omega = angle_from_node(evec)
    f = theta - Omega - omega                      # true anomaly

    bound = (e < 1.0) & (a > 0.0)
    ecc = np.where(bound, e, 0.0)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(f / 2.0), np.sqrt(1.0 + ecc) * np.cos(f / 2.0))
    M = np.where(bound, E - ecc * np.sin(E), np.nan)

    deg = np.degrees
    return {
        "a": a,
        "e": e,
        "inc": deg(inc),
        "Omega": np.mod(deg(Omega), 360.0),
        "omega": np.mod(deg(omega), 360.0),
        "M": np.mod(deg(M), 360.0),
        "varpi": np.mod(deg(Omega + omega), 360.0),
        "lam": np.mod(deg(Omega + omega + M), 360.0),
    }


def cartesian(a, e, inc, Omega, omega, M, mu):
    """Heliocentric position and velocity arrays (n, 3) from elliptic elements (degrees)."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    inc, Omega, omega, M = (np.radians(np.asarray(x, float)) for x in (inc, Omega, omega, M))
    E = M + e * np.sin(M)
    for _ in range(50):
        step = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - step
        if np.max(np.abs(step)) < 1e-14:
            break
    f = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0))
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(f))
    u = omega + f
    cO, sO, ci, si = np.cos(Omega), np.sin(Omega), np.cos(inc), np.sin(inc)
    pos = np.stack([
        r * (cO * np.cos(u) - sO * np.sin(u) * ci),
        r * (sO * np.cos(u) + cO * np.sin(u) * ci),
        r * np.sin(u) * si,
    ], axis=-1)
    k = np.sqrt(mu / p)
    vu = -np.sin(u) - e * np.sin(omega)   # velocity components along the node frame
    vw = np.cos(u) + e * np.cos(omega)
    vel = np.stack([
        k * (cO * vu - sO * vw * ci),
        k * (sO * vu + cO * vw * ci),
        k * vw * si,
    ], axis=-1)
    return pos, vel


def resonant_angle(lam_deg, lam_jupiter_deg):
    """sigma = lambda - lambda_J in (-180, 180]; L4 librates about +60, L5 about -60."""
    return wrap180(np.asarray(lam_deg) - np.asarray(lam_jupiter_deg))


def libration_integral_f(sigma_deg):
    """f(sigma) = (1 + 4|sin(sigma/2)|^3) / (2|sin(sigma/2)|), V24 Eq. 3."""
    s = np.abs(np.sin(np.radians(sigma_deg) / 2.0))
    return (1.0 + 4.0 * s**3) / (2.0 * s)


def abc(a, e, inc, Omega, varpi, sigma, jupiter, mass_ratio):
    """V24's quasi-proper amplitudes A (inclination), B (eccentricity), C (libration).

    ``jupiter`` is a dict with a, e, inc, Omega, varpi for Jupiter at the same time.
    ``mass_ratio`` is mu = m_J / (M_central + m_J). Angles in degrees.
    A = |sin I e^{i Omega} - sin I_J e^{i Omega_J}|
    B = |e e^{i(varpi - varpi_J)} - e_J e^{+-i pi/3}|, + for L4 (sigma > 0), - for L5
    C from (da/a_J)^2 = (8 mu / 3) [C - f(sigma)]
    """
    rad = np.radians
    zeta = np.sin(rad(inc)) * np.exp(1j * rad(Omega))
    zeta_j = np.sin(rad(jupiter["inc"])) * np.exp(1j * rad(jupiter["Omega"]))
    A = np.abs(zeta - zeta_j)

    side = np.where(np.asarray(sigma) >= 0.0, 1.0, -1.0)
    z = np.asarray(e) * np.exp(1j * rad(np.asarray(varpi) - jupiter["varpi"]))
    B = np.abs(z - jupiter["e"] * np.exp(1j * side * np.pi / 3.0))

    da = (np.asarray(a) - jupiter["a"]) / jupiter["a"]
    C = 3.0 * da**2 / (8.0 * mass_ratio) + libration_integral_f(sigma)
    return A, B, C
