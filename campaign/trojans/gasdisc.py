"""Gas disc of Pirani et al. 2019a (stage 9): Ida, Guillot & Morbidelli 2016, alpha = 1e-3.

A steady accretion disc (Ida+2016 sect. 2), parametrised by the stellar accretion rate
Mdot(t) of Hartmann+1998 as used by Bitsch+2015 (BLJ15 eq. 1):

    log10(Mdot / Msun/yr) = -8.00 - 1.40 log10((t + 1e5 yr) / 1e6 yr)

Pirani's eq. 1 drops the 1e5 yr offset, which makes Mdot infinite at t = 0; BLJ15, whose
tracks she follows, has it. The two differ by < 7% after 2 Myr (3 Myr: 2.05e-9 vs 2.15e-9).
The gas is removed at T_GAS_END = 3 Myr (Pirani sect. 3.1, BLJ15 sect. 2.1).

Ida+2016 (M = 1 Msun, L = 1 Lsun):
    h_vis = 0.027 alpha3^-1/10 Mdot8^1/5 (r/au)^1/20          (eq. 9)
    h_irr = 0.024 (r/au)^2/7                                   (eq. 10)
    h     = max(h_vis, h_irr)                                  (sect. 2, T = max(T_vis, T_irr))
    Sigma = Mdot / (3 pi alpha h^2 r^2 Omega)                  (eq. 11)
    rho_mid = Sigma / (sqrt(2 pi) h r)                         (her h_g convention, footnote 1)
    dlnP/dlnr = -xi + q - 2 = -51/20 (viscous), -39/14 (irradiation)   (P ~ Sigma h r^-2)

Units: au, yr, Msun (as the rest of campaign/), radii in au, angles in radians.
Every function is vectorised in r and is a pure function of (r, t).
"""

import numpy as np

from trojans import constants

ALPHA = 1.0e-3
T_GAS_END = 3.0e6                 # yr
MDOT_T_OFFSET = 1.0e5             # yr, BLJ15 eq. 1
GM = constants.G * 1.0            # Ida's M* = 1 Msun

G_CM3_TO_MSUN_AU3 = (constants.AU_KM * 1e5) ** 3 / 1.98840987e33
G_CM2_TO_MSUN_AU2 = (constants.AU_KM * 1e5) ** 2 / 1.98840987e33


def mdot(t_yr):
    """Stellar accretion rate in Msun/yr (zero once the gas is gone)."""
    t = np.asarray(t_yr, float)
    m = 10.0 ** (-8.00 - 1.40 * np.log10((t + MDOT_T_OFFSET) / 1e6))
    return np.where(t < T_GAS_END, m, 0.0)


def gas_present(t_yr) -> bool:
    return float(t_yr) < T_GAS_END


def aspect_ratios(r_au, t_yr, alpha=ALPHA):
    """(h_vis, h_irr) of Ida+2016 eqs. 9-10."""
    r = np.asarray(r_au, float)
    mdot8 = np.maximum(mdot(t_yr), 1e-30) / 1e-8
    h_vis = 0.027 * (alpha / 1e-3) ** -0.1 * mdot8 ** 0.2 * r ** (1 / 20)
    h_irr = 0.024 * r ** (2 / 7)
    return h_vis, h_irr


def aspect_ratio(r_au, t_yr, alpha=ALPHA):
    h_vis, h_irr = aspect_ratios(r_au, t_yr, alpha)
    return np.maximum(h_vis, h_irr)


def omega(r_au):
    return np.sqrt(GM / np.asarray(r_au, float) ** 3)


def sigma(r_au, t_yr, alpha=ALPHA):
    """Gas surface density in Msun/au^2 (0 after T_GAS_END)."""
    r = np.asarray(r_au, float)
    h = aspect_ratio(r, t_yr, alpha)
    return mdot(t_yr) / (3.0 * np.pi * alpha * h**2 * r**2 * omega(r))


def rho_mid(r_au, t_yr, alpha=ALPHA):
    """Midplane gas density in Msun/au^3."""
    r = np.asarray(r_au, float)
    return sigma(r, t_yr, alpha) / (np.sqrt(2.0 * np.pi) * aspect_ratio(r, t_yr, alpha) * r)


def dlnp_dlnr(r_au, t_yr, alpha=ALPHA):
    """Midplane pressure gradient: -2.55 where viscous heating wins, -2.79 where irradiation does."""
    h_vis, h_irr = aspect_ratios(r_au, t_yr, alpha)
    return np.where(h_vis >= h_irr, -0.6 + 0.05 - 2.0, -15 / 14 + 2 / 7 - 2.0)


def eta(r_au, t_yr, alpha=ALPHA):
    """Sub-Keplerian fraction of the gas, eta = -(1/2) h^2 dlnP/dlnr > 0 (Pirani eq. A.8, |.|)."""
    return -0.5 * aspect_ratio(r_au, t_yr, alpha) ** 2 * dlnp_dlnr(r_au, t_yr, alpha)


def sigma_power(r_au, t_yr, alpha=ALPHA):
    """beta in Sigma ~ r^-beta: 3/5 viscous, 15/14 irradiation (Ida+2016 sect. 2)."""
    h_vis, h_irr = aspect_ratios(r_au, t_yr, alpha)
    return np.where(h_vis >= h_irr, 0.6, 15 / 14)
