"""Growth-and-migration tracks for Pirani et al. 2019a, rebuilt from the papers she cites (stage 9).

Pirani sect. 3.1: tracks "generated similarly to Bitsch, Lambrechts & Johansen 2015 (BLJ15)",
with the Ida, Guillot & Morbidelli 2016 disc (alpha = 1e-3) instead of the Bitsch+2015 disc,
the BLJ15 pebble flux replaced (Bitsch+2018), and "for the gas accretion and planetary
migration we followed the prescriptions in BLJ15". The pieces, as implemented here:

Disc: ``gasdisc`` (Ida+2016 eqs. 9-13, Mdot(t) of BLJ15 eq. 1, gone at 3 Myr).

Pebbles (Ida+2016 sect. 3.5, eqs. 59-61, with chi = 1, eta' = eta, zeta = 1/(1 + tau^2)):
    Mdot_c = C zeta^-1 b^2 / (4 sqrt(2 pi) tau h_p) (1 + 3 b / (2 eta)) Mdot_F,  <= Mdot_F
    b = min(1, sqrt(3 tau^(1/3) R_H / eta)) 2 kappa tau^(1/3) R_H,  C = min(sqrt(8/pi) h_p / b, 1)
    h_p = (1 + tau/alpha)^(-1/2) h_g,  kappa = exp(-(tau / min(2, tau*))^0.65), tau* = 4 (M/M*)/eta^3
  Pebble flux Mdot_F: Ida+2016 eq. 54, 9.4e-4 alpha3^-1 Mdot8 (t/Myr)^(-1/3) M_E/yr, times
  ``flux_factor`` (Pirani gives no value: the one free parameter of the core phase).
  Stokes number: Ida+2016 eq. 47 (drift/growth balance), tau = sqrt(3 sqrt(3) pi alpha Mdot_F /
  (16 |dlnP/dlnr| eta Mdot*)), or a fixed ``tau_s``.

Pebble isolation (BLJ15 eq. 16, Lambrechts+2014): M_iso = 20 (h/0.05)^3 M_E. Before it, 10%
of the accreted mass is envelope (BLJ15 sect. 2.3).

Gas (BLJ15 sect. 2.3): envelope contraction at the Piso & Youdin 2014 rate (BLJ15 eq. 17;
f = 0.2, kappa_env = 0.05 cm^2/g, rho_c = 5.5 g/cm^3) while M_env < M_core; then runaway,
the smaller of the Machida+2010 branches (BLJ15 eqs. 18-19), capped at 0.8 Mdot* (Lubow &
D'Angelo 2006).

Migration (BLJ15 sect. 2.4): type I scaled from Tanaka+2002's isothermal torque,
    adot_I = -k_I 2 (2.7 + 1.1 beta) q (Sigma r^2 / M*) h^-2 v_K,
where ``k_I`` stands for the Paardekooper+2011 torque (which needs the opacity-dependent
thermal diffusion, not given by Ida+2016; Pirani's Fig. 13 gives k_I ~ 0.45). Partial gap:
times f(P) of Crida & Morbidelli 2007 (BLJ15 eq. 23), with P of Crida+2006 (BLJ15 eq. 21).
For P < 1: type II, adot = -r / tau_II, tau_II = tau_nu max(1, M / (4 pi Sigma r^2)) (BLJ15 eq. 22).

Units: au, yr, Msun internally; masses reported in M_E.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from trojans import constants, gasdisc
from trojans.disc import M_EARTH

K_B = 1.380649e-16
M_H = 1.6735e-24
MU = 2.34
G_CGS = 6.674e-8
MSUN_G = 1.98840987e33
AU_CM = constants.AU_KM * 1e5
YR_S = constants.YEAR_S


@dataclass(frozen=True)
class TrackParams:
    a0_au: float = 18.4
    t0_yr: float = 2.31e6
    m_seed_earth: float = 0.018
    m_final_earth: float = 1e9           # stop growth here (default: never)
    flux_factor: float = 1.0             # times Ida+2016 eq. 54
    tau_s: float = 0.0                   # 0: Ida eq. 47
    torque: str = "pbk11"                # "pbk11" (BLJ15) or "tanaka" (times k_I)
    k_I: float = 0.45                    # type-I torque relative to Tanaka+2002 isothermal (torque="tanaka")
    torque_scale: float = 1.0            # times the PBK11 torque (calibrated on Pirani's Fig. 13)
    z_dust: float = 0.005                # micron-dust metallicity for the opacity (BLJ15: 0.5%)
    iso_factor: float = 1.0              # times BLJ15 eq. 16
    env_frac: float = 0.1                # envelope fraction before isolation (BLJ15)
    f_py: float = 0.2                    # Piso & Youdin fudge factor
    kappa_env: float = 0.05
    gas_cap: float = 0.8                 # of Mdot*
    gap: bool = True
    alpha: float = gasdisc.ALPHA
    t_end_yr: float = gasdisc.T_GAS_END
    dt_yr: float = 100.0


def mdot_flux(t, p: TrackParams):
    """Pebble mass flux, Msun/yr (Ida+2016 eq. 54 times flux_factor)."""
    mdot8 = gasdisc.mdot(t) / 1e-8
    return p.flux_factor * 9.4e-4 * (p.alpha / 1e-3) ** -1 * mdot8 * (t / 1e6) ** (-1 / 3) * M_EARTH


def stokes(a, t, p: TrackParams):
    if p.tau_s > 0:
        return p.tau_s
    eta = float(gasdisc.eta(a, t))
    dlnp = abs(float(gasdisc.dlnp_dlnr(a, t)))
    return math.sqrt(3 * math.sqrt(3) * math.pi * p.alpha * mdot_flux(t, p) / (16 * dlnp * eta * gasdisc.mdot(t)))


def pebble_rate(m, a, t, p: TrackParams):
    """Ida+2016 eqs. 59-61, in Msun/yr."""
    tau = stokes(a, t, p)
    eta = float(gasdisc.eta(a, t))
    hg = float(gasdisc.aspect_ratio(a, t))
    hp = hg / math.sqrt(1 + tau / p.alpha)
    rh = (m / 3.0) ** (1 / 3)
    tau_star = 4 * m / eta**3
    kappa = math.exp(-(tau / min(2.0, tau_star)) ** 0.65)
    b = min(1.0, math.sqrt(3 * tau ** (1 / 3) * rh / eta)) * 2 * kappa * tau ** (1 / 3) * rh
    C = min(math.sqrt(8 / math.pi) * hp / b, 1.0)
    zeta = 1 / (1 + tau**2)
    flux = mdot_flux(t, p)
    rate = C / zeta * b**2 / (4 * math.sqrt(2 * math.pi) * tau * hp) * (1 + 3 * b / (2 * eta)) * flux
    return min(rate, flux)


def isolation_mass(a, t, p: TrackParams):
    return p.iso_factor * 20.0 * (float(gasdisc.aspect_ratio(a, t)) / 0.05) ** 3 * M_EARTH


def temperature(a, t):
    cs = float(gasdisc.aspect_ratio(a, t)) * math.sqrt(constants.G / a) * AU_CM / YR_S
    return MU * M_H * cs**2 / K_B


def contraction_rate(m_core, m_env, a, t, p: TrackParams):
    """Piso & Youdin 2014 (BLJ15 eq. 17), Msun/yr."""
    me = M_EARTH
    rate = (0.00175 * p.f_py**-2 * (p.kappa_env / 1.0) ** -1 * (m_core / me) ** (11 / 3)
            * (m_env / (0.1 * me)) ** -1 * (temperature(a, t) / 81.0) ** -0.5)      # M_E / Myr
    return rate * me / 1e6


def runaway_rate(m, a, t, p: TrackParams):
    """Machida+2010 (BLJ15 eqs. 18-19), smaller branch, capped at gas_cap * Mdot*."""
    om = math.sqrt(constants.G / a**3)
    sig = float(gasdisc.sigma(a, t))
    H = float(gasdisc.aspect_ratio(a, t)) * a
    rh = a * (m / 3.0) ** (1 / 3)
    low = 0.83 * om * sig * H**2 * (rh / H) ** 4.5
    high = 0.14 * om * sig * H**2
    return min(low, high, p.gas_cap * float(gasdisc.mdot(t)))


def gap_parameter(m, a, t, p: TrackParams):
    h = float(gasdisc.aspect_ratio(a, t))
    rh = a * (m / 3.0) ** (1 / 3)
    reynolds = 1 / (p.alpha * h * h)
    return 0.75 * h * a / rh + 50.0 / (m * reynolds)


def f_gap(P):
    """Crida & Morbidelli 2007 (BLJ15 eq. 23)."""
    return (P - 0.541) / 4 if P < 2.4646 else 1.0 - math.exp(-P**0.75 / 3)


SIGMA_SB = 5.670374e-5
GAMMA = 1.4


def opacity(T, p: "TrackParams"):
    """Bell & Lin 1994 ice-grain branch (T < 166.8 K), kappa = 2e-4 T^2 cm^2/g, scaled by Z_dust/0.01."""
    return 2e-4 * T**2 * (p.z_dust / 0.01)


def pbk11_torque(m, a, t, p: "TrackParams"):
    """Paardekooper, Baruteau & Kley 2011 total type-I torque in units of Gamma_0 = (q/h)^2 Sigma r^4 Omega^2."""
    h = float(gasdisc.aspect_ratio(a, t))
    om = math.sqrt(constants.G / a**3)
    al = float(gasdisc.sigma_power(a, t))                      # Sigma ~ r^-al
    hv, hi = gasdisc.aspect_ratios(a, t)
    be = 0.9 if hv >= hi else 3 / 7                           # T ~ r^-be (Ida+2016)
    xi = be - (GAMMA - 1.0) * al
    T = temperature(a, t)
    rho = float(gasdisc.rho_mid(a, t)) * MSUN_G / AU_CM**3
    H_cm = h * a * AU_CM
    om_s = om / YR_S
    chi = 16 * GAMMA * (GAMMA - 1) * SIGMA_SB * T**4 / (3 * opacity(T, p) * rho**2 * H_cm**2 * om_s**2)   # cm^2/s
    chi_code = chi / AU_CM**2 * YR_S                          # au^2/yr
    Q = 2 * chi_code / (3 * h**3 * a**2 * om)
    ge = 2 * Q * GAMMA / (GAMMA * Q + 0.5 * math.sqrt(2 * math.sqrt((GAMMA**2 * Q**2 + 1) ** 2 - 16 * Q**2 * (GAMMA - 1))
                                                     + 2 * GAMMA**2 * Q**2 - 2))
    q = m
    xs = 1.1 / ge**0.25 * math.sqrt(q / h)                    # in units of r
    nu = p.alpha * h * h * a * a * om
    pnu = 2 / 3 * math.sqrt(a * a * om * xs**3 / (2 * math.pi * nu))
    pchi = math.sqrt(a * a * om * xs**3 / (2 * math.pi * chi_code))

    def F(x):
        return 1 / (1 + (x / 1.3) ** 2)

    def G(x):
        c = math.sqrt(8 / (45 * math.pi))
        return 16 / 25 * (45 * math.pi / 8) ** 0.75 * x**1.5 if x < c else 1 - 9 / 25 * (8 / (45 * math.pi)) ** (4 / 3) * x ** (-8 / 3)

    def K(x):
        c = math.sqrt(28 / (45 * math.pi))
        return 16 / 25 * (45 * math.pi / 28) ** 0.75 * x**1.5 if x < c else 1 - 9 / 25 * (28 / (45 * math.pi)) ** (4 / 3) * x ** (-8 / 3)

    lind = (-2.5 - 1.7 * be + 0.1 * al) / ge
    hs_baro = 1.1 * (1.5 - al) / ge
    hs_ent = 7.9 * xi / ge / ge
    lin_baro = 0.7 * (1.5 - al) / ge
    lin_ent = (2.2 - 1.4 / ge) * xi / ge
    corot = (hs_baro * F(pnu) * G(pnu) + (1 - K(pnu)) * lin_baro
             + hs_ent * F(pnu) * F(pchi) * math.sqrt(G(pnu) * G(pchi))
             + math.sqrt((1 - K(pnu)) * (1 - K(pchi))) * lin_ent)
    return lind + corot


def migration_rate(m, a, t, p: TrackParams):
    """da/dt in au/yr (negative inward) and the regime."""
    h = float(gasdisc.aspect_ratio(a, t))
    sig = float(gasdisc.sigma(a, t))
    beta = float(gasdisc.sigma_power(a, t))
    vk = math.sqrt(constants.G / a)
    if p.torque == "pbk11":
        type1 = p.torque_scale * 2 * pbk11_torque(m, a, t, p) * m * sig * a * a / h**2 * vk   # adot = 2 Gamma/(m r Omega)
    else:
        type1 = -p.k_I * 2 * (2.7 + 1.1 * beta) * m * sig * a * a / h**2 * vk
    if not p.gap:
        return type1, "I"
    P = gap_parameter(m, a, t, p)
    if P >= 1.0:
        return type1 * f_gap(P), "I_gap" if P < 2.4646 else "I"
    nu = p.alpha * h * h * a * a * math.sqrt(constants.G / a**3)
    tau_nu = a * a / nu
    tau2 = tau_nu * max(1.0, m / (4 * math.pi * sig * a * a))
    return -a / tau2, "II"


def integrate(p: TrackParams, path=None):
    """Forward Euler (dt_yr steps; dt << all growth and migration times). Returns a dict of arrays.

    ``path = (m_earth, a_au)``: prescribe a(m) (e.g. Pirani's own Fig. 1 curve) instead of
    integrating the migration; only the growth (the clock) is then modelled.
    """
    if p.alpha != gasdisc.ALPHA:
        raise ValueError(f"alpha {p.alpha} differs from the disc's gasdisc.ALPHA = {gasdisc.ALPHA} (Pirani: 1e-3)")
    if path is not None:
        pm, pa = (np.asarray(x, float) for x in path)
        o = np.argsort(pm)
        pm, pa = np.log10(pm[o]), pa[o]

        def a_of(mass_msun):
            return float(np.interp(math.log10(mass_msun / M_EARTH), pm, pa))
    t, a = p.t0_yr, p.a0_au if path is None else float(pa[0])
    m = p.m_seed_earth * M_EARTH
    m_core, m_env = (1 - p.env_frac) * m, p.env_frac * m
    phase = "pebble"
    rows = []
    m_iso_reached = None
    while t < p.t_end_yr:
        adot, regime = migration_rate(m, a, t, p)
        if phase == "pebble":
            dm = pebble_rate(m, a, t, p)
            if m + dm * p.dt_yr >= isolation_mass(a, t, p):
                phase, m_iso_reached = "contraction", m / M_EARTH
                dm = 0.0
            dcore, denv = (1 - p.env_frac) * dm, p.env_frac * dm
        elif phase == "contraction":
            denv = min(contraction_rate(m_core, m_env, a, t, p), runaway_rate(m, a, t, p))
            dcore = 0.0
            if m_env >= m_core:
                phase = "runaway"
        else:
            denv, dcore = runaway_rate(m, a, t, p), 0.0
        if m >= p.m_final_earth * M_EARTH:
            denv = dcore = 0.0
        rows.append((t, a, m / M_EARTH, adot * 1e6, (dcore + denv) / M_EARTH * 1e6, phase, regime))
        m_core += dcore * p.dt_yr
        m_env += denv * p.dt_yr
        m = m_core + m_env
        if path is None:
            a += adot * p.dt_yr
        else:
            a_new = a_of(m)
            rows[-1] = rows[-1][:3] + ((a_new - a) / p.dt_yr * 1e6,) + rows[-1][4:]
            a = a_new
        t += p.dt_yr
        if a < 0.1:
            break
    rows.append((t, a, m / M_EARTH, 0.0, 0.0, phase, "end"))
    cols = list(zip(*rows))
    out = {k: np.array(v) for k, v in zip(["t", "a", "m", "adot_au_myr", "mdot_me_myr"], cols[:5])}
    out["phase"] = np.array(cols[5])
    out["regime"] = np.array(cols[6])
    out["m_iso_reached"] = m_iso_reached
    out["params"] = asdict(p)
    return out
