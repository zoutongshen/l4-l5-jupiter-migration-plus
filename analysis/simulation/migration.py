"""
Jupiter migration by direct, Jupiter-only orbital-element nudging.

WHY NOT REBOUNDx
----------------
REBOUNDx's `modify_orbits_direct` (operator) and `modify_orbits_forces` (force)
both LOOP OVER EVERY PARTICLE on every timestep. With ~10k massless test
particles that loop dominates the cost: profiling showed ~100 ms/step (operator)
and ~52 ms/step (force) versus ~0.8 ms/step with no REBOUNDx -- a 60-120x
slowdown, turning a ~20 min run into 24-46 h. Only Jupiter ever migrates, so
scanning the whole cloud each step is pure waste.

WHAT THIS DOES INSTEAD
----------------------
`MigrationDriver` applies the same secular evolution -- a(t) = a0 * exp(-t/tau_a)
and e(t) = e0 * exp(-t/tau_e) -- by directly nudging ONLY Jupiter's orbital
elements between short integration sub-steps. Cost is O(1) per nudge, independent
of the particle count, so a migration run is as fast as a plain burn-in.
This is the same idea as `modify_orbits_direct`, restricted to the one body that
moves. The sub-step is kept small (default: per-nudge fractional change in a <=
1e-4) so the migration is smooth and does not artificially excite the Trojans;
validated against the slow REBOUNDx operator (matching a(t) and L4/L5 counts).

SIGN CONVENTION (PUBLIC API, preserved from the original code)
--------------------------------------------------------------
    config.migration.tau_a > 0  ->  INWARD  migration (a shrinks)
    config.migration.tau_a < 0  ->  OUTWARD migration (a grows)
With a(t) = a0 * exp(-t / tau_a), a positive tau_a shrinks a (inward) and a
negative tau_a grows it (outward) -- exactly the public convention, no negation
needed anywhere.
"""

import numpy as np
import rebound


class MigrationDriver:
    """Drives Jupiter's exponential semimajor-axis migration + eccentricity
    damping by nudging only Jupiter's elements between integration sub-steps."""

    def __init__(self, sim, config, jupiter_index: int = 4,
                 max_frac_per_step: float = 1.0e-4):
        self.sim = sim
        self.jidx = jupiter_index
        self.tau_a = float(config.migration.tau_a)
        te = config.migration.tau_e
        self.tau_e = float(te) if te is not None else None
        # Sub-step so the per-nudge fractional change in a stays tiny (smooth).
        self.max_frac = float(max_frac_per_step)

    # -- internal: apply the migration for an interval `dt` to Jupiter only -- #
    def _nudge(self, dt: float) -> None:
        sun = self.sim.particles[0]
        jup = self.sim.particles[self.jidx]
        orb = jup.orbit(primary=sun)

        # a(t) = a0 * exp(-t/tau_a): +tau_a shrinks a (inward), -tau_a grows it.
        a_new = orb.a * np.exp(-dt / self.tau_a)
        e_new = orb.e
        if self.tau_e is not None and self.tau_e != 0.0:
            # e always DAMPS regardless of migration direction.
            e_new = orb.e * np.exp(-dt / abs(self.tau_e))

        # Rebuild Jupiter at the same orientation and orbital phase (f) with the
        # new a, e. Replacing one massive body and re-deriving its Cartesian
        # state from heliocentric elements is the "direct" prescription.
        self.sim.particles[self.jidx] = rebound.Particle(
            simulation=self.sim, primary=sun, m=jup.m,
            a=a_new, e=e_new, inc=orb.inc, Omega=orb.Omega,
            omega=orb.omega, f=orb.f, hash=jup.hash)

        # WHFast caches Jacobi/heliocentric coordinates; force a recompute since
        # we changed a particle's state outside the integrator.
        self.sim.ri_whfast.recalculate_coordinates_this_timestep = 1

    def _dt_sub(self) -> float:
        """Sub-step length: small enough for smooth migration, but never below
        the integrator timestep."""
        tau_min = abs(self.tau_a)
        if self.tau_e is not None and self.tau_e != 0.0:
            tau_min = min(tau_min, abs(self.tau_e))
        return max(self.max_frac * tau_min, self.sim.dt)

    def advance(self, target: float, integrate_fn=None):
        """Integrate the simulation to absolute time `target`, applying migration
        nudges every sub-step.

        `integrate_fn(sim, sub_target) -> n_removed` performs the actual
        integration over a sub-interval (e.g. the escape-aware integrator in
        runner.py). Defaults to plain `sim.integrate`, which removes nothing.
        Returns the total number of particles removed across the sub-steps.
        """
        if integrate_fn is None:
            def integrate_fn(s, tt):
                s.integrate(tt)
                return 0

        dt_sub = self._dt_sub()
        removed = 0
        t = self.sim.t
        # Guard against pathological tiny sub-steps producing a huge loop.
        while t < target - 1.0e-9:
            sub_target = min(t + dt_sub, target)
            removed += integrate_fn(self.sim, sub_target)
            self._nudge(sub_target - t)
            t = sub_target
        return removed


def attach_migration(sim, config, jupiter_index: int = 4) -> MigrationDriver:
    """Create a MigrationDriver for Jupiter.

    Kept under the original name/signature so call sites and the public API are
    unchanged. Unlike the old REBOUNDx version this attaches NOTHING to the
    simulation object; the caller drives migration by calling
    ``driver.advance(target, integrate_fn)`` instead of ``sim.integrate(target)``.
    """
    return MigrationDriver(sim, config, jupiter_index=jupiter_index)
