"""
Jupiter migration via REBOUNDx `modify_orbits_direct`.

This drives the same orbit-averaged effect as the original AMUSE
`JupiterMigrationCode.kick()` -- an exponential semimajor-axis drift plus
eccentricity damping (Papaloizou & Larwood 2000 prescription) -- but applied
as an OPERATOR, which is the correct, symplectic-compatible way to do migration
under the WHFast integrator. (The force-based `modify_orbits_forces` is
velocity-dependent and triggers a warning / non-symplectic kick under WHFast;
the operator form produces the identical secular a(t) and e(t) evolution that
`MigrationConfig.compute_tau_from_duration` is derived for.)

Sign convention (PUBLIC API, preserved from the original code):
    config.migration.tau_a > 0  ->  INWARD  migration (a shrinks)
    config.migration.tau_a < 0  ->  OUTWARD migration (a grows)

REBOUNDx uses the opposite convention -- da/dt = a / tau_a, so a NEGATIVE
REBOUNDx tau_a shrinks the orbit. We negate exactly once, here, so the rest of
the codebase keeps the "positive = inward" convention.
"""

import reboundx


def attach_migration(sim, config, jupiter_index: int = 4):
    """
    Attach migration + eccentricity damping to Jupiter.

    Args:
        sim: REBOUND simulation (Jupiter at `jupiter_index`).
        config: SimulationConfig with config.migration populated.
        jupiter_index: index of Jupiter in the particle list.

    Returns:
        The reboundx.Extras instance (keep a reference alive for the run).
    """
    rebx = reboundx.Extras(sim)
    mod = rebx.load_operator("modify_orbits_direct")
    rebx.add_operator(mod)

    jupiter = sim.particles[jupiter_index]

    # da/dt = a / tau_a_rebx. Inward (user tau_a > 0) needs tau_a_rebx < 0.
    jupiter.params["tau_a"] = -config.migration.tau_a

    # de/dt = e / tau_e_rebx. Damping needs tau_e_rebx < 0.
    if config.migration.tau_e is not None:
        jupiter.params["tau_e"] = -abs(config.migration.tau_e)

    return rebx
