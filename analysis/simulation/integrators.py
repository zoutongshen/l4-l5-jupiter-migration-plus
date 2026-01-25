"""
Integrator classes for Jupiter Trojan simulations.

- ExternalGravityField: Container for massive bodies
- PlanetesimalIntegrator: Vectorized leapfrog for test particles
- JupiterMigrationCode: Type II migration physics for Jupiter
"""

import numpy as np
from amuse.units import units, constants


class ExternalGravityField:
    """Simple container to hold massive bodies reference for planetesimal integrator."""

    def __init__(self, massive_bodies):
        self.massive_bodies = massive_bodies


class PlanetesimalIntegrator:
    """
    Vectorized Leapfrog integrator for planetesimals in external gravitational field.

    Optimizations:
    - Keeps positions/velocities in NumPy arrays between steps
    - Only syncs back to AMUSE Particles when requested (at snapshots)
    - Fully vectorized force calculations
    """

    def __init__(self, particles, gravity_field):
        self.particles = particles
        self.gravity_field = gravity_field
        self.model_time = 0 | units.yr

        # Cache particle state in arrays (AU, AU/yr units)
        self._x = np.array([p.x.value_in(units.AU) for p in particles])
        self._y = np.array([p.y.value_in(units.AU) for p in particles])
        self._z = np.array([p.z.value_in(units.AU) for p in particles])
        self._vx = np.array([p.vx.value_in(units.AU / units.yr) for p in particles])
        self._vy = np.array([p.vy.value_in(units.AU / units.yr) for p in particles])
        self._vz = np.array([p.vz.value_in(units.AU / units.yr) for p in particles])

    def _massive_arrays(self):
        """Extract massive body positions and masses as arrays."""
        bodies = self.gravity_field.massive_bodies
        mx = np.array([b.x.value_in(units.AU) for b in bodies])
        my = np.array([b.y.value_in(units.AU) for b in bodies])
        mz = np.array([b.z.value_in(units.AU) for b in bodies])
        masses = np.array([b.mass.value_in(units.MSun) for b in bodies])
        return mx, my, mz, masses

    def evolve_model(self, tend):
        """Evolve using vectorized Leapfrog kick-drift-kick scheme."""
        dt = tend - self.model_time
        dt_val = dt.value_in(units.yr)

        mx, my, mz, masses = self._massive_arrays()
        G_val = constants.G.value_in(units.AU ** 3 / (units.MSun * units.yr ** 2))

        def compute_accel(x, y, z):
            """Compute gravitational acceleration from all massive bodies."""
            # Shape: (n_planetesimals, n_massive)
            dx = x[:, None] - mx[None, :]
            dy = y[:, None] - my[None, :]
            dz = z[:, None] - mz[None, :]

            r2 = dx * dx + dy * dy + dz * dz
            r3 = r2 * np.sqrt(r2)

            # Avoid division by zero
            r3 = np.maximum(r3, 1e-10)

            factor = (-G_val * masses) / r3
            ax = np.sum(factor * dx, axis=1)
            ay = np.sum(factor * dy, axis=1)
            az = np.sum(factor * dz, axis=1)
            return ax, ay, az

        # Half kick
        ax, ay, az = compute_accel(self._x, self._y, self._z)
        self._vx += 0.5 * ax * dt_val
        self._vy += 0.5 * ay * dt_val
        self._vz += 0.5 * az * dt_val

        # Full drift
        self._x += self._vx * dt_val
        self._y += self._vy * dt_val
        self._z += self._vz * dt_val

        # Half kick
        ax, ay, az = compute_accel(self._x, self._y, self._z)
        self._vx += 0.5 * ax * dt_val
        self._vy += 0.5 * ay * dt_val
        self._vz += 0.5 * az * dt_val

        self.model_time = tend

    def sync_to_particles(self):
        """Write cached arrays back to AMUSE Particles (call at snapshots)."""
        for i, p in enumerate(self.particles):
            p.x = self._x[i] | units.AU
            p.y = self._y[i] | units.AU
            p.z = self._z[i] | units.AU
            p.vx = self._vx[i] | units.AU / units.yr
            p.vy = self._vy[i] | units.AU / units.yr
            p.vz = self._vz[i] | units.AU / units.yr

    def update_from_particles(self):
        """Update cached arrays from AMUSE Particles (if particles were modified externally)."""
        self._x = np.array([p.x.value_in(units.AU) for p in self.particles])
        self._y = np.array([p.y.value_in(units.AU) for p in self.particles])
        self._z = np.array([p.z.value_in(units.AU) for p in self.particles])
        self._vx = np.array([p.vx.value_in(units.AU / units.yr) for p in self.particles])
        self._vy = np.array([p.vy.value_in(units.AU / units.yr) for p in self.particles])
        self._vz = np.array([p.vz.value_in(units.AU / units.yr) for p in self.particles])


class JupiterMigrationCode:
    """
    Apply Type II migration to Jupiter.

    Physics (matching original notebook implementation):
    - Migration: Tangential force to change semi-major axis
    - Eccentricity damping: Damps deviation from circular velocity

    Sign convention:
    - Positive tau_a: INWARD migration (removes angular momentum)
    - Negative tau_a: OUTWARD migration (adds angular momentum)
    """

    def __init__(self, gravity_code, massive_bodies, jupiter_index=1):
        """
        Initialize migration code.

        Args:
            gravity_code: AMUSE gravity integrator (e.g., Huayno)
            massive_bodies: Particles set with massive bodies
            jupiter_index: Index of Jupiter in the particles set (default: 1)
                          Note: This depends on your particle order!
        """
        self.code = gravity_code
        self.massive_bodies = massive_bodies
        self.jupiter_idx = jupiter_index
        self.model_time = 0 | units.yr
        self.timestep = None

        # Migration parameters (set after initialization)
        self.tau_a = None  # Migration timescale (years, as AMUSE quantity)
        self.tau_e = None  # Eccentricity damping timescale (years, as AMUSE quantity)

    def evolve_model(self, tend):
        """Evolve model time (Bridge will call kick())."""
        self.model_time = tend

    def kick(self, dt):
        """
        Apply migration force to Jupiter.

        Called by AMUSE Bridge during integration.
        Physics matches original notebook implementation exactly.

        Args:
            dt: Timestep for the kick (AMUSE quantity with time units)
        """
        # Sync from integrator to local particles
        channel_from_code = self.code.particles.new_channel_to(self.massive_bodies)
        channel_from_code.copy()

        jupiter = self.massive_bodies[self.jupiter_idx]
        sun = self.massive_bodies[0]

        # Relative position and velocity (Jupiter relative to Sun)
        r_vec = jupiter.position - sun.position
        v_vec = jupiter.velocity - sun.velocity

        r = r_vec.length()
        r_hat = r_vec / r

        # Radial and tangential velocity components
        v_r = (v_vec * r_hat).sum()  # Radial velocity (dot product)
        v_r_vec = v_r * r_hat
        v_tan_vec = v_vec - v_r_vec  # Tangential velocity vector
        v_tan = v_tan_vec.length()

        # Migration force (changes semi-major axis)
        if self.tau_a is not None:
            # Type II migration via tangential force
            # Theory: da/dt = -2a/tau_a for inward migration
            # For circular orbits: v = sqrt(GM/a), so v ∝ 1/sqrt(a)
            # Therefore: dv/dt = -v/(2*tau_a)

            if v_tan > (0 | units.km / units.s):  # Avoid division by zero
                a_migration = -(v_tan_vec / (2.0 * self.tau_a))

                jupiter.vx += a_migration[0] * dt
                jupiter.vy += a_migration[1] * dt
                jupiter.vz += a_migration[2] * dt

        # Eccentricity damping (circularizes orbit)
        if self.tau_e is not None:
            # Circular velocity at current radius
            v_circ = (constants.G * sun.mass / r).sqrt()

            # Circular velocity vector (perpendicular to radius)
            # Direction: tangential, magnitude: v_circ
            if v_tan > (0 | units.km / units.s):
                v_circ_vec = (v_circ / v_tan) * v_tan_vec
            else:
                v_circ_vec = v_tan_vec  # Fallback if v_tan ~ 0

            # Eccentricity velocity (deviation from circular)
            v_ecc = v_vec - v_circ_vec

            # Damp eccentricity: dv/dt = -v_ecc / tau_e
            a_damping = -v_ecc / self.tau_e

            jupiter.vx += a_damping[0] * dt
            jupiter.vy += a_damping[1] * dt
            jupiter.vz += a_damping[2] * dt

        # Sync modified velocities back to integrator
        channel_to_code = self.massive_bodies.new_channel_to(self.code.particles)
        channel_to_code.copy()

    def get_particles(self):
        """Return particles (required by Bridge)."""
        return self.massive_bodies

    @property
    def particles(self):
        """Particles property (required by Bridge)."""
        return self.massive_bodies
