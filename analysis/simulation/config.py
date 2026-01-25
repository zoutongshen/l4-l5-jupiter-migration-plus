"""
Configuration classes for Jupiter Trojan simulations.

Defines all parameters for:
- Solar system setup (planet masses, orbits)
- Simulation runtime parameters
- Migration physics
- Trojan detection criteria
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal
from amuse.units import units


# Default planet data (mass in Earth masses, a in AU, e dimensionless, i in degrees)
DEFAULT_PLANET_DATA = {
    'Venus':   {'mass': 0.815,  'a': 0.72,  'e': 0.007,  'i': 3.39},
    'Earth':   {'mass': 1.0,    'a': 1.0,   'e': 0.017,  'i': 0.0},
    'Mars':    {'mass': 0.107,  'a': 1.52,  'e': 0.093,  'i': 1.85},
    'Jupiter': {'mass': 317.8,  'a': 5.2,   'e': 0.0489, 'i': 1.30},
    'Saturn':  {'mass': 95.16,  'a': 9.58,  'e': 0.0565, 'i': 2.49},
    'Uranus':  {'mass': 14.54,  'a': 19.22, 'e': 0.0457, 'i': 0.77},
    'Neptune': {'mass': 17.15,  'a': 30.11, 'e': 0.0113, 'i': 1.77},
}


@dataclass
class TrojanCriteria:
    """Criteria for identifying Trojan asteroids."""

    # Co-orbital criterion: particle must be within this fraction of Jupiter's semi-major axis
    coorbital_tolerance: float = 0.07  # 7%

    # Angular separation from L4/L5 points (degrees)
    sector_half_width: float = 25.0  # ±25° from ±60°

    # L4 is at +60° from Jupiter, L5 is at -60°
    l4_angle: float = 60.0
    l5_angle: float = -60.0


@dataclass
class MigrationConfig:
    """Configuration for Jupiter migration physics.

    Sign convention:
    - Positive tau_a: INWARD migration (removes angular momentum)
    - Negative tau_a: OUTWARD migration (adds angular momentum)
    """

    enabled: bool = False

    # Migration timescale (None = no migration)
    tau_a: Optional[float] = None  # in years

    # Eccentricity damping timescale (typically tau_a / 100)
    tau_e: Optional[float] = None  # in years

    # Target semi-major axis for migration
    target_a: float = 5.2  # AU

    # Starting semi-major axis (for computing tau from duration)
    initial_a: float = 5.6  # AU

    # Direction: 'inward' or 'outward'
    direction: Literal['inward', 'outward'] = 'inward'

    def compute_tau_from_duration(self, duration_yr: float) -> float:
        """
        Compute tau_a from desired migration duration.

        For exponential migration: a(t) = a0 * exp(-t/tau_a)
        Duration = time to reach target_a from initial_a

        tau_a = duration / ln(initial_a / target_a)
        """
        import numpy as np
        ratio = self.initial_a / self.target_a
        tau = duration_yr / np.log(ratio)

        # Sign convention: positive for inward, negative for outward
        if self.direction == 'outward':
            tau = -abs(tau)
        else:
            tau = abs(tau)

        return tau


@dataclass
class SimulationConfig:
    """Main configuration for Jupiter Trojan simulations."""

    # Simulation name (used for output files)
    name: str = "jupiter_simulation"

    # Simulation type: 'burnin', 'source_tracking', 'migration'
    sim_type: Literal['burnin', 'source_tracking', 'migration'] = 'burnin'

    # Star mass
    star_mass: float = 1.0  # Solar masses

    # Planet data (can override defaults)
    planet_data: Dict = field(default_factory=lambda: DEFAULT_PLANET_DATA.copy())

    # Jupiter initial semi-major axis (can differ from planet_data for migration)
    jupiter_initial_a: float = 5.2  # AU

    # Planetesimal configuration
    n_planetesimals: int = 10000
    planetesimal_a_min: float = 4.0  # AU
    planetesimal_a_max: float = 6.5  # AU
    planetesimal_e_max: float = 0.1
    planetesimal_i_max: float = 5.0  # degrees

    # Trojan initialization
    trojan_emphasis: bool = True  # If True, concentrates particles near L4/L5
    
    # For source tracking: use broader disk
    disk_a_min: float = 2.0   # AU
    disk_a_max: float = 35.0  # AU

    # Simulation runtime
    end_time: float = 50000.0  # years
    dt: float = 0.1  # years (integration timestep)
    n_snapshots: int = 200

    # Integrator parameters
    huayno_timestep: float = 0.05  # Internal integrator timestep

    # Migration configuration
    migration: MigrationConfig = field(default_factory=MigrationConfig)

    # Trojan detection criteria
    trojan_criteria: TrojanCriteria = field(default_factory=TrojanCriteria)

    # Input/Output
    input_file: Optional[str] = None  # For loading burn-in state
    output_file: Optional[str] = None  # Auto-generated if None
    checkpoint_interval: int = 50  # Save checkpoint every N snapshots

    # Random seed for reproducibility
    random_seed: int = 42

    def __post_init__(self):
        """Generate output filename if not provided."""
        if self.output_file is None:
            import os
            os.makedirs('results/pkl', exist_ok=True)
            self.output_file = f"results/pkl/{self.name}.pkl"

    @classmethod
    def burnin(cls, duration_kyr: float = 50, jupiter_a: float = 5.2,
               input_file: Optional[str] = None, **kwargs) -> 'SimulationConfig':
        """Create configuration for burn-in simulation."""
        name = f"jupiter_burnin_{int(duration_kyr)}kyr_{jupiter_a}au"
        return cls(
            name=name,
            sim_type='burnin',
            jupiter_initial_a=jupiter_a,
            end_time=duration_kyr * 1000,
            input_file=input_file,
            migration=MigrationConfig(enabled=False),
            **kwargs
        )

    @classmethod
    def source_tracking(cls, duration_kyr: float = 100, **kwargs) -> 'SimulationConfig':
        """Create configuration for source tracking (pure disk) simulation."""
        name = f"jupiter_source_tracking_{int(duration_kyr)}kyr"
        return cls(
            name=name,
            sim_type='source_tracking',
            planetesimal_a_min=2.0,
            planetesimal_a_max=35.0,
            end_time=duration_kyr * 1000,
            migration=MigrationConfig(enabled=False),
            **kwargs
        )

    @classmethod
    def migration_inward(cls, duration_kyr: float = 50,
                         initial_a: float = 5.6, target_a: float = 5.2,
                         input_file: Optional[str] = None, **kwargs) -> 'SimulationConfig':
        """Create configuration for inward migration simulation."""
        name = f"jupiter_migration_{int(duration_kyr)}kyr_inward"

        mig_config = MigrationConfig(
            enabled=True,
            initial_a=initial_a,
            target_a=target_a,
            direction='inward'
        )
        mig_config.tau_a = mig_config.compute_tau_from_duration(duration_kyr * 1000)
        mig_config.tau_e = abs(mig_config.tau_a) / 2

        return cls(
            name=name,
            sim_type='migration',
            jupiter_initial_a=initial_a,
            end_time=duration_kyr * 1000,
            input_file=input_file,
            migration=mig_config,
            **kwargs
        )

    @classmethod
    def migration_outward(cls, duration_kyr: float = 50,
                          initial_a: float = 4.8, target_a: float = 5.2,
                          input_file: Optional[str] = None, **kwargs) -> 'SimulationConfig':
        """Create configuration for outward migration simulation."""
        name = f"jupiter_migration_{int(duration_kyr)}kyr_outward"

        mig_config = MigrationConfig(
            enabled=True,
            initial_a=initial_a,
            target_a=target_a,
            direction='outward'
        )
        mig_config.tau_a = mig_config.compute_tau_from_duration(duration_kyr * 1000)
        mig_config.tau_e = abs(mig_config.tau_a) / 2

        return cls(
            name=name,
            sim_type='migration',
            jupiter_initial_a=initial_a,
            end_time=duration_kyr * 1000,
            input_file=input_file,
            migration=mig_config,
            **kwargs
        )
