"""Chunk runner: integrate one chunk of Trojans and record what happens to each.

Every ``check_every_steps`` WHFast steps the runner reads all states and
  - logs each Trojan's first entry within ``record_hill_radii`` and within
    ``remove_hill_radii`` of any giant, removing it at the latter;
  - removes Trojans with non-finite states or beyond ``r_max_au``;
  - logs the first time sigma = lambda - lambda_J crosses 180 deg and 0 deg;
  - adds the sample to per-window accumulators.

At the end of each window it stores, per Trojan: mean a, e, I, A, B, C; the
sigma range and circular mean; the libration class; the sample count; and
Jupiter's osculating a and mass at the window's end.

Optionally (``snapshot_every_yr`` > 0) it also stores instantaneous snapshots of
every particle (a, e, I, sigma, heliocentric x, y, z; NaN once removed) in
``snapshots.h5``: at t = 0, every ``snapshot_fine_every_yr`` up to
``snapshot_fine_until_yr``, and every ``snapshot_every_yr`` after. With
``snapshot_every_yr = window_yr`` the last check is always a snapshot.

Windows, snapshots, events and a REBOUND snapshot are written together every
``checkpoint_every_windows`` windows, and ``latest.json`` is switched to them in
one rename, so the files on disk always describe one consistent moment.

The caller chooses the output directory; this module only writes inside it.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rebound

from trojans import constants, diagnostics
from trojans.outputs import write_json

# Libration classes stored per window.
NO_SAMPLES, TADPOLE_L4, TADPOLE_L5, HORSESHOE, QUASI_SATELLITE, CIRCULATING = range(6)
CLASS_NAMES = ["no_samples", "tadpole_L4", "tadpole_L5", "horseshoe", "quasi_satellite", "circulating"]

# Removal reasons stored per Trojan.
ALIVE, REMOVED_HILL, REMOVED_R_MAX, REMOVED_NON_FINITE, REMOVED_COLLISION = range(5)
REMOVAL_NAMES = ["alive", "hill", "r_max", "non_finite", "collision"]

WINDOW_FIELDS = ["a", "e", "inc", "A", "B", "C", "sigma_min", "sigma_max", "sigma_cmean"]
JUPITER_FIELDS = ["jupiter_a", "jupiter_m"]          # per window; absent in files from before stage 8
SNAPSHOT_FIELDS = ["a", "e", "inc", "sigma", "x", "y", "z"]
SNAPSHOT_SCALARS = ["t", "jupiter_a", "jupiter_m", "jupiter_lam"]   # lam in deg, for the rotating frame


@dataclass(frozen=True)
class Settings:
    t_end_yr: float
    dt_yr: float
    check_every_steps: int = 8
    window_yr: float = 1.0e4
    checkpoint_every_windows: int = 10
    record_hill_radii: float = 2.1
    remove_hill_radii: float = 1.0
    r_max_au: float = 100.0
    safe_mode: int = 1
    r_crit_hill: float = 3.0          # MERCURIUS switchover radius in Hill radii (REBOUND default 3, as MERCURY)
    corrector: int = 0
    trojan_mass_msun: float = 0.0
    self_gravity: bool = False
    snapshot_every_yr: float = 0.0
    snapshot_fine_every_yr: float = 0.0
    snapshot_fine_until_yr: float = 0.0
    # Stage 9 (Pirani reproduction): "mercurius" for the hybrid integrator; removal on
    # impact with the Sun (radius in au) or a planet (radii set by the forcing in
    # runner.planet_radius_au; zero = off). Defaults keep earlier stages unchanged.
    integrator: str = "whfast"
    sun_radius_au: float = 0.0

    @property
    def check_interval_yr(self) -> float:
        return self.check_every_steps * self.dt_yr

    @property
    def checks_per_window(self) -> int:
        return max(1, round(self.window_yr / self.check_interval_yr))

    @property
    def n_windows(self) -> int:
        return int(np.ceil(self.t_end_yr / (self.checks_per_window * self.check_interval_yr)))

    @property
    def total_checks(self) -> int:
        return self.n_windows * self.checks_per_window

    @property
    def snapshots_enabled(self) -> bool:
        return self.snapshot_every_yr > 0.0 or self.snapshot_fine_every_yr > 0.0

    def snapshot_due(self, check_index: int, t_yr: float) -> bool:
        """Whether the sample at this check is also stored as a snapshot."""
        if not self.snapshots_enabled:
            return False
        if check_index == 0:
            return True
        fine = self.snapshot_fine_every_yr > 0.0 and t_yr <= self.snapshot_fine_until_yr
        every = self.snapshot_fine_every_yr if fine else self.snapshot_every_yr
        if every <= 0.0:
            return False
        return check_index % max(1, round(every / self.check_interval_yr)) == 0


def classify_windows(n, crossed0, crossed180, sigma_cmean):
    """Libration class from which of sigma = 0 and sigma = 180 deg were crossed."""
    cls = np.full(np.shape(n), CIRCULATING, dtype=np.int8)
    cls[crossed180 & ~crossed0] = HORSESHOE
    cls[crossed0 & ~crossed180] = QUASI_SATELLITE
    tadpole = ~crossed0 & ~crossed180
    cls[tadpole & (sigma_cmean > 0)] = TADPOLE_L4
    cls[tadpole & (sigma_cmean <= 0)] = TADPOLE_L5
    cls[np.asarray(n) == 0] = NO_SAMPLES
    return cls


def sigma_crossings(sigma_prev, sigma_now):
    """Which samples crossed sigma = 180 deg (a wrap) or sigma = 0 deg (a sign change)."""
    jump = np.abs(sigma_now - sigma_prev) > 180.0
    sign_change = (np.sign(sigma_now) != np.sign(sigma_prev)) & ~jump
    valid = np.isfinite(sigma_prev)
    return jump & valid, sign_change & valid


class ChunkRunner:
    """``planets`` names the massive bodies after the Sun, in simulation order; Jupiter first.

    ``forcing``, if given, is called as ``forcing(sim, runner)`` after every block of
    steps, before sampling. It may change the planets (e.g. Jupiter's mass) and must
    then update ``runner.hill`` and ``runner.m_jupiter`` to match. It must be a pure
    function of ``sim.t`` so that a resumed run continues it exactly.
    """

    def __init__(self, sim: rebound.Simulation, spkids, settings: Settings, giants: dict, out_dir: Path,
                 planets=constants.GIANTS, forcing=None):
        self.sim = sim
        self.s = settings
        self.out_dir = Path(out_dir)
        self.spkids = np.asarray(spkids, dtype=np.int64)
        self._order = np.argsort(self.spkids)
        self.n = len(self.spkids)
        self.planets = tuple(planets)
        self.forcing = forcing
        self.hill = np.array([giants[p]["hill_radius"] for p in self.planets])
        # Where the Trojans start. Not sim.N_active: with a self-gravitating swarm
        # every particle is active, while the massive bodies are still the first
        # five and are never removed.
        self.n_planets = 1 + len(self.planets)
        self.m_jupiter = constants.GIANT_MASS["jupiter"]

        self.check_index = 0
        self.window_index = 0
        self.events = {
            "t_hill_record": np.full(self.n, np.nan), "planet_hill_record": np.full(self.n, -1, np.int8),
            "t_hill_remove": np.full(self.n, np.nan), "planet_hill_remove": np.full(self.n, -1, np.int8),
            "t_removed": np.full(self.n, np.nan), "removal_reason": np.zeros(self.n, np.int8),
            "t_sigma_cross180": np.full(self.n, np.nan), "t_sigma_cross0": np.full(self.n, np.nan),
        }
        self.sigma_prev = np.full(self.n, np.nan)
        self._reset_window()
        self._window_buffer = []
        self.snapshot_index = 0
        self._snapshot_buffer = []
        self._jupiter_now = (np.nan, np.nan)
        self.planet_radius_au = np.zeros(len(self.planets))

        if settings.integrator == "mercurius":
            sim.ri_mercurius.safe_mode = settings.safe_mode
            sim.ri_mercurius.r_crit_hill = settings.r_crit_hill
        else:
            sim.ri_whfast.safe_mode = settings.safe_mode
            sim.ri_whfast.corrector = settings.corrector
            if settings.safe_mode == 0:
                sim.ri_whfast.keep_unsynchronized = 1

    def recalculate_coordinates(self) -> None:
        """Tell the integrator that particles were changed or removed from outside it."""
        if self.s.integrator == "mercurius":
            self.sim.ri_mercurius.recalculate_coordinates_this_timestep = 1
        else:
            self.sim.ri_whfast.recalculate_coordinates_this_timestep = 1

    # --- construction -------------------------------------------------------

    @property
    def mass_ratio(self) -> float:
        return self.m_jupiter / (constants.CENTRAL_MASS + self.m_jupiter)

    @classmethod
    def start(cls, sim, spkids, settings, giants, out_dir, planets=constants.GIANTS, forcing=None):
        runner = cls(sim, spkids, settings, giants, out_dir, planets, forcing)
        if forcing is not None:
            forcing(sim, runner)
        runner._init_windows_file()
        if settings.snapshots_enabled:
            runner._init_snapshots_file()
        runner._sample(accumulate=False)
        return runner

    @classmethod
    def resume(cls, spkids, settings, giants, out_dir, planets=constants.GIANTS, forcing=None):
        out_dir = Path(out_dir)
        latest = json.loads((out_dir / "latest.json").read_text())
        sim = rebound.Simulation(str(out_dir / latest["snapshot"]))
        runner = cls(sim, spkids, settings, giants, out_dir, planets, forcing)
        if forcing is not None:
            forcing(sim, runner)
        state = np.load(out_dir / latest["state"], allow_pickle=False)
        runner.check_index = int(state["check_index"])
        runner.window_index = int(state["window_index"])
        runner.sigma_prev = state["sigma_prev"]
        for key in runner.events:
            runner.events[key] = state[key]
        with h5py.File(out_dir / "windows.h5", "a") as f:
            for name in ["t_end", "class", "n_samples", *WINDOW_FIELDS, *JUPITER_FIELDS]:
                if name in f:
                    f[name].resize(runner.window_index, axis=0)
        if settings.snapshots_enabled:
            runner.snapshot_index = int(state["snapshot_index"]) if "snapshot_index" in state else 0
            with h5py.File(out_dir / "snapshots.h5", "a") as f:
                for name in [*SNAPSHOT_SCALARS, *SNAPSHOT_FIELDS]:
                    f[name].resize(runner.snapshot_index, axis=0)
        return runner

    # --- main loop ------------------------------------------------------------

    def run(self, log=print) -> None:
        total_checks = self.s.total_checks
        started = time.time()
        while self.check_index < total_checks:
            self.sim.steps(self.s.check_every_steps)
            if self.s.safe_mode == 0:
                self.sim.synchronize()
            if self.forcing is not None:
                self.forcing(self.sim, self)
            self.check_index += 1
            self._sample(accumulate=True)
            if self.check_index % self.s.checks_per_window == 0:
                self._close_window()
                last = self.check_index == total_checks
                if self.window_index % self.s.checkpoint_every_windows == 0 or last:
                    self._checkpoint()
                    alive = int(np.sum(self.events["removal_reason"] == ALIVE))
                    log(f"t = {self.sim.t / 1e6:8.3f} Myr  windows {self.window_index}/{self.s.n_windows}"
                        f"  alive {alive}/{self.n}  wall {time.time() - started:8.0f} s")

    # --- sampling -------------------------------------------------------------

    def _states(self):
        n_total = self.sim.N
        xyz = np.empty((n_total, 3))
        vxyz = np.empty((n_total, 3))
        hashes = np.empty(n_total, dtype=np.uint32)
        self.sim.serialize_particle_data(xyz=xyz, vxvyvz=vxyz, hash=hashes)
        return xyz, vxyz, hashes

    def _sample(self, accumulate: bool) -> None:
        xyz, vxyz, hashes = self._states()
        n_active = self.n_planets
        sun_x, sun_v = xyz[0], vxyz[0]
        planets = xyz[1:n_active] - sun_x
        jup_x, jup_v = planets[0], vxyz[1] - sun_v

        pos = xyz[n_active:] - sun_x
        vel = vxyz[n_active:] - sun_v
        slots = self._order[np.searchsorted(self.spkids, hashes[n_active:].astype(np.int64), sorter=self._order)]
        t = self.sim.t

        finite = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vel), axis=1)
        dist = np.linalg.norm(pos[:, None, :] - planets[None, :, :], axis=2) / self.hill[None, :]
        nearest = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        closest = dist[np.arange(len(slots)), nearest]

        ev = self.events
        enter_record = finite & (closest < self.s.record_hill_radii) & np.isnan(ev["t_hill_record"][slots])
        ev["t_hill_record"][slots[enter_record]] = t
        ev["planet_hill_record"][slots[enter_record]] = nearest[enter_record]

        hill_hit = finite & (closest < self.s.remove_hill_radii)
        r_sun = np.linalg.norm(pos, axis=1)
        far = finite & (r_sun > self.s.r_max_au)
        impact = finite & (r_sun < self.s.sun_radius_au)
        if np.any(self.planet_radius_au > 0.0):
            phys = dist * self.hill[None, :]
            impact |= finite & np.any(phys < self.planet_radius_au[None, :], axis=1)
        reason = np.where(~finite, REMOVED_NON_FINITE, np.where(impact, REMOVED_COLLISION, np.where(
            hill_hit, REMOVED_HILL, np.where(far, REMOVED_R_MAX, ALIVE))))
        gone = reason != ALIVE
        ev["t_hill_remove"][slots[hill_hit]] = t
        ev["planet_hill_remove"][slots[hill_hit]] = nearest[hill_hit]
        ev["t_removed"][slots[gone]] = t
        ev["removal_reason"][slots[gone]] = reason[gone]

        keep = ~gone
        k_slots = slots[keep]
        mu = constants.G * constants.CENTRAL_MASS
        el = diagnostics.elements(pos[keep], vel[keep], mu)
        jup = diagnostics.elements(jup_x[None], jup_v[None], constants.G * (constants.CENTRAL_MASS + self.m_jupiter))
        jup = {k: float(v[0]) for k, v in jup.items()}
        self._jupiter_now = (jup["a"], self.m_jupiter)
        sigma = diagnostics.resonant_angle(el["lam"], jup["lam"])

        crossed180, crossed0 = sigma_crossings(self.sigma_prev[k_slots], sigma)
        first180 = crossed180 & np.isnan(ev["t_sigma_cross180"][k_slots])
        first0 = crossed0 & np.isnan(ev["t_sigma_cross0"][k_slots])
        ev["t_sigma_cross180"][k_slots[first180]] = t
        ev["t_sigma_cross0"][k_slots[first0]] = t
        self.sigma_prev[k_slots] = sigma

        if accumulate:
            A, B, C = diagnostics.abc(el["a"], el["e"], el["inc"], el["Omega"], el["varpi"], sigma, jup, self.mass_ratio)
            w = self.window
            w["n"][k_slots] += 1
            for name, values in (("a", el["a"]), ("e", el["e"]), ("inc", el["inc"]), ("A", A), ("B", B), ("C", C)):
                w[name][k_slots] += values
            rad = np.radians(sigma)
            w["sin"][k_slots] += np.sin(rad)
            w["cos"][k_slots] += np.cos(rad)
            w["sigma_min"][k_slots] = np.minimum(w["sigma_min"][k_slots], sigma)
            w["sigma_max"][k_slots] = np.maximum(w["sigma_max"][k_slots], sigma)
            w["crossed0"][k_slots] |= crossed0
            w["crossed180"][k_slots] |= crossed180

        if self.s.snapshot_due(self.check_index, t):
            snap = {"t": t, "jupiter_a": jup["a"], "jupiter_m": self.m_jupiter, "jupiter_lam": jup["lam"]}
            columns = (("a", el["a"]), ("e", el["e"]), ("inc", el["inc"]), ("sigma", sigma),
                       ("x", pos[keep, 0]), ("y", pos[keep, 1]), ("z", pos[keep, 2]))
            for name, values in columns:
                row = np.full(self.n, np.nan, dtype=np.float32)
                row[k_slots] = values
                snap[name] = row
            self._snapshot_buffer.append(snap)
            self.snapshot_index += 1

        for h in hashes[n_active:][gone]:
            self.sim.remove(hash=int(h))
        if gone.any():
            self.recalculate_coordinates()

    # --- windows and checkpoints ----------------------------------------------

    def _reset_window(self) -> None:
        n = self.n
        self.window = {
            "n": np.zeros(n, np.int32),
            **{name: np.zeros(n) for name in ["a", "e", "inc", "A", "B", "C", "sin", "cos"]},
            "sigma_min": np.full(n, np.inf), "sigma_max": np.full(n, -np.inf),
            "crossed0": np.zeros(n, bool), "crossed180": np.zeros(n, bool),
        }

    def _close_window(self) -> None:
        w = self.window
        n = w["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            means = {name: np.where(n > 0, w[name] / n, np.nan) for name in ["a", "e", "inc", "A", "B", "C"]}
        cmean = np.degrees(np.arctan2(w["sin"], w["cos"]))
        record = {
            "t_end": self.sim.t,
            "class": classify_windows(n, w["crossed0"], w["crossed180"], cmean),
            "n_samples": n.astype(np.int32),
            **{k: v.astype(np.float32) for k, v in means.items()},
            "sigma_min": np.where(n > 0, w["sigma_min"], np.nan).astype(np.float32),
            "sigma_max": np.where(n > 0, w["sigma_max"], np.nan).astype(np.float32),
            "sigma_cmean": np.where(n > 0, cmean, np.nan).astype(np.float32),
            "jupiter_a": self._jupiter_now[0], "jupiter_m": self._jupiter_now[1],
        }
        self._window_buffer.append(record)
        self.window_index += 1
        self._reset_window()

    def _init_windows_file(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.out_dir / "windows.h5", "w") as f:
            f.attrs["settings"] = json.dumps(asdict(self.s))
            f.attrs["class_names"] = json.dumps(CLASS_NAMES)
            f.create_dataset("spkid", data=self.spkids)
            for name in ["t_end", *JUPITER_FIELDS]:
                f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            for name, dtype in [("class", "i1"), ("n_samples", "i4")] + [(k, "f4") for k in WINDOW_FIELDS]:
                f.create_dataset(name, shape=(0, self.n), maxshape=(None, self.n), dtype=dtype,
                                 chunks=(1, self.n), compression="lzf")

    def _init_snapshots_file(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.out_dir / "snapshots.h5", "w") as f:
            f.attrs["settings"] = json.dumps(asdict(self.s))
            f.create_dataset("spkid", data=self.spkids)
            for name in SNAPSHOT_SCALARS:
                f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            for name in SNAPSHOT_FIELDS:
                f.create_dataset(name, shape=(0, self.n), maxshape=(None, self.n), dtype="f4",
                                 chunks=(1, self.n), compression="lzf")

    def _flush_snapshots(self) -> None:
        """Append buffered snapshots, then read the last one back (the SSD has corrupted files silently)."""
        if not self._snapshot_buffer:
            return
        with h5py.File(self.out_dir / "snapshots.h5", "a") as f:
            for snap in self._snapshot_buffer:
                i = f["t"].shape[0]
                for name in [*SNAPSHOT_SCALARS, *SNAPSHOT_FIELDS]:
                    f[name].resize(i + 1, axis=0)
                    f[name][i] = snap[name]
            n_rows = f["t"].shape[0]
            last = self._snapshot_buffer[-1]
            ok = n_rows == self.snapshot_index and f["t"][-1] == last["t"] and all(
                np.array_equal(f[name][-1], last[name], equal_nan=True) for name in SNAPSHOT_FIELDS)
        if not ok:
            raise IOError(f"snapshots.h5 read-back failed in {self.out_dir}")
        self._snapshot_buffer = []

    def _checkpoint(self) -> None:
        with h5py.File(self.out_dir / "windows.h5", "a") as f:
            has_jupiter = all(name in f for name in JUPITER_FIELDS)
            for record in self._window_buffer:
                i = f["t_end"].shape[0]
                f["t_end"].resize(i + 1, axis=0)
                f["t_end"][i] = record["t_end"]
                for name in ["class", "n_samples", *WINDOW_FIELDS]:
                    f[name].resize(i + 1, axis=0)
                    f[name][i] = record[name]
                if has_jupiter:
                    for name in JUPITER_FIELDS:
                        f[name].resize(i + 1, axis=0)
                        f[name][i] = record[name]
        self._window_buffer = []
        self._flush_snapshots()

        tag = f"w{self.window_index:06d}"
        snapshot, state = f"checkpoint_{tag}.bin", f"state_{tag}.npz"
        self.sim.save_to_file(str(self.out_dir / snapshot), delete_file=True)
        np.savez(self.out_dir / state, check_index=self.check_index, window_index=self.window_index,
                 snapshot_index=self.snapshot_index, sigma_prev=self.sigma_prev, **self.events)
        self.events_frame().to_parquet(self.out_dir / "events.parquet", index=False)

        previous = self.out_dir / "latest.json"
        old = json.loads(previous.read_text()) if previous.exists() else None
        write_json(previous, {"snapshot": snapshot, "state": state, "t_yr": self.sim.t,
                              "window_index": self.window_index, "check_index": self.check_index})
        if old and old["snapshot"] != snapshot:
            for name in (old["snapshot"], old["state"]):
                (self.out_dir / name).unlink(missing_ok=True)

    def events_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"spkid": self.spkids, **self.events})
        frame["removal_reason"] = pd.Categorical.from_codes(frame["removal_reason"], REMOVAL_NAMES)
        for col in ["planet_hill_record", "planet_hill_remove"]:
            frame[col] = [self.planets[i] if i >= 0 else None for i in frame[col]]
        return frame


def settings_from_config(config: dict, dt_yr: float) -> Settings:
    """Settings from a run config.

    ``[integrator] dt_scale`` (benchmarks only) rescales the IC time step.
    ``[particles]`` (mass runs only) gives the Trojans mass; absent, they stay massless.
    """
    fields = {**config.get("integrator", {}), **config.get("diagnostics", {}), **config.get("escape", {})}
    dt_scale = float(fields.pop("dt_scale", 1.0))
    particles = config.get("particles", {})
    return Settings(t_end_yr=float(config["run"]["t_end_yr"]), dt_yr=dt_yr * dt_scale,
                    trojan_mass_msun=float(particles.get("trojan_mass_msun", 0.0)),
                    self_gravity=bool(particles.get("self_gravity", False)), **fields)


def write_settings(directory: Path, settings: Settings) -> None:
    write_json(directory / "settings.json", asdict(settings))
