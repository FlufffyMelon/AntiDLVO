"""
Logger class for Monte Carlo simulation output and trajectory writing.
Features organized output directories with timestamps and numerical data analysis.
"""

import time
import pandas as pd
from datetime import datetime
from typing import Optional, TextIO, Dict, List, Any
from pathlib import Path
from .system import System
from .topology import Topology
from .units import Units
import numpy as np
import atexit


class Logger:
    """
    Logger class for handling simulation output, logging, and data analysis.
    Creates organized output directories with timestamps and exports data for analysis.
    """

    def __init__(
        self,
        experiment_name: str = "simulation",
        results_dir: str = "results",
        log_interval: int = 100,
        xyz_interval: int = 1000,
        overwrite: bool = True,
        units: Units = None,
    ):
        """
        Initialize the logger.

        Args:
            experiment_name: Name of the experiment for folder creation
            results_dir: Base results directory
            log_interval: Steps between log entries
            xyz_interval: Steps between XYZ frames
            overwrite: Whether to overwrite existing files
            units: Units system for labeling outputs
        """
        self.experiment_name = experiment_name
        self.base_results_dir = results_dir
        self.log_interval = log_interval
        self.xyz_interval = xyz_interval
        self.units = units or Units()

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(results_dir) / f"{experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define output file paths
        self.log_file = self.output_dir / "simulation.log"
        self.xyz_file = self.output_dir / "trajectory.xyz"
        self.data_file = self.output_dir / "simulation_data.csv"
        self.config_backup_file = self.output_dir / "config_backup.yaml"

        # Initialize files
        self._init_log_file()
        self._init_xyz_file()
        self._init_data_file()

        # Persistent file handles for streaming writes
        self._log_handle: Optional[TextIO] = open(self.log_file, "a")
        self._xyz_handle: Optional[TextIO] = open(self.xyz_file, "a")
        self._data_handle: Optional[TextIO] = open(self.data_file, "w")

        # CSV header management
        self._data_header_written: bool = False
        self._data_columns: List[str] = []
        self._action_keys: List[str] = []

        # Track simulation start time
        self.start_time = time.time()

        # Ensure safe close on interpreter exit
        atexit.register(self.finalize)

    def _init_log_file(self) -> None:
        """Initialize the log file."""
        with open(self.log_file, "w") as f:
            f.write("# Monte Carlo Simulation Log\n")
            f.write(f"# Experiment: {self.experiment_name}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Output directory: {self.output_dir}\n")
            f.write(
                f"# Units: energy={self.units.energy_label}, length={self.units.length_label}, "
            )
            f.write(
                f"temperature={self.units.temperature_label}, pressure={self.units.pressure_label}\n"
            )
            f.write(
                "Columns: Step, Time(s), N_atoms, E_pot, E_tail, E_total, Temperature, Volume, Density, P_id, P_vir, P_tail, P_total, Mu_id, Acceptance_ratio, Action_ratios\n"
            )

    def _init_xyz_file(self) -> None:
        """Initialize the XYZ trajectory file."""
        self.xyz_file.touch()

    def _init_data_file(self) -> None:
        """Initialize the CSV data file (header written on first log_step)."""
        # Header will be written lazily on first use
        pass

    def backup_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Backup the configuration used for this simulation.

        Args:
            config_dict: Configuration dictionary to backup
        """
        import yaml

        with open(self.config_backup_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def log_step(
        self, step: int, system: System, topology: Topology, sampler=None
    ) -> None:
        """
        Log a simulation step with data collection.

        Args:
            step: Current step number
            system: System object
            topology: Topology object
            sampler: Sampler object (optional)
        """
        current_time = time.time() - self.start_time

        # Compute energy and pressure with LJ tail corrections (sparse, at logging only)
        e_dict = topology.compute_energy_with_tail(system)
        p_dict = topology.compute_pressure_with_tail(system)
        # Optional wedging pressure (only if external wall potential is present)
        solvation_force = topology.compute_solvation_force(system)
        energy = e_dict["energy"]
        energy_tail = e_dict["energy_tail"]
        energy_total = e_dict["energy_total"]
        pressure_ideal = p_dict["pressure_ideal"]
        pressure_virial = p_dict["pressure_virial"]
        pressure_tail = p_dict["pressure_tail"]
        pressure_total = p_dict["pressure_total"]

        volume = system.get_volume()
        density = system.get_density()
        mu_id = (
            system.mu_id if getattr(system, "mu_id", None) is not None else float("nan")
        )

        # Acceptance ratios if sampler provided
        acceptance_total = 0.0
        action_ratios = ""
        action_data = {}

        if sampler:
            ratios = sampler.get_acceptance_ratios()
            acceptance_total = ratios.get("total", 0.0)
            action_ratios = ",".join(
                [f"{k}:{v:.3f}" for k, v in ratios.items() if k != "total"]
            )
            action_data = {f"acceptance_{k}": v for k, v in ratios.items()}

        # Write to log file (keep header format unchanged)
        line = (
            f"{step:>10d} {current_time:>10.2f} {system.N_atoms:>8d} "
            f"{energy:>15.6e} {energy_tail:>15.6e} {energy_total:>15.6e} {system.temp:>10.2f} {volume:>12.4f} "
            f"{density:>12.6f} {pressure_ideal:>12.6f} {pressure_virial:>12.6f} {pressure_tail:>12.6f} {pressure_total:>12.6f} {mu_id:>15.6e} {acceptance_total:>12.4f} {action_ratios}"
        )
        if solvation_force is not None:
            line += f" wp={solvation_force:.6e}"
        line += "\n"
        if self._log_handle:
            self._log_handle.write(line)
            self._log_handle.flush()

        # Streaming CSV: write header lazily, then a single row per call
        if not self._data_header_written:
            base_cols = [
                "step",
                "time_s",
                "n_atoms",
                "energy_pot",
                "energy_tail",
                "energy_total",
                "temperature",
                "volume",
                "density",
                "pressure_ideal",
                "pressure_virial",
                "pressure_tail",
                "pressure_total",
                "mu_id",
                "acceptance_total",
            ]
            # Include action-specific acceptance columns (if any)
            for k in self._action_keys:
                base_cols.append(f"acceptance_{k}")
            # Include solvation_force column if present in this run
            if solvation_force is not None:
                base_cols.append("solvation_force")
            self._data_columns = base_cols
            if self._data_handle:
                self._data_handle.write(",".join(self._data_columns) + "\n")
                self._data_handle.flush()
            self._data_header_written = True

        # Assemble the row mapping
        row_map: Dict[str, Any] = {
            "step": step,
            "time_s": current_time,
            "n_atoms": system.N_atoms,
            "energy_pot": energy,
            "energy_tail": energy_tail,
            "energy_total": energy_total,
            "temperature": system.temp,
            "volume": volume,
            "density": density,
            "pressure_ideal": pressure_ideal,
            "pressure_virial": pressure_virial,
            "pressure_tail": pressure_tail,
            "pressure_total": pressure_total,
            "mu_id": mu_id,
            "acceptance_total": acceptance_total,
        }
        for k in self._action_keys:
            row_map[f"acceptance_{k}"] = action_data.get(f"acceptance_{k}", 0.0)
        if "solvation_force" in self._data_columns:
            row_map["solvation_force"] = (
                float(solvation_force) if solvation_force is not None else np.nan
            )

        # Serialize and write
        if self._data_handle:
            out = []
            for col in self._data_columns:
                v = row_map.get(col)
                if isinstance(v, float):
                    out.append(f"{v:.6e}")
                else:
                    out.append(str(v))
            self._data_handle.write(",".join(out) + "\n")
            self._data_handle.flush()

    def _save_data_to_csv(self) -> None:
        """Save collected simulation data to CSV file."""
        if not self.simulation_data:
            return

        df = pd.DataFrame(self.simulation_data)
        df.to_csv(self.data_file, index=False)

    def write_xyz(self, system: System, step: int, topology: Topology = None) -> None:
        """
        Write current system configuration to extended XYZ file.

        Args:
            system: System object
            step: Current step number
            topology: Topology object for energy calculation (optional)
        """
        if system.N_atoms == 0:
            return

        positions, _, _, types, names, charges, masses, _ = system.get_active_atoms()

        # Build extxyz header key=value pairs
        # Lattice as 9-vector row-major
        a, b, c = system.box
        lattice = [float(a), 0.0, 0.0, 0.0, float(b), 0.0, 0.0, 0.0, float(c)]
        lattice_str = " ".join(f"{v:.2f}" for v in lattice)

        # pbc as 3-vector of bools
        pbc_vals = ["T" if bool(x) else "F" for x in system.pbc]
        pbc_str = " ".join(pbc_vals)

        # Properties string describing per-atom columns
        properties_str = 'Properties="species:S:1:pos:R:3:charge:R:1:mass:R:1:type:I:1"'

        # Optional energy
        energy_str = (
            f"energy={topology.get_energy(system):.6f}" if topology else "energy=nan"
        )

        # Time and temperature, pressure
        elapsed = time.time() - self.start_time
        temp_str = f"temperature={system.temp:.6f}"
        press_str = (
            f"pressure={system.pressure:.6f}"
            if system.pressure is not None
            else "pressure=nan"
        )
        step_str = f"step={step}"
        time_str = f"time={elapsed:.6f}"
        units_str = (
            f'units_energy="{self.units.energy_label}" units_length="{self.units.length_label}" '
            f'units_temperature="{self.units.temperature_label}" units_pressure="{self.units.pressure_label}"'
        )
        ensemble_str = f'ensemble="{system.ensemble}"'

        # Create a view of positions in [0, L) for PBC axes, leave non-PBC axes unchanged
        pos_out = positions.copy()
        for dim in range(3):
            if system.pbc[dim]:
                L = system.box[dim]
                # numpy.mod maps negatives into [0, L)
                pos_out[:, dim] = np.mod(pos_out[:, dim], L)

        if self._xyz_handle:
            self._xyz_handle.write(f"{system.N_atoms}\n")
            self._xyz_handle.write(
                f'Lattice="{lattice_str}" pbc="{pbc_str}" {properties_str} {energy_str} {temp_str} {press_str} {step_str} {time_str} {units_str} {ensemble_str}\n'
            )

            # Per-atom lines in the order specified by Properties
            for i in range(system.N_atoms):
                species = names[i] if names[i] else "X"
                x, y, z = pos_out[i]
                q = charges[i]
                m = masses[i]
                t = int(types[i])
                self._xyz_handle.write(
                    f"{species} {x:>12.6f} {y:>12.6f} {z:>12.6f} {q:>12.6f} {m:>12.6f} {t}\n"
                )
            self._xyz_handle.flush()

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._log_handle:
            self._log_handle.write(f"# INFO [{timestamp}]: {message}\n")
            self._log_handle.flush()

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._log_handle:
            self._log_handle.write(f"# WARNING [{timestamp}]: {message}\n")
            self._log_handle.flush()

    def log_error(self, message: str) -> None:
        """Log an error message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self._log_handle:
            self._log_handle.write(f"# ERROR [{timestamp}]: {message}\n")
            self._log_handle.flush()

    def log_simulation_start(self, system: System, config: dict) -> None:
        """Log simulation startup information."""
        self.log_info("Simulation started")
        self.log_info(f"Ensemble: {system.ensemble}")
        self.log_info(f"Temperature: {system.temp} {self.units.temperature_label}")
        if system.pressure is not None:
            self.log_info(f"Pressure: {system.pressure} {self.units.pressure_label}")
        if system.mu is not None:
            self.log_info(f"Chemical potential: {system.mu} {self.units.energy_label}")
        self.log_info(f"Initial atoms: {system.N_atoms}")
        self.log_info(f"Box dimensions: {system.box} {self.units.length_label}")
        self.log_info(f"PBC: {system.pbc}")
        self.log_info(f"Output directory: {self.output_dir}")

        # Backup configuration
        self.backup_config(config)

    def log_simulation_end(self, system: System, sampler=None) -> None:
        """Log simulation completion information."""
        total_time = time.time() - self.start_time
        self.log_info("Simulation completed")
        self.log_info(f"Final atoms: {system.N_atoms}")
        self.log_info(f"Total time: {total_time:.2f} seconds")

        if sampler:
            self.log_info(f"Total moves: {sampler.n_moves}")
            ratios = sampler.get_acceptance_ratios()
            for action, ratio in ratios.items():
                self.log_info(f"{action} acceptance ratio: {ratio:.4f}")

        # Ensure flush of streaming files
        if self._log_handle:
            try:
                self._log_handle.flush()
            except Exception:
                pass
        if self._xyz_handle:
            try:
                self._xyz_handle.flush()
            except Exception:
                pass
        if getattr(self, "_data_handle", None):
            try:
                self._data_handle.flush()
            except Exception:
                pass

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir

    def finalize(self) -> None:
        """Finalize logging and close file handles."""
        try:
            if self._log_handle:
                self._log_handle.close()
        finally:
            self._log_handle = None

        try:
            if self._xyz_handle:
                self._xyz_handle.close()
        finally:
            self._xyz_handle = None

        try:
            if getattr(self, "_data_handle", None):
                self._data_handle.close()
        finally:
            self._data_handle = None

    def __del__(self):
        """Cleanup when logger is destroyed."""
        self.finalize()


# Backward compatibility alias
AdvancedLogger = Logger
