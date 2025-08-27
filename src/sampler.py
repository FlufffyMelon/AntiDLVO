"""
Sampler class for performing Monte Carlo moves in different ensembles.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .system import System
from .topology import Topology
from .units import Units
from .logger import Logger
from .profiler import Profiler


class Sampler:
    """
    Monte Carlo sampler that performs various MC moves based on ensemble and configuration.
    """

    def __init__(
        self,
        system: System,
        topology: Topology,
        actions: List[Dict[str, Any]],
        max_displacement: float = 0.1,
        logger: Logger = None,
        units: Units = None,
        insert_type_probabilities: Optional[Dict[int, float]] = None,
        max_volume_scale: float = 0.1,
        debug: bool = False,
        profiler: Optional[Profiler] = None,
        force_recompute: bool = False,
    ):
        """
        Initialize the Monte Carlo sampler.

        Args:
                system: System object
                topology: Topology object with force field
                actions: List of dictionaries with 'action' and 'probability' keys
                max_displacement: Maximum displacement for translation moves
                logger: Logger for output
                units: Units system
                insert_type_probabilities: Optional mapping of type_id to probability for insertions (muVT)
                max_volume_scale: Maximum relative scale for volume move (delta_max in description)
                debug: Enable profiling if True
                profiler: Optional Profiler instance
        """
        self.system = system
        self.topology = topology
        self.max_displacement = max_displacement
        self.logger = logger
        self.units = units or Units()
        self.insert_type_probabilities = insert_type_probabilities
        self.max_volume_scale = max_volume_scale
        self.debug = debug
        self.profiler = profiler or Profiler(enabled=debug)
        self.force_recompute = bool(force_recompute)
        self._audit_E_before_real: Optional[float] = None

        # Validate and normalize action probabilities
        self.actions = self._validate_actions(actions)

        # Monte Carlo statistics
        self.n_moves = 0
        self.n_accepted = {"total": 0}
        self.n_attempted = {"total": 0}

        # Initialize action-specific counters
        for action_info in self.actions:
            action = action_info["action"]
            self.n_accepted[action] = 0
            self.n_attempted[action] = 0

        # Boltzmann constant (in appropriate units)
        self.kb = self._get_boltzmann_constant()

        # Physical constants for de Broglie wavelength (standard units)
        self.h = self._get_planck_constant()
        self.pi = np.pi

        # Random number generator
        self.rng = np.random.default_rng()

        # Initialize ideal gas chemical potential cache
        self.system.mu_id = self._compute_mu_id()

    def _validate_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize action probabilities."""
        if not actions:
            raise ValueError("At least one action must be specified")

        # Check probabilities sum to 1
        total_prob = sum(action["probability"] for action in actions)
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Action probabilities must sum to 1.0, got {total_prob}")

        # Validate ensemble-specific actions
        valid_actions = self._get_valid_actions()

        for action_info in actions:
            action = action_info["action"]
            if action not in valid_actions:
                raise ValueError(
                    f"Action '{action}' not valid for {self.system.ensemble} ensemble"
                )

        return actions

    def _get_valid_actions(self) -> List[str]:
        """Get valid actions for the current ensemble."""
        base_actions = ["translate"]

        if self.system.ensemble == "muVT":
            base_actions.extend(["insert", "delete"])
        elif self.system.ensemble == "NPT":
            base_actions.extend(["volume"])

        return base_actions

    def _get_boltzmann_constant(self) -> float:
        """Get Boltzmann constant in appropriate units."""
        if self.units.system_name == "lj":
            return 1.0  # kB = 1 in LJ units
        else:
            return 8.314462618e-3  # kJ/(mol⋅K)

    def _get_planck_constant(self) -> float:
        """Get Planck constant in units consistent with standard MC energies.
        We use h in kJ*ps/mol to remain consistent with nm, kJ/mol, and ps.
        h (J*s) = 6.62607015e-34; convert: J -> kJ (1e-3), s -> ps (1e12), per mole (NA),
        but since energies here are per mole, we use h per mole: h * NA. Using kJ,ps per mol:
        h_kJ_ps_per_mol = 6.62607015e-34 * 1e-3 * 6.02214076e23 * 1e12
        """
        if self.units.system_name == "lj":
            # In reduced LJ units, set h = 1 consistently with epsilon, sigma, mass scales
            return 1.0
        else:
            return 6.62607015e-34 * 1e-3 * 6.02214076e23 * 1e12

    def _compute_mu_id(self) -> float:
        """Compute ideal-gas chemical potential: mu_id = kT * ln(Lambda^3 * rho).
        Lambda is computed using average particle mass when in standard units, and 1.0 in LJ units.
        Returns NaN if density is zero.
        """
        rho = self.system.get_density()
        if self.system.N <= 0 or rho <= 0.0:
            return float("nan")
        if self.units.system_name == "lj":
            Lambda = 1.0
        else:
            avg_mass = (
                float(np.mean(self.system.masses[: self.system.N]))
                if self.system.N > 0
                else 1.0
            )
            Lambda = self._thermal_de_broglie_wavelength(avg_mass)
        kbT = self.kb * self.system.temp
        return kbT * float(np.log((Lambda**3) * rho))

    def _select_action(self) -> str:
        """Select an action based on probabilities."""
        rand = self.rng.random()
        cumulative_prob = 0.0

        for action_info in self.actions:
            cumulative_prob += action_info["probability"]
            if rand <= cumulative_prob:
                return action_info["action"]

        # Fallback to last action
        return self.actions[-1]["action"]

    def move(self, n_steps: int) -> None:
        """
        Perform n Monte Carlo steps.

        Args:
                n_steps: Number of MC steps to perform
        """
        for _ in range(n_steps):
            with self.profiler.measure("step_total"):
                # Select action
                with self.profiler.measure("select_action"):
                    action = self._select_action()

                # Perform action
                if action == "translate":
                    with self.profiler.measure("action_translate"):
                        self._attempt_translation()
                elif action == "insert":
                    with self.profiler.measure("action_insert"):
                        self._attempt_insertion()
                elif action == "delete":
                    with self.profiler.measure("action_delete"):
                        self._attempt_deletion()
                elif action == "volume":
                    with self.profiler.measure("action_volume"):
                        self._attempt_volume_change()

                self.n_moves += 1

                # Log if needed
                if self.logger and self.n_moves % self.logger.log_interval == 0:
                    with self.profiler.measure("log_step"):
                        self.logger.log_step(
                            self.n_moves, self.system, self.topology, self
                        )

                if self.logger and self.n_moves % self.logger.xyz_interval == 0:
                    with self.profiler.measure("write_xyz"):
                        self.logger.write_xyz(self.system, self.n_moves, self.topology)

    def _attempt_translation(self) -> None:
        """Attempt to translate a random atom (NVT/NPT/muVT)."""
        if self.system.N == 0:
            return

        if self.force_recompute:
            self.topology.recompute_caches(self.system)
            self._audit_E_before_real = float(self.topology.get_energy(self.system))

        # Select random atom
        atom_id = self.rng.integers(0, self.system.N)

        # Generate random displacement
        displacement = (self.rng.random(3) - 0.5) * 2 * self.max_displacement
        old_position = self.system.positions[atom_id].copy()
        new_position = self.system.apply_pbc(old_position + displacement)

        # Calculate energy difference
        with self.profiler.measure("energy_diff_translate"):
            energy_diff, delta_virial, delta_S_c, delta_S_s = (
                self.topology.get_energy_difference_translation(
                    atom_id, new_position, self.system
                )
            )

        self.n_attempted["translate"] += 1
        self.n_attempted["total"] += 1

        # Accept or reject (NVT rule also applies to NPT/muVT translations)
        beta = 1.0 / (self.kb * self.system.temp)

        accepted = energy_diff <= 0.0 or self.rng.random() < np.exp(-beta * energy_diff)
        if accepted:
            with self.profiler.measure("translate_accepted"):
                # Update position
                self.system.positions[atom_id] = new_position
                # Update caches: potential energy and virial
                if self.system.potential_energy is None or self.system.virial is None:
                    self.topology.recompute_caches(self.system)
                else:
                    # self.system.potential_energy += getattr(
                    #     self.topology, "_last_delta_energy", 0.0
                    # )
                    self.system.potential_energy += energy_diff
                    # self.system.virial += getattr(self.topology, "_last_delta_virial", 0.0)
                    self.system.virial += delta_virial
                # Update Ewald structure factors if enabled
                if self.system.ewald_handler:
                    with self.profiler.measure("ewald_update_structure_factors"):
                        # self.system.ewald_handler.update_structure_factors(
                        #     self.system.positions[: self.system.N],
                        #     self.system.charges[: self.system.N],
                        # )
                        self.system.ewald_handler.update_structure_factors_from_delta(
                            atom_id, delta_S_c, delta_S_s
                        )
                    with self.profiler.measure("ewald_update_dipole_moment"):
                        self.system.ewald_handler.update_dipole_moment(
                            self.system.positions[: self.system.N],
                            self.system.charges[: self.system.N],
                        )
                self.n_accepted["translate"] += 1
                self.n_accepted["total"] += 1
        # else:
        # Rejected; clear last deltas
        # self.topology._last_delta_energy = 0.0
        # self.topology._last_delta_virial = 0.0

        if self.force_recompute:
            self._force_recompute_audit_after(
                "translate", float(energy_diff) if accepted else 0.0
            )

    def _select_insertion_type(self) -> int:
        """Select a type for insertion based on provided probabilities or default to 0."""
        if not self.insert_type_probabilities:
            return 0
        items = list(self.insert_type_probabilities.items())
        types, probs = zip(*items)
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(types), p=probs)
        return int(types[idx])

    def _thermal_de_broglie_wavelength(self, mass: float) -> float:
        """Compute thermal de Broglie wavelength Lambda.
        Lambda = sqrt(h^2 / (2*pi*m*kT))
        Units:
          - Standard: mass in amu, convert to kg/mol consistent with h units; but since h is per mol,
                we treat m as molar mass (kg/mol) scaled appropriately. We approximate by using m in kg/mol via amu*1e-3.
          - LJ: return 1.0 by convention if using reduced units.
        """
        T = self.system.temp
        if self.units.system_name == "lj":
            return 1.0
        # convert mass amu -> kg/mol: 1 amu = 1 g/mol = 1e-3 kg/mol
        molar_mass_kg_per_mol = mass * 1e-3
        return np.sqrt(
            (self.h * self.h) / (2.0 * self.pi * molar_mass_kg_per_mol * self.kb * T)
        )

    def _force_recompute_audit_after(self, action: str, predicted_delta: float) -> None:
        """Print diagnostics for force_recompute comparing delta-based vs real energies."""
        if not self.force_recompute or self._audit_E_before_real is None:
            return
        dE_delta = float(predicted_delta)
        E_cached_after = float(self.system.potential_energy or 0.0)
        # Real energy after the move (non-mutating)
        E_real_after, _ = self.topology.compute_energy_virial(self.system)
        dE_real = E_real_after - float(self._audit_E_before_real)
        print(
            f"[force_recompute] {action}: dE(delta)={dE_delta:.2e}, dE(real)={dE_real:.2e}, "
            f"E_cached={E_cached_after:.2e}, E_real={E_real_after:.2e}"
        )
        self._audit_E_before_real = None

    def _attempt_insertion(self) -> None:
        """Attempt to insert a new atom (muVT only)."""
        if self.system.ensemble != "muVT":
            return

        if self.force_recompute:
            self.topology.recompute_caches(self.system)
            self._audit_E_before_real = float(self.topology.get_energy(self.system))

        # Generate random position
        position = self.rng.random(3) * self.system.box

        # Choose atom type
        atom_type = self._select_insertion_type()
        # Pull default properties from topology if available
        name = self.topology.type_id_to_name.get(atom_type, f"T{atom_type}")
        charge = self.topology.get_type_property(atom_type, "charge", 0.0)
        mass = self.topology.get_type_property(atom_type, "mass", 1.0)

        # Calculate insertion energy
        with self.profiler.measure("energy_insert"):
            insertion_delta_energy, insertion_delta_virial = (
                self.topology.get_energy_insertion(
                    position, atom_type, charge, self.system
                )
            )
            # print(f"insertion_energy: {insertion_energy}")

        # Thermal wavelength
        Lambda = self._thermal_de_broglie_wavelength(mass)

        # Acceptance probability (grand canonical)
        beta = 1.0 / (self.kb * self.system.temp)
        volume = self.system.get_volume()
        N = self.system.N

        log_acc = (
            np.log(volume)
            - 3.0 * np.log(Lambda)
            - np.log(N + 1)
            + beta * (self.system.mu - insertion_delta_energy)
        )

        self.n_attempted["insert"] += 1
        self.n_attempted["total"] += 1

        accepted = log_acc > 0 or self.rng.random() < np.exp(log_acc)
        if accepted:
            self.system.add_atom(position, atom_type, name, charge, mass)
            # Update caches
            if self.system.potential_energy is None or self.system.virial is None:
                self.topology.recompute_caches(self.system)
            else:
                self.system.potential_energy += insertion_delta_energy
                self.system.virial += insertion_delta_virial

            # Update Ewald structure factors if enabled
            if self.system.ewald_handler:
                self.system.ewald_handler.update_structure_factors(
                    self.system.positions[: self.system.N],
                    self.system.charges[: self.system.N],
                )
                self.system.ewald_handler.update_dipole_moment(
                    self.system.positions[: self.system.N],
                    self.system.charges[: self.system.N],
                )

            # Update ideal gas chemical potential
            self.system.mu_id = self._compute_mu_id()
            self.n_accepted["insert"] += 1
            self.n_accepted["total"] += 1
        # else:
        #     self.topology._last_delta_energy = 0.0
        #     self.topology._last_delta_virial = 0.0

        if self.force_recompute:
            self._force_recompute_audit_after(
                "insert", insertion_delta_energy if accepted else 0.0
            )

    def _attempt_deletion(self) -> None:
        """Attempt to delete a random atom (muVT only)."""
        if self.system.ensemble != "muVT" or self.system.N == 0:
            return

        if self.force_recompute:
            self.topology.recompute_caches(self.system)
            self._audit_E_before_real = float(self.topology.get_energy(self.system))

        # Select random atom
        atom_id = self.rng.integers(0, self.system.N)

        # Mass for Lambda
        atom_type = int(self.system.types[atom_id])
        mass = float(self.topology.get_type_property(atom_type, "mass", 1.0) or 1.0)
        Lambda = self._thermal_de_broglie_wavelength(mass)

        # Compute single-atom energy and set negative deltas in topology for cache update
        with self.profiler.measure("energy_delete"):
            deletion_delta_energy, deletion_delta_virial = (
                self.topology.get_energy_deletion(atom_id, self.system)
            )

        # Acceptance probability (grand canonical)
        beta = 1.0 / (self.kb * self.system.temp)
        volume = self.system.get_volume()
        N = self.system.N

        log_acc = (
            np.log(N)
            + 3.0 * np.log(Lambda)
            - np.log(volume)
            - beta * (self.system.mu + deletion_delta_energy)
        )

        self.n_attempted["delete"] += 1
        self.n_attempted["total"] += 1

        accepted = log_acc > 0 or self.rng.random() < np.exp(log_acc)
        if accepted:
            # Remove atom from arrays after caches updated by deltas
            self.system.remove_atom(atom_id)
            # Apply cached deltas to energy/virial if caches are valid, otherwise remove then recompute
            if self.system.potential_energy is None or self.system.virial is None:
                self.system.remove_atom(atom_id)
                self.topology.recompute_caches(self.system)
            else:
                self.system.potential_energy += deletion_delta_energy  # -??
                self.system.virial += deletion_delta_virial  # -??
            # Update ideal gas chemical potential
            self.system.mu_id = self._compute_mu_id()
            self.n_accepted["delete"] += 1
            self.n_accepted["total"] += 1
        # else:
        #     self.topology._last_delta_energy = 0.0
        #     self.topology._last_delta_virial = 0.0

        if self.force_recompute:
            self._force_recompute_audit_after(
                "delete",
                deletion_delta_energy if accepted else 0.0,  # -??
            )

    def _attempt_volume_change(self) -> None:
        """Attempt to change system volume (NPT only) with relative scaling."""
        if self.system.ensemble != "NPT":
            return

        if self.force_recompute:
            self.topology.recompute_caches(self.system)
            self._audit_E_before_real = float(self.topology.get_energy(self.system))

        # Ensure caches are valid
        if self.system.potential_energy is None or self.system.virial is None:
            self.topology.recompute_caches(self.system)

        # Relative volume change via side length scale
        delta = (self.rng.random() * 2.0 - 1.0) * self.max_volume_scale
        scale_factor = 1.0 + delta
        if scale_factor <= 0.0:
            return

        old_box = self.system.box.copy()
        old_positions = self.system.positions[: self.system.N].copy()
        old_volume = self.system.get_volume()
        old_energy = float(self.system.potential_energy)
        old_virial = float(self.system.virial)

        # Apply changes temporarily
        new_box = old_box * scale_factor
        self.system.update_box(new_box)
        self.system.positions[: self.system.N] *= scale_factor

        # Compute new energy/virial by recomputing caches once
        with self.profiler.measure("energy_after_volume"):
            self.topology.recompute_caches(self.system)
            new_energy = float(self.system.potential_energy)

        new_volume = self.system.get_volume()

        # Acceptance probability
        beta = 1.0 / (self.kb * self.system.temp)
        energy_diff = new_energy - old_energy
        pv_term = self.system.pressure * (new_volume - old_volume)
        # The provided formula uses (N+1); we honor that
        log_volume_term = (self.system.N + 1) * np.log(new_volume / old_volume)

        log_acc = -beta * (energy_diff + pv_term) + log_volume_term

        self.n_attempted["volume"] += 1
        self.n_attempted["total"] += 1

        accepted = log_acc > 0 or self.rng.random() < np.exp(log_acc)
        if accepted:
            # Accept: caches already reflect new state
            # Update ideal gas chemical potential (density changed)
            self.system.mu_id = self._compute_mu_id()
            self.n_accepted["volume"] += 1
            self.n_accepted["total"] += 1
        else:
            # Reject: restore old state and old caches
            self.system.update_box(old_box)
            self.system.positions[: self.system.N] = old_positions
            self.system.potential_energy = old_energy
            self.system.virial = old_virial

        if self.force_recompute:
            self._force_recompute_audit_after(
                "volume", energy_diff if accepted else 0.0
            )

    def get_acceptance_ratios(self) -> Dict[str, float]:
        """Get acceptance ratios for all move types."""
        ratios = {}
        for action in self.n_attempted:
            if self.n_attempted[action] > 0:
                ratios[action] = self.n_accepted[action] / self.n_attempted[action]
            else:
                ratios[action] = 0.0
        return ratios

    def reset_statistics(self) -> None:
        """Reset Monte Carlo statistics."""
        self.n_moves = 0
        for action in self.n_accepted:
            self.n_accepted[action] = 0
            self.n_attempted[action] = 0

    def __str__(self) -> str:
        """String representation of sampler."""
        ratios = self.get_acceptance_ratios()
        return (
            f"Sampler: {self.n_moves} moves, "
            f"total acceptance = {ratios.get('total', 0.0):.3f}"
        )
