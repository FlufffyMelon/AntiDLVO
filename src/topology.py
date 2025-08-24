"""
Topology class for managing force field interactions and calculating energies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from forces import *
from .system import System
from .units import Units
from .profiler import Profiler


class Topology:
    """
    Topology class that defines all interactions between atom types
    and provides methods for energy calculations.
    """

    def __init__(self, units: Units = None, profiler: Optional[Profiler] = None):
        """
        Initialize topology.

        Args:
            units: Units system to use
        """
        self.units = units or Units()
        self.profiler = profiler or Profiler(enabled=False)
        # Dictionary mapping (type_i, type_j) -> list of Force objects
        self.interactions: Dict[Tuple[int, int], List[Force]] = {}
        # One-body forces by type
        self.one_body_forces: Dict[int, List[OneBodyForce]] = {}
        # Mapping from type name to integer id (optional, for configuration convenience)
        self.type_name_to_id: Dict[str, int] = {}
        self.type_id_to_name: Dict[int, str] = {}
        # Per-type properties (e.g., mass, charge)
        self.type_properties: Dict[int, Dict[str, float]] = {}
        # Store last computed deltas for sampler cache updates
        # self._last_delta_energy: float = 0.0
        # self._last_delta_virial: float = 0.0

    def register_type(self, type_id: int, name: str) -> None:
        """Register a mapping between a human-readable type name and id."""
        self.type_name_to_id[name] = type_id
        self.type_id_to_name[type_id] = name

    def set_type_properties(self, type_id: int, props: Dict[str, float]) -> None:
        """Set per-type properties such as mass and charge."""
        if type_id not in self.type_id_to_name:
            raise ValueError(f"Unknown type id {type_id} when setting properties")
        self.type_properties[type_id] = dict(props)

    def get_type_property(
        self, type_id: int, key: str, default: Optional[float] = None
    ) -> Optional[float]:
        return self.type_properties.get(type_id, {}).get(key, default)

    def add_interaction(self, type1: int, type2: int, force: Force) -> None:
        """
        Add an interaction between two atom types.
        """
        key = (min(type1, type2), max(type1, type2))
        if key not in self.interactions:
            self.interactions[key] = []
        self.interactions[key].append(force)

    def remove_interaction(self, type1: int, type2: int) -> None:
        key = (min(type1, type2), max(type1, type2))
        if key in self.interactions:
            del self.interactions[key]

    def add_one_body_force(self, atom_type: int, force: OneBodyForce) -> None:
        self.one_body_forces.setdefault(atom_type, []).append(force)

    def get_interactions(self, type1: int, type2: int) -> List[Force]:
        key = (min(type1, type2), max(type1, type2))
        return self.interactions.get(key, [])

    def _single_pair_energy_forces_virial(
        self,
        position: np.ndarray,
        atom_type: int,
        charge: float,
        system: System,
        exclude_index: Optional[Union[int, np.ndarray]] = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Calculate only PAIR energy, net force on the particle, and virial contributions
        for one particle interacting with other particles (excluding indices as requested).

        Returns:
            Tuple of (energy_scalar, force_on_i, virial_scalar) where:
            - energy_scalar: float of the sum of pair energies to neighbors considered
            - force_on_i: (3,) net force on the particle from considered neighbors
            - virial_scalar: float of the sum of the pair virial contributions sum_j r_ij · F_ij
        """
        positions, types, names, charges, masses = system.get_active_atoms()
        N = system.N
        if N == 0:
            return 0.0, np.zeros(3), 0.0

        idx_all = np.arange(N)
        if exclude_index is not None:
            if np.isscalar(exclude_index):
                mask_valid = idx_all != int(exclude_index)
            else:
                mask_valid = ~np.isin(idx_all, exclude_index)
        else:
            mask_valid = np.ones(N, dtype=bool)

        pos_j = positions[mask_valid]
        types_j = types[mask_valid]
        charges_j = charges[mask_valid]

        # MIC distance vectors to all others
        dr_ij = pos_j - position[np.newaxis, :]
        dr_ij = np.where(
            system.pbc[np.newaxis, :],
            dr_ij
            - system.box[np.newaxis, :]
            * np.round(dr_ij * system.inv_box[np.newaxis, :]),
            dr_ij,
        )

        # Initialize outputs
        energy_scalar = 0.0
        virial_scalar = 0.0
        net_force_i = np.zeros(3)

        # Process pairwise interactions
        unique_pairs = set((int(atom_type), int(t2)) for t2 in np.unique(types_j))
        for t1, t2 in unique_pairs:
            forces = self.get_interactions(t1, t2)
            if not forces:
                continue
            # Mask for this neighbor type
            mask_t = types_j == t2
            dr_ij_t = dr_ij[mask_t]
            charges_t = charges_j[mask_t]

            for f in forces:
                energy, forces_matrix = f(
                    position[np.newaxis, :],
                    dr_ij_t,
                    float(charge),
                    float(charges_t[0]),
                    system,
                )
                energy_scalar += energy
                virial_scalar += np.sum(forces_matrix[: dr_ij_t.shape[0], :] * dr_ij_t)
                # Force on i is opposite to forces on j from i
                net_force_i += np.sum(forces_matrix, axis=0)

        return energy_scalar, net_force_i, virial_scalar

    def _single_one_body_energy_forces(
        self,
        position: np.ndarray,
        atom_type: int,
        charge: float,
        system: System,
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate ONE-BODY energy, net force for a single particle.
        Returns (energy_scalar, force_vector).
        """
        energy_scalar = 0.0
        force_vec = np.zeros(3)
        t_i = int(atom_type)
        if t_i in self.one_body_forces:
            for ob in self.one_body_forces[t_i]:
                energy, forces = ob(position, t_i, float(charge), system=system)
                energy_scalar += energy
                force_vec += np.sum(forces, axis=0)

        return energy_scalar, force_vec

    def _single_energy_forces_virial(
        self,
        position: np.ndarray,
        atom_type: int,
        charge: float,
        system: System,
        exclude_index: Optional[Union[int, np.ndarray]] = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Wrapper that combines pair and one-body contributions for a single particle.

        Returns:
            (energy_scalar, force_on_i, virial_scalar)
        """
        pair_e, pair_force, pair_w = self._single_pair_energy_forces_virial(
            position, atom_type, charge, system, exclude_index=exclude_index
        )
        one_e, one_force = self._single_one_body_energy_forces(
            position, atom_type, charge, system
        )
        energy_scalar = pair_e + one_e
        force_on_i = pair_force + one_force
        virial_scalar = pair_w

        return energy_scalar, force_on_i, virial_scalar

    def recompute_caches(self, system: System) -> None:
        """Compute and cache total potential energy and virial.
        Ensures one-body contributions are not halved, while pair contributions are.
        """
        energy, virial = self.compute_energy_virial(system)
        system.potential_energy = energy
        system.virial = virial

    def compute_energy_virial(self, system: System) -> Tuple[float, float]:
        """Compute total potential energy and virial using split pair/one-body paths."""
        pair_energy, virial = self._total_pair_energy_virial(system)
        one_energy = self._total_one_body_energy(system)

        # Add Ewald k-space, self-energy, and dipole correction contributions if enabled
        ewald_energy = 0.0
        if system.ewald_handler:
            # K-space contribution
            kspace_energy = system.ewald_handler.compute_total_kspace_energy()

            # Self-energy correction
            positions, types, names, charges, masses = system.get_active_atoms()
            self_energy = system.ewald_handler.compute_total_self_energy(charges)

            # Dipole correction for slab geometry
            dipole_energy = system.ewald_handler.compute_dipole_correction(
                system.get_volume()
            )
            ewald_energy += kspace_energy + self_energy + dipole_energy

        return pair_energy + one_energy + ewald_energy, virial

    def _total_pair_energy_virial(self, system: System) -> Tuple[float, float]:
        """
        Total PAIR energy and virial using j > i optimization.
        Returns (energy_total, virial_total).
        """
        if system.N <= 1:
            return 0.0, 0.0
        positions, types, names, charges, masses = system.get_active_atoms()
        energy_total = 0.0
        virial_total = 0.0
        for i in range(system.N):
            energy, _, virial = self._single_pair_energy_forces_virial(
                positions[i],
                int(types[i]),
                float(charges[i]),
                system,
                exclude_index=np.arange(0, i + 1, dtype=int),
            )
            energy_total += energy
            virial_total += virial

        return energy_total, virial_total

    def _total_one_body_energy(self, system: System) -> float:
        """
        Total ONE-BODY energy vectorized over positions per type.
        Returns (energy_total).
        """
        if system.N == 0:
            return 0.0
        positions, types, names, charges, masses = system.get_active_atoms()
        energy_total = 0.0
        unique_types = np.unique(types)
        for t in unique_types:
            t_int = int(t)
            mask_t = types == t
            pos_t = positions[mask_t]
            q_t = charges[mask_t]
            # Accumulate over all one-body forces for this type
            if t_int in self.one_body_forces:
                for ob in self.one_body_forces[t_int]:
                    energy, _ = ob(pos_t, t_int, q_t, system=system)
                    energy_total += energy

        return energy_total

    def get_energy(self, system: System) -> float:
        if system.potential_energy is None:
            self.recompute_caches(system)
        return float(system.potential_energy or 0.0)

    def get_energy_difference_translation(
        self, atom_id: int, new_position: np.ndarray, system: System
    ) -> float:
        if atom_id >= system.N or atom_id < 0:
            raise IndexError(f"Atom index {atom_id} out of range")
        positions, types, names, charges, masses = system.get_active_atoms()
        old_position = positions[atom_id]
        # if self._has_ewald():
        # Energy with old position
        # energy_before, _ = self.compute_energy_virial(system)
        energy_before, _, virial_old = self._single_energy_forces_virial(
            old_position,
            int(types[atom_id]),
            float(charges[atom_id]),
            system,
            exclude_index=atom_id,
        )
        # Temporarily set new position
        # positions[atom_id] = new_position
        # system.ewald_update_structure_factors()
        # energy_after, _ = self.compute_energy_virial(system)
        energy_after, _, virial_new = self._single_energy_forces_virial(
            new_position,
            int(types[atom_id]),
            float(charges[atom_id]),
            system,
            exclude_index=atom_id,
        )
        # Restore old position
        # positions[atom_id] = old_position
        delta_energy = energy_after - energy_before
        delta_virial = virial_new - virial_old

        if system.ewald_handler:
            # K-space contribution
            delta_kspace_energy = system.ewald_handler.delta_kspace_energy(
                new_position, old_position, charges[atom_id]
            )

            # Dipole correction for slab geometry
            delta_dipole_energy = system.ewald_handler.delta_dipole_correction(
                new_position, old_position, charges[atom_id], system.get_volume()
            )

            delta_energy += delta_kspace_energy + delta_dipole_energy

        # self._last_delta_energy = delta_energy
        # self._last_delta_virial = delta_virial

        return delta_energy, delta_virial
        # energy_old, _, virial_old = self._single_energy_forces_virial(
        #     old_position,
        #     int(types[atom_id]),
        #     float(charges[atom_id]),
        #     system,
        #     exclude_index=atom_id,
        # )
        # energy_new, _, virial_new = self._single_energy_forces_virial(
        #     new_position,
        #     int(types[atom_id]),
        #     float(charges[atom_id]),
        #     system,
        #     exclude_index=atom_id,
        # )
        # self._last_delta_energy = energy_new - energy_old
        # self._last_delta_virial = virial_new - virial_old

        # return self._last_delta_energy

    def get_energy_insertion(
        self, position: np.ndarray, atom_type: int, charge: float, system: System
    ) -> float:
        # if self._has_ewald():
        # energy_before, _ = self.compute_energy_virial(system)
        # Temporarily add atom
        # idx = system.add_atom(
        #     position,
        #     int(atom_type),
        #     self.type_id_to_name.get(int(atom_type), f"T{atom_type}"),
        #     float(charge),
        #     float(self.get_type_property(int(atom_type), "mass", 1.0) or 1.0),
        # )
        # system.ewald_update_structure_factors()
        # energy_after, _ = self.compute_energy_virial(system)
        delta_energy, _, delta_virial = self._single_energy_forces_virial(
            position,
            int(atom_type),
            float(charge),
            system,
            exclude_index=None,
        )
        # Remove temporary atom
        # system.remove_atom(idx)
        # system.ewald_update_structure_factors()
        # self._last_delta_energy = energy
        # self._last_delta_virial = virial
        if system.ewald_handler:
            # K-space contribution
            delta_kspace_energy = system.ewald_handler.delta_kspace_energy(
                position, None, charge
            )

            # Self-energy correction
            delta_self_energy = system.ewald_handler.delta_self_energy(charge)

            # Dipole correction for slab geometry
            delta_dipole_energy = system.ewald_handler.delta_dipole_correction(
                position, None, charge, system.get_volume()
            )

            delta_energy += delta_kspace_energy
            delta_energy += delta_self_energy
            delta_energy += delta_dipole_energy

        return delta_energy, delta_virial

        # energy, _, virial = self._single_energy_forces_virial(
        #     position,
        #     int(atom_type),
        #     float(charge),
        #     system,
        #     exclude_index=None,
        # )
        # self._last_delta_energy = energy
        # self._last_delta_virial = virial

        # return self._last_delta_energy

    def get_energy_deletion(self, atom_id: int, system: System) -> float:
        """Compute single-atom energy/virial for deletion and store negative deltas for caches."""
        if atom_id >= system.N or atom_id < 0:
            raise IndexError(f"Atom index {atom_id} out of range")
        positions, types, names, charges, masses = system.get_active_atoms()
        # if self._has_ewald():
        # energy_before, _ = self.compute_energy_virial(system)
        # Temporarily remove atom
        # saved = (
        #     positions[atom_id].copy(),
        #     int(types[atom_id]),
        #     names[atom_id],
        #     float(charges[atom_id]),
        #     float(masses[atom_id]),
        # )
        # system.remove_atom(atom_id)
        # system.ewald_update_structure_factors()
        # energy_after, _ = self.compute_energy_virial(system)
        delta_energy, _, delta_virial = self._single_energy_forces_virial(
            positions[atom_id],
            types[atom_id],
            charges[atom_id],
            system,
            exclude_index=atom_id,
        )
        delta_energy = -delta_energy
        delta_virial = -delta_virial
        # Add back atom at original tail position
        # system.add_atom(*saved)
        # system.ewald_update_structure_factors()
        # self._last_delta_energy = -energy
        # self._last_delta_virial = -virial

        if system.ewald_handler:
            # K-space contribution
            delta_kspace_energy = system.ewald_handler.delta_kspace_energy(
                None, positions[atom_id], charges[atom_id]
            )

            # Self-energy correction
            delta_self_energy = -system.ewald_handler.delta_self_energy(
                charges[atom_id]
            )

            # Dipole correction for slab geometry
            delta_dipole_energy = system.ewald_handler.delta_dipole_correction(
                None, positions[atom_id], charges[atom_id], system.get_volume()
            )

            delta_energy += delta_kspace_energy
            delta_energy += delta_self_energy
            delta_energy += delta_dipole_energy

        return delta_energy, delta_virial

        # energy, _, virial = self._single_energy_forces_virial(
        #     positions[atom_id],
        #     int(types[atom_id]),
        #     float(charges[atom_id]),
        #     system,
        #     exclude_index=atom_id,
        # )
        # self._last_delta_energy = -energy
        # self._last_delta_virial = -virial

        # return energy

    def compute_energy_with_tail(self, system: System) -> Dict[str, float]:
        if system.potential_energy is None:
            with self.profiler.measure("recompute_caches"):
                self.recompute_caches(system)
        density = system.get_density()
        rc = self._get_lj_cutoff()
        energy_tail = 0.0
        if rc is not None and density > 0:
            inv_rc3 = (1.0 / rc) ** 3
            inv_rc9 = inv_rc3**3
            energy_tail = (
                (8.0 / 3.0) * np.pi * density * ((1.0 / 3.0) * inv_rc9 - inv_rc3)
                # * system.N
            )
        return {
            "energy": float(system.potential_energy or 0.0),
            "energy_tail": energy_tail,
            "energy_total": float(system.potential_energy or 0.0) + energy_tail,
        }

    def compute_pressure_with_tail(self, system: System) -> Dict[str, float]:
        if system.virial is None:
            with self.profiler.measure("recompute_caches"):
                self.recompute_caches(system)
        volume = system.get_volume()
        density = system.get_density()
        kbT = (
            (1.0 * system.temp)
            if self.units.system_name == "lj"
            else (8.314462618e-3 * system.temp)
        )
        pressure_ideal = density * kbT
        pressure_config = (
            (float(system.virial or 0.0) / (3.0 * volume)) if system.N > 1 else 0.0
        )
        pressure = pressure_ideal + pressure_config
        rc = self._get_lj_cutoff()
        pressure_tail = 0.0
        if rc is not None and density > 0:
            inv_rc3 = (1.0 / rc) ** 3
            inv_rc9 = inv_rc3**3
            pressure_tail = (
                (16.0 / 3.0) * np.pi * (density**2) * ((2.0 / 3.0) * inv_rc9 - inv_rc3)
            )
        return {
            "pressure": pressure,
            "pressure_tail": pressure_tail,
            "pressure_total": pressure + pressure_tail,
        }

    def compute_solvation_force(self, system: System) -> Optional[float]:
        """Compute solvation force f_s = mean(dU_w/dz) / A when ExternalWallPotential is present.

        Returns None if no ExternalWallPotential is configured for any type.
        """
        # Identify if any types have an ExternalWallPotential one-body force
        types_with_wall = False
        for t_id, flist in self.one_body_forces.items():
            walls = [f for f in flist if isinstance(f, ExternalWallPotential)]
            if walls:
                types_with_wall = True
                break

        if not types_with_wall:
            return None

        positions, types, names, charges, masses = system.get_active_atoms()
        if system.N == 0:
            return None

        # A = float(system.box[0] * system.box[1])
        # if A <= 0.0:
        #     return None

        Fz_total = []
        unique_types = np.unique(types)
        for t in unique_types:
            t_int = int(t)
            if t_int not in self.one_body_forces:
                continue
            mask_t = types == t
            pos_t = positions[mask_t]
            # Accumulate over all one-body forces for this type
            for ob in self.one_body_forces[t_int]:
                if isinstance(ob, ExternalWallPotential):
                    _, force_vec = ob(pos_t, t_int, 0.0, system=system)

                    if ob.style == "bottom":
                        Fz_total.append(force_vec[:, 2])
                    elif ob.style == "top":
                        Fz_total.append(-force_vec[:, 2])

        if not Fz_total:
            return None

        Fz_total = np.concatenate(Fz_total)
        f_s = np.mean(Fz_total)

        return f_s

    def _get_lj_cutoff(self) -> Optional[float]:
        for key, flist in self.interactions.items():
            for f in flist:
                if isinstance(f, LennardJones):
                    return float(f.cutoff)
        return None

    def __str__(self) -> str:
        n_interactions = len(self.interactions)
        n_types = len(self.type_id_to_name)
        n_one_body = sum(len(v) for v in self.one_body_forces.values())
        return (
            f"Topology: {n_interactions} pair interaction types, "
            f"{n_one_body} one-body forces, {n_types} types registered"
        )
