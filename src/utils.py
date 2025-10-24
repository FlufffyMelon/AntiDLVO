"""
Utility functions for building and running Monte Carlo simulations.
Consolidates helpers used by mc_main.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from omegaconf import OmegaConf

try:
    # For robust isinstance checks with OmegaConf containers
    from omegaconf import ListConfig, DictConfig  # type: ignore
except Exception:  # pragma: no cover
    ListConfig = tuple()  # fallback to avoid NameError; not used if import works
    DictConfig = tuple()

from .ewald import EwaldHandler
from .system import System
from .topology import Topology
from .sampler import Sampler
from .forces import *
from .units import Units
from .logger import Logger
from .profiler import Profiler
from .molecule import Molecule, Particle, Dipole


def load_config(config_file: str, overrides: Optional[List[str]] = None):
    """Load configuration from YAML file using OmegaConf with interpolation/resolvers and CLI overrides.

    - Supports ${var} style interpolation (OmegaConf built-in)
    - Adds an eval resolver: ${eval:"${size}*10"} or ${eval:${size}*10}
    - Applies CLI overrides via dotlist, e.g., ["system.size=20"]
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Configuration file {config_file} not found")

    # Register a simple eval resolver with mathematical functions
    def _eval_resolver(expr: str):
        try:
            # Provide mathematical functions in the eval context
            import math
            import numpy as np

            eval_globals = {
                "sqrt": math.sqrt,
                "log": math.log,
                "exp": math.exp,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "ceil": math.ceil,
                "floor": math.floor,
                "abs": abs,
                "min": min,
                "max": max,
                "pi": math.pi,
                "e": math.e,
                "np": np,
                "math": math,
            }
            return eval(expr, eval_globals, {})
        except Exception as e:
            raise ValueError(f"Failed to eval expression '{expr}': {e}")

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", _eval_resolver, use_cache=False)

    cfg = OmegaConf.load(config_file)

    # Apply overrides if provided
    if overrides:
        dotlist = overrides
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    return cfg


def create_units(cfg) -> Units:
    """Create units system from configuration."""
    system_type = cfg.units.get("system", "standard")
    return Units(system=system_type)


def create_system(cfg, units: Units) -> System:
    """Create system from configuration."""
    ensemble = cfg.simulation.get("ensemble", "NVT")
    temperature = cfg.simulation.get("temperature", 300.0)
    pressure = cfg.simulation.get("pressure") if ensemble == "NPT" else None
    mu = cfg.simulation.get("mu") if ensemble == "muVT" else None

    # Ewald parameters
    ewald_handler = None
    if cfg.get("ewald", None) is not None:
        alpha = cfg.ewald.get("alpha", None)
        real_cut = cfg.ewald.get("real_cut", None)
        n_c = cfg.ewald.get("n_c", None)
        dielectric = cfg.ewald.get("dielectric", None)
        z_scale_factor = cfg.ewald.get("z_scale_factor", 1.0)
        dipole_correction = cfg.ewald.get("dipole_correction", False)

        if not (
            alpha is None
            or real_cut is None
            or n_c is None
            or dielectric is None
            or dipole_correction is None
        ):
            ewald_handler = EwaldHandler(
                alpha=alpha,
                real_cut=real_cut,
                n_c=n_c,
                dielectric=dielectric,
                z_scale_factor=z_scale_factor,
                dipole_correction=dipole_correction,
                units=units,
            )

    n_atoms = cfg.system.get("n_atoms", 0)
    capacity = cfg.system.get("initial_capacity", 128)
    # Resolve interpolations for box explicitly
    box_val = cfg.system.get("box", [10.0, 10.0, 10.0])
    try:
        box_resolved = OmegaConf.to_container(box_val, resolve=True)
    except Exception:
        box_resolved = box_val
    box = (
        list(box_resolved)
        if isinstance(box_resolved, (list, tuple))
        else [10.0, 10.0, 10.0]
    )

    pbc = list(cfg.system.get("pbc", [True, True, True]))

    # Create system
    system = System(
        n_atoms=0,
        initial_capacity=capacity,
        box=box,
        pbc=pbc,
        ensemble=ensemble,
        temp=temperature,
        pressure=pressure,
        mu=mu,
        ewald_handler=ewald_handler,
        units=units,
    )

    system._target_n_atoms = n_atoms
    return system


def parse_type_pairs(pair_types: Any) -> List[Tuple[str, str]]:
    """
    Parse a pair types specification which can contain nested lists with parentheses.
    For example: [Na, (Ip, Im)] will be parsed to pairs [(Na, Ip), (Na, Im)]
    Also handles string representations like '(Na, Cl)' by parsing the comma-separated values.

    Args:
        pair_types: List or tuple of types, can include nested lists with parentheses or strings

    Returns:
        List of all possible type pairs as tuples
    """
    # Ensure pair_types is a list-like structure
    if not isinstance(pair_types, (list, tuple, ListConfig)):
        raise ValueError("pair_types must be a list-like structure")

    # Ensure we have exactly 2 elements for a pair
    if len(pair_types) != 2:
        raise ValueError("Each pair must have exactly two elements")

    # Helper function to parse string groups like "(Na, Cl)" into a list of types
    def parse_group(item):
        if isinstance(item, str):
            # Check if it's a group expression like "(Na, Cl)"
            item = item.strip()
            if item.startswith("(") and item.endswith(")"):
                # Extract content within parentheses and split by comma
                content = item[1:-1]
                return [t.strip() for t in content.split(",")]
            return [item]
        elif isinstance(item, (list, tuple, ListConfig)):
            return list(item)
        else:
            return [str(item)]

    # Parse the first and second elements
    first_types = parse_group(pair_types[0])
    second_types = parse_group(pair_types[1])

    # Generate all possible pairs
    result = []
    for first in first_types:
        for second in second_types:
            result.append((str(first), str(second)))

    return result


def setup_initial_configuration(
    system: System, type_specs: list, topology: Topology, cfg=None
):
    """
    Setup initial atom configuration.

    Modes (cfg.system.init.mode):
      - "random": random positions with minimum separation (cfg.system.init.min_distance)
      - "uniform": positions on a uniform 3D grid covering the box
      - "SCC": simple cubic lattice with alternating types

    type_specs format:
    [
        {
            "Particle": {
                "type": "Na",
                "mass": float,
                "charge": float,
                "count": int
            }
        },
        {
            "Dipole": {
                "type_plus": "Ip",
                "type_minus": "Im",
                "type_ghost": "G",
                "length": float,
                "ghost_count": int,
                "mass": float,
                "charge": float,
                "count": int
            }
        }
    ]
    """
    # Create a mapping from type names to type IDs
    type_name_to_id = topology.type_name_to_id

    # Count total molecules and process the specs
    total_molecules = 0
    processed_specs = []

    for i, spec_entry in enumerate(type_specs):
        # Create local copy to avoid modifying the original
        local_spec_entry = spec_entry

        # Check if it's a dict or DictConfig
        if isinstance(local_spec_entry, dict):
            pass  # Good, it's a dict
        elif isinstance(local_spec_entry, DictConfig):
            local_spec_entry = OmegaConf.to_container(local_spec_entry, resolve=True)
        else:
            continue

        # Handle the nested dictionary structure
        if "Particle" in local_spec_entry and local_spec_entry["Particle"]:
            molecule_type = "Particle"
            spec = local_spec_entry["Particle"]
            if isinstance(spec, DictConfig):
                spec = OmegaConf.to_container(spec, resolve=True)
        elif "Dipole" in local_spec_entry and local_spec_entry["Dipole"]:
            molecule_type = "Dipole"
            spec = local_spec_entry["Dipole"]
            if isinstance(spec, DictConfig):
                spec = OmegaConf.to_container(spec, resolve=True)
        else:
            raise ValueError(f"Unknown molecule type: {local_spec_entry.keys()[0]}")

        # Get count and add to total
        count = int(spec.get("count", 0))
        if count <= 0:
            continue

        total_molecules += count
        processed_specs.append((molecule_type, spec, count))

    if total_molecules <= 0:
        return

    # Determine initialization mode
    init_mode = None
    if cfg is not None and "system" in cfg and "init" in cfg.system:
        init_mode = cfg.system.init.get("mode")
        min_distance = float(cfg.system.init.get("min_distance"))
        pad_val = cfg.system.init.get("padding")
        try:
            pad_resolved = OmegaConf.to_container(pad_val, resolve=True)
        except Exception:
            pad_resolved = pad_val
        pad_arr = np.array(list(pad_resolved)[:3], dtype=float)
        # Clamp to valid range [0, box/2)
        box = system.box
        max_pad = 0.5 * box - 1e-9
        padding = np.clip(pad_arr, 0.0, max_pad)
    init_mode = init_mode or "random"

    if init_mode == "uniform":
        _init_uniform_configuration(system, processed_specs, type_name_to_id, padding)
    elif init_mode == "SCC":
        _init_scc_configuration(system, processed_specs, type_name_to_id, padding)
    elif init_mode == "random":
        _init_random_configuration(
            system, processed_specs, type_name_to_id, min_distance, padding
        )
    else:
        raise ValueError(f"Unknown init mode: {init_mode}")

    if system.ewald_handler:
        system.ewald_handler.update_structure_factors(
            system.positions[: system.N_atoms],
            system.charges[: system.N_atoms],
        )
        system.ewald_handler.update_dipole_moment(
            system.positions[: system.N_atoms],
            system.charges[: system.N_atoms],
        )


def _create_molecule(
    position: np.ndarray,
    molecule_type: str,
    spec: dict,
    type_name_to_id: Dict[str, int],
) -> Molecule:
    """
    Create a molecule based on its specification.

    Args:
        position: Position for the molecule
        molecule_type: Type of molecule ("Particle" or "Dipole")
        spec: Specification dictionary for the molecule
        type_name_to_id: Mapping from type names to type IDs

    Returns:
        Created molecule
    """
    if molecule_type == "Particle":
        # Create a single particle
        type_name = spec.get("type", "")
        if type_name not in type_name_to_id:
            raise ValueError(f"Unknown particle type: {type_name}")

        type_id = type_name_to_id[type_name]
        name = spec.get("name", type_name) or type_name
        mass = float(spec.get("mass", 1.0))
        charge = float(spec.get("charge", 0.0))

        return Particle(
            position=position, type_id=type_id, name=name, charge=charge, mass=mass
        )

    elif molecule_type == "Dipole":
        # Create a dipole
        type_plus_name = spec.get("type_plus", "")
        type_minus_name = spec.get("type_minus", "")
        type_ghost_name = spec.get("type_ghost", "")

        if type_plus_name not in type_name_to_id:
            raise ValueError(f"Unknown positive dipole type: {type_plus_name}")
        if type_minus_name not in type_name_to_id:
            raise ValueError(f"Unknown negative dipole type: {type_minus_name}")
        if type_ghost_name and type_ghost_name not in type_name_to_id:
            raise ValueError(f"Unknown ghost dipole type: {type_ghost_name}")

        type_plus_id = type_name_to_id[type_plus_name]
        type_minus_id = type_name_to_id[type_minus_name]
        type_ghost_id = (
            type_name_to_id.get(type_ghost_name) if type_ghost_name else None
        )

        length = float(spec.get("length", 0.1))
        ghost_count = int(spec.get("ghost_count", 0))
        name = spec.get("name", "dipole") or "dipole"
        mass = float(spec.get("mass", 1.0))
        charge = float(spec.get("charge", 1.0))

        # Random orientation for the dipole
        phi = np.random.random() * 2 * np.pi
        theta = np.random.random() * np.pi
        orientation = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        )

        return Dipole(
            position=position,
            orientation=orientation,
            length=length,
            type_plus=type_plus_id,
            type_plus_name=type_plus_name,
            type_minus=type_minus_id,
            type_minus_name=type_minus_name,
            type_ghost=type_ghost_id,
            type_ghost_name=type_ghost_name,
            charge=charge,
            mass=mass,
            ghost_count=ghost_count,
        )

    else:
        raise ValueError(f"Unknown molecule type: {molecule_type}")


def _init_uniform_configuration(
    system: System,
    processed_specs: List[Tuple[str, dict, int]],
    type_name_to_id: Dict[str, int],
    padding: np.ndarray,
):
    """Initialize uniform grid configuration with molecules."""
    # Calculate total number of molecules
    total_molecules = sum(count for _, _, count in processed_specs)

    # Build a uniform grid covering the box
    Lx, Ly, Lz = system.box
    # Effective lengths after padding (only for initial placement)
    eff_L = np.maximum([Lx, Ly, Lz] - 2.0 * padding, 1e-9)
    origin = padding

    # Choose grid divisions to make cell sizes as isotropic as possible
    nx = ny = nz = 1
    # Increment along the axis with the largest current cell size until we have enough cells
    while nx * ny * nz < total_molecules:
        cell_sizes = np.array([eff_L[0] / nx, eff_L[1] / ny, eff_L[2] / nz])
        axis = int(np.argmax(cell_sizes))
        if axis == 0:
            nx += 1
        elif axis == 1:
            ny += 1
        else:
            nz += 1

    # Generate grid points centered in cells
    xs = origin[0] + (np.arange(nx) + 0.5) * (eff_L[0] / nx)
    ys = origin[1] + (np.arange(ny) + 0.5) * (eff_L[1] / ny)
    zs = origin[2] + (np.arange(nz) + 0.5) * (eff_L[2] / nz)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij"))
    grid = grid.reshape(3, -1).T  # (nx*ny*nz, 3)
    positions_list = grid[:total_molecules]
    # Shuffle positions to avoid type clustering
    shuffle_idx = np.random.permutation(total_molecules)
    positions_list = positions_list[shuffle_idx, :]

    # Create a list of all molecules to place
    molecules_to_place = []
    for molecule_type, spec, count in processed_specs:
        for _ in range(count):
            molecules_to_place.append((molecule_type, spec))

    # Shuffle to randomize placement order
    np.random.shuffle(molecules_to_place)

    # Place molecules at grid points
    for position, (molecule_type, spec) in zip(positions_list, molecules_to_place):
        molecule = _create_molecule(position, molecule_type, spec, type_name_to_id)
        system.add_molecule(molecule)


def _init_scc_configuration(
    system: System,
    processed_specs: List[Tuple[str, dict, int]],
    type_name_to_id: Dict[str, int],
    padding: np.ndarray,
):
    """Initialize simple cubic lattice configuration with alternating atoms. (only for one atom molecules)"""
    # Check for dipoles
    for molecule_type, _, _ in processed_specs:
        if molecule_type == "Dipole":
            raise ValueError("SCC mode does not support dipole molecules")

    # We need exactly two particle types
    particle_specs = [
        (spec, count)
        for molecule_type, spec, count in processed_specs
        if molecule_type == "Particle"
    ]
    if len(particle_specs) != 2:
        raise ValueError("SCC mode requires exactly two particle types")

    # Count total molecules
    total_N = sum(count for _, count in particle_specs)
    if total_N % 2 != 0:
        raise ValueError("SCC mode requires even number of atoms")

    # Check if we can form a perfect simple cubic lattice
    atoms_per_unit_cell = 2  # SCC has 2 atoms per unit cell
    if total_N % atoms_per_unit_cell != 0:
        raise ValueError(
            f"SCC mode requires total atoms to be divisible by {atoms_per_unit_cell}"
        )

    # Calculate unit cells needed
    unit_cells = total_N // atoms_per_unit_cell
    # Find cube root to determine lattice dimensions
    lattice_dim = int(round(unit_cells ** (1 / 3)))
    if lattice_dim**3 != unit_cells:
        raise ValueError(f"Cannot form perfect cubic SCC lattice with {total_N} atoms")

    Lx, Ly, Lz = system.box
    # Effective lengths after padding
    eff_L = np.maximum([Lx, Ly, Lz] - 2.0 * padding, 1e-9)
    origin = padding

    # SCC lattice parameter (distance between unit cells)
    a = min(eff_L) / lattice_dim

    # Count molecules of each type to add
    remaining_counts = {0: particle_specs[0][1], 1: particle_specs[1][1]}

    atom_count = 0
    for i in range(lattice_dim):
        for j in range(lattice_dim):
            for k in range(lattice_dim):
                # Unit cell origin
                cell_origin = origin + a * np.array([i, j, k])

                # Place 2 atoms in this unit cell (alternating types)
                for basis_idx in range(2):
                    if atom_count >= total_N:
                        break

                    # Calculate position
                    if basis_idx == 0:
                        position = cell_origin  # First atom at origin
                    else:
                        position = cell_origin + 0.5 * a * np.array(
                            [1, 1, 1]
                        )  # Second at center

                    # Determine atom type based on position (alternating pattern)
                    # Use sum of coordinates to determine type
                    coord_sum = i + j + k + basis_idx
                    type_idx = coord_sum % 2

                    # Skip if we've used all molecules of this type
                    if remaining_counts[type_idx] <= 0:
                        continue

                    # Create and add molecule
                    spec, _ = particle_specs[type_idx]
                    molecule = _create_molecule(
                        position, "Particle", spec, type_name_to_id
                    )
                    system.add_molecule(molecule)
                    remaining_counts[type_idx] -= 1
                    atom_count += 1


def _init_random_configuration(
    system: System,
    processed_specs: List[Tuple[str, dict, int]],
    type_name_to_id: Dict[str, int],
    min_distance: float,
    padding: np.ndarray,
):
    """Initialize random configuration with molecules with minimum separation."""
    box = system.box
    eff_L = np.maximum(box - 2.0 * padding, 1e-9)
    origin = padding
    max_attempts = 10000
    min_dist = np.inf

    # For each type specification, create and add molecules
    for molecule_type, spec, count in processed_specs:
        for _ in range(count):
            placed = False
            attempts = 0

            while not placed and attempts < max_attempts:
                # Generate position within effective length
                position = origin + np.random.random(3) * eff_L

                # Create a molecule to test (without adding it)
                try:
                    molecule = _create_molecule(
                        position, molecule_type, spec, type_name_to_id
                    )
                except ValueError as e:
                    print(f"Error creating molecule: {e}")
                    break

                # Check if we can perform energy check
                too_close = False

                # Get all particles from the molecule
                molecule_particles = molecule.get_particles()

                # Check each particle in the molecule against existing atoms
                for particle_pos, type_id, _, _, _ in molecule_particles:
                    # Check distance to other particles
                    if system.N_atoms > 0:
                        # Get distances using the exact same method as in topology._single_pair_energy_forces_virial
                        # to ensure consistency with energy calculations
                        positions = system.positions[: system.N_atoms]
                        dr_vectors = positions - particle_pos[np.newaxis, :]

                        # Apply minimum image convention exactly as in topology
                        dr_vectors = np.where(
                            system.pbc[np.newaxis, :],
                            dr_vectors
                            - system.box[np.newaxis, :]
                            * np.round(dr_vectors * system.inv_box[np.newaxis, :]),
                            dr_vectors,
                        )

                        # Calculate distances
                        distances = np.linalg.norm(dr_vectors, axis=1)

                        if np.any(distances < min_distance):
                            too_close = True
                            break

                        # Track the minimum distance for debugging
                        if len(distances) > 0:
                            min_dist = min(min_dist, np.min(distances))

                if not too_close:
                    # Add the molecule to the system
                    system.add_molecule(molecule)
                    placed = True

                attempts += 1

            if not placed:
                # Fallback: place anyway (may be close)
                print(f"Failed to place molecule: {molecule_type}")
                # position = origin + np.random.random(3) * eff_L
                # try:
                #     molecule = _create_molecule(
                #         position, molecule_type, spec, type_name_to_id
                #     )
                #     system.add_molecule(molecule)
                # except ValueError as e:
                #     print(f"Error creating molecule in fallback: {e}")

    print(f"min_dist: {min_dist}")


def _configure_pair_force(force_name: str, params: dict, units: Units):
    name = force_name.lower()
    if name in ("lj", "lennardjones", "lennard_jones"):
        if params.get("sigma") is not None and params.get("epsilon") is not None:
            return LennardJones(
                sigma=float(params.get("sigma")),
                epsilon=float(params.get("epsilon")),
                cutoff=float(params.get("cutoff", 2.5)),
                units=units,
            )
        else:
            raise ValueError(f"LJ parameters not provided for {force_name}")
    if name in ("ewald", "ewald_sum", "ewald_summation"):
        params = {
            name: params.get(name)
            for name in ["exclude_intermol"]
            if params.get(name) is not None
        }
        return EwaldReal(
            **params,
            units=units,
        )
    if name in ("coulomb", "coulomb_sum"):
        # Extract Coulomb parameters
        nx_images = params.get("nx_images", 0)
        ny_images = params.get("ny_images", 0)
        nz_images = params.get("nz_images", 0)
        cutoff = params.get("cutoff", None)
        exclude_intermol = params.get("exclude_intermol", False)

        from .forces.coulomb import Coulomb
        return Coulomb(
            units=units,
            cutoff=cutoff,
            nx_images=nx_images,
            ny_images=ny_images,
            nz_images=nz_images,
            exclude_intermol=exclude_intermol,
        )
    if name in ("huggins_mayer", "huggins_mayer_potential"):
        if (
            params.get("sigma") is not None
            and params.get("b") is not None
            and params.get("B") is not None
            and params.get("c") is not None
            and params.get("d") is not None
            and params.get("cutoff") is not None
        ):
            return HugginsMayer(
                sigma=float(params.get("sigma")),
                b=float(params.get("b")),
                B=float(params.get("B")),
                c=float(params.get("c")),
                d=float(params.get("d")),
                cutoff=float(params.get("cutoff")),
                units=units,
            )
        else:
            raise ValueError(f"Huggins-Mayer parameters not provided for {force_name}")
    if name in ("hard_sphere", "hard_sphere_potential"):
        params = {
            name: params.get(name)
            for name in ["r1", "r2", "exclude_intermol"]
            if params.get(name) is not None
        }
        return HardSphere(
            **params,
            units=units,
        )
    raise ValueError(f"Unknown force kind: {force_name}")
    return None


def _configure_one_body_force(field_cfg: dict, units: Units):
    kind = str(field_cfg.get("kind", "uniform")).lower()
    if kind in ("wall", "external_wall", "wall_potential"):
        if field_cfg.get("style") is not None:
            return Wall_10_4_3(field_cfg.get("style"), units=units)
        else:
            raise ValueError(f"Wall style not provided for {kind}")
    if kind in ("hard_wall"):
        if field_cfg.get("style") is not None and field_cfg.get("r_wall") is not None:
            return HardWall(
                r_wall=float(field_cfg.get("r_wall")),
                style=field_cfg.get("style"),
                units=units,
            )
        else:
            raise ValueError(f"Hard wall parameters not provided for {kind}")
    if kind in ("electric_field"):
        # Two options: specify voltage between walls or direct field vector
        if (
            field_cfg.get("voltage") is not None
            and field_cfg.get("direction") is not None
        ):
            return ElectricField(
                voltage=float(field_cfg.get("voltage")),
                direction=field_cfg.get("direction"),
                units=units,
            )
        else:
            raise ValueError(
                "Either voltage and direction must be provided for electric_field"
            )
    raise ValueError(f"Unknown one-body field kind: {kind}")


def check_initial_configuration_energy(system: System, topology: Topology) -> None:
    """
    Check the initial configuration for problematic high-energy interactions.
    This helps identify particles that are too close to each other or to walls.
    """
    print("\n====== INITIAL CONFIGURATION ENERGY CHECK ======")

    # Get all atoms
    positions, molecule_ids, types, names, charges, masses = system.get_active_atoms()

    # Check particle-particle overlaps
    print("\nChecking for particle-particle overlaps...")
    for i in range(system.N_atoms):
        for j in range(i + 1, system.N_atoms):
            # Calculate distance
            r_ij = system.get_minimum_image_distance(positions[i], positions[j])
            distance = np.linalg.norm(r_ij)

            # Check against hard sphere radius (assuming 0.1 from config)
            if distance < 0.2:  # 2 * r = 0.2
                print(
                    f"  Warning: Particles {i} ({names[i]}) and {j} ({names[j]}) are too close: {distance:.4f} nm"
                )

    # Check wall overlaps if no PBC in z
    if not system.pbc[2]:
        print("\nChecking for wall overlaps...")
        H = system.box[2]
        wall_distance = 0.1  # r_wall from config

        for i in range(system.N_atoms):
            z = positions[i, 2]
            if z < wall_distance:
                print(
                    f"  Warning: Particle {i} ({names[i]}) too close to bottom wall: {z:.4f} nm"
                )
            if z > (H - wall_distance):
                print(
                    f"  Warning: Particle {i} ({names[i]}) too close to top wall: {(H - z):.4f} nm"
                )

    # Check total energy
    energy = topology.get_energy(system)
    print(f"\nInitial configuration energy: {energy:.4e} kJ/mol")
    if energy > 1e10:
        print(
            "  WARNING: Extremely high energy detected! Check for overlapping particles or wall violations."
        )

    print("\n==============================================")


def create_topology(cfg, units: Units) -> Topology:
    """Create topology with force field from configuration (new schema)."""
    debug = bool(cfg.get("debug", False))
    profiler = Profiler(enabled=debug)
    topology = Topology(units=units, profiler=profiler)

    types_cfg = cfg.get("types", [])
    if not types_cfg:
        raise ValueError("No 'types' section in configuration")

    # Register types from the new format
    for type_entry in types_cfg:
        # Check if it's a dict
        if not isinstance(type_entry, dict) and not isinstance(type_entry, DictConfig):
            continue

        # Convert from OmegaConf container to dict if needed
        if isinstance(type_entry, (DictConfig)):
            type_entry = OmegaConf.to_container(type_entry, resolve=True)

        # Handle nested dictionaries
        if "Particle" in type_entry and type_entry["Particle"] is not None:
            properties = type_entry["Particle"]

            # Register a single type for the particle
            type_name = str(properties.get("type", ""))
            if not type_name:
                continue

            type_id = len(topology.type_name_to_id)
            topology.register_type(type_id, type_name)

            # Set type properties
            type_properties = {
                "mass": float(properties.get("mass", 1.0)),
                "charge": float(properties.get("charge", 0.0)),
            }
            topology.set_type_properties(type_id, type_properties)

        elif "Dipole" in type_entry and type_entry["Dipole"] is not None:
            properties = type_entry["Dipole"]

            # Register types for positive, negative, and ghost particles
            type_plus = str(properties.get("type_plus", ""))
            type_minus = str(properties.get("type_minus", ""))
            type_ghost = str(properties.get("type_ghost", ""))

            if type_plus:
                plus_id = len(topology.type_name_to_id)
                topology.register_type(plus_id, type_plus)
                topology.set_type_properties(
                    plus_id,
                    {
                        "mass": float(properties.get("mass", 1.0)),
                        "charge": abs(float(properties.get("charge", 1.0))),
                    },
                )

            if type_minus:
                minus_id = len(topology.type_name_to_id)
                topology.register_type(minus_id, type_minus)
                topology.set_type_properties(
                    minus_id,
                    {
                        "mass": float(properties.get("mass", 1.0)),
                        "charge": -abs(float(properties.get("charge", 1.0))),
                    },
                )

            if type_ghost:
                ghost_id = len(topology.type_name_to_id)
                topology.register_type(ghost_id, type_ghost)
                topology.set_type_properties(
                    ghost_id,
                    {
                        "mass": 0.0,
                        "charge": 0.0,
                    },
                )

    # Build forces per pair
    pairs_cfg = cfg.get("pairs", [])
    if pairs_cfg:
        # Explicit pairs config
        for pair in pairs_cfg:
            pair_types = pair.get("types")
            if not pair_types:
                raise ValueError("Each pairs entry must have 'types' field")

            # Handle special syntax in yaml
            if len(pair_types) == 2:
                # Parse the types, handling cases like [Na, (Ip, Im)]
                type_pairs = parse_type_pairs(pair_types)
            else:
                raise ValueError(f"Invalid pair types: {pair_types}")

            forces_cfg = pair.get("forces", [])

            for type1_name, type2_name in type_pairs:
                if (
                    type1_name not in topology.type_name_to_id
                    or type2_name not in topology.type_name_to_id
                ):
                    continue

                type1_id = topology.type_name_to_id[type1_name]
                type2_id = topology.type_name_to_id[type2_name]

                # Add all forces for this pair
                for fcfg in forces_cfg:
                    force = _configure_pair_force(fcfg.get("kind", ""), fcfg, units)
                    if force is None:
                        raise ValueError(
                            f"Unknown force kind in pair {(type1_name, type2_name)}: {fcfg}"
                        )

                    topology.add_interaction(type1_id, type2_id, force)

    # One-body fields
    for entry in cfg.get("one_body", []) or []:
        type_spec = entry.get("type")
        if type_spec is None:
            raise ValueError(f"Missing 'type' in one-body entry: {entry}")

        # Parse type(s) - handle groups like "(Na, Cl, Ip, Im, G)"
        type_names = []
        if isinstance(type_spec, str):
            # Check if it's a group expression like "(Na, Cl, Ip, Im, G)"
            type_spec = type_spec.strip()
            if type_spec.startswith("(") and type_spec.endswith(")"):
                # Extract content within parentheses and split by comma
                content = type_spec[1:-1]
                type_names = [t.strip() for t in content.split(",")]
            else:
                type_names = [type_spec]
        elif isinstance(type_spec, (list, tuple, ListConfig)):
            type_names = list(type_spec)
        else:
            type_names = [str(type_spec)]

        # Validate all type names exist in the topology
        valid_types = []
        for type_name in type_names:
            if type_name in topology.type_name_to_id:
                valid_types.append((type_name, topology.type_name_to_id[type_name]))
            else:
                print(
                    f"Warning: One-body type '{type_name}' not found in topology, skipping"
                )

        if not valid_types:
            print(f"Warning: No valid types found in one-body entry: {entry}")
            continue

        forces_list = entry.get("forces", [])

        # Accept OmegaConf ListConfig as a list-like container
        if not isinstance(forces_list, (list, tuple)) and not (
            "ListConfig" in globals() and isinstance(forces_list, ListConfig)
        ):
            forces_list = [forces_list]

        # For each valid type, add all forces
        for type_name, type_id in valid_types:
            for fcfg in forces_list:
                if isinstance(fcfg, str):
                    fcfg = {"kind": fcfg}
                if not fcfg.get("enabled", True):
                    continue

                field = _configure_one_body_force(fcfg, units)
                topology.add_one_body_force(type_id, field)

    # Print summary of configured interactions
    print("\n====== TOPOLOGY CONFIGURATION SUMMARY ======")

    # Print registered types
    print("\nRegistered Types:")
    for type_id, type_name in topology.type_id_to_name.items():
        props = topology.type_properties.get(type_id, {})
        mass = props.get("mass", "N/A")
        charge = props.get("charge", "N/A")
        print(f"  {type_id}: {type_name} (mass={mass}, charge={charge})")

    # Print pair interactions
    print("\nPair Interactions:")
    for (type1_id, type2_id), forces in topology.interactions.items():
        type1_name = topology.type_id_to_name.get(type1_id, f"Type{type1_id}")
        type2_name = topology.type_id_to_name.get(type2_id, f"Type{type2_id}")
        force_names = [f.__class__.__name__ for f in forces]
        print(
            f"  {type1_name}({type1_id}) - {type2_name}({type2_id}): {', '.join(force_names)}"
        )

    # Print one-body forces
    print("\nOne-Body Forces:")
    for type_id, forces in topology.one_body_forces.items():
        type_name = topology.type_id_to_name.get(type_id, f"Type{type_id}")
        force_names = [f.__class__.__name__ for f in forces]
        print(f"  {type_name}({type_id}): {', '.join(force_names)}")

    print("\n==========================================")

    return topology


def create_sampler(
    system: System, topology: Topology, cfg, logger: Logger, units: Units
) -> Sampler:
    """Create Monte Carlo sampler from configuration."""
    actions = list(cfg.sampler.get("actions", []))
    if not actions:
        actions = [{"action": "translate", "probability": 1.0}]

    max_displacement = cfg.sampler.get("max_displacement", 0.1)
    max_rotation = cfg.sampler.get("max_rotation", 10)
    debug = bool(cfg.get("debug", False))
    force_recompute = bool(cfg.sampler.get("force_recompute", False))

    return Sampler(
        system=system,
        topology=topology,
        actions=actions,
        max_displacement=max_displacement,
        max_rotation=max_rotation,
        logger=logger,
        units=units,
        debug=debug,
        profiler=topology.profiler,
        force_recompute=force_recompute,
    )


def create_logger(cfg) -> Logger:
    """Create logger from configuration."""
    exp_name = cfg.experiment.get("name", "simulation")
    results_dir = cfg.experiment.get("results_dir", "results")
    log_interval = cfg.logger.get("log_interval", 100)
    xyz_interval = cfg.logger.get("xyz_interval", 100)

    return Logger(
        experiment_name=exp_name,
        results_dir=results_dir,
        log_interval=log_interval,
        xyz_interval=xyz_interval,
        overwrite=True,
        units=create_units(
            cfg
        ),  # units passed again for labeling consistency if called standalone
    )
