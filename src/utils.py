"""
Utility functions for building and running Monte Carlo simulations.
Consolidates helpers used by mc_main.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
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
from forces import *
from .units import Units
from .logger import Logger
from .profiler import Profiler


def load_config(config_file: str, overrides: Optional[List[str]] = None):
    """Load configuration from YAML file using OmegaConf with interpolation/resolvers and CLI overrides.

    - Supports ${var} style interpolation (OmegaConf built-in)
    - Adds an eval resolver: ${eval:"${size}*10"} or ${eval:${size}*10}
    - Applies CLI overrides via dotlist, e.g., ["system.size=20"]
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Configuration file {config_file} not found")

    # Register a simple eval resolver (dangerous if untrusted inputs)
    def _eval_resolver(expr: str):
        try:
            return eval(expr, {}, {})
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


def load_lj_parameters(lj_file: str) -> dict:
    """Load Lennard-Jones parameters from YAML file."""
    if not Path(lj_file).exists():
        raise FileNotFoundError(f"LJ parameters file {lj_file} not found")

    lj_data = OmegaConf.load(lj_file)
    return lj_data.get("lj_parameters", {})


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

        if not (alpha is None or real_cut is None or n_c is None or dielectric is None):
            ewald_handler = EwaldHandler(
                alpha=alpha,
                real_cut=real_cut,
                n_c=n_c,
                dielectric=dielectric,
                z_scale_factor=z_scale_factor,
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


def setup_initial_configuration(
    system: System, atom_types_spec: dict, topology: Topology, cfg=None
):
    """Setup initial atom configuration.

    Modes (cfg.system.init.mode):
      - "random": random positions with minimum separation (cfg.system.init.min_distance)
      - "uniform": positions on a uniform 3D grid covering the box

    atom_types_spec format:
      {
        type_name: { count: int | null, fraction: float | null, name: str, mass, charge }
      }
    If counts are not provided, use fractions to distribute system._target_n_atoms.
    """
    target_n_atoms = getattr(system, "_target_n_atoms", 0)
    if target_n_atoms <= 0:
        return

    # Determine counts per type
    type_names = list(atom_types_spec.keys())
    counts = {}
    if any("count" in atom_types_spec[t] for t in type_names):
        for t in type_names:
            counts[t] = int(atom_types_spec[t].get("count", 0))
    else:
        fracs = np.array(
            [float(atom_types_spec[t].get("fraction", 0.0)) for t in type_names]
        )
        if fracs.sum() <= 0:
            counts[type_names[0]] = target_n_atoms
            for t in type_names[1:]:
                counts[t] = 0
        else:
            fracs = fracs / fracs.sum()
            alloc = np.floor(fracs * target_n_atoms).astype(int)
            remainder = target_n_atoms - int(alloc.sum())
            fractional = (fracs * target_n_atoms) - alloc
            order = np.argsort(-fractional)
            for k in range(remainder):
                alloc[order[k]] += 1
            for i, t in enumerate(type_names):
                counts[t] = int(alloc[i])

    init_mode = None
    min_distance = 0.5
    padding = np.zeros(3, dtype=float)
    if cfg is not None and "system" in cfg and "init" in cfg.system:
        init_mode = cfg.system.init.get("mode")
        min_distance = float(cfg.system.init.get("min_distance", min_distance))
        pad_val = cfg.system.init.get("padding", [0.0, 0.0, 0.0])
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
        # Build a uniform grid covering the box and place atoms of all types
        total_N = sum(counts.values())
        Lx, Ly, Lz = system.box
        # Effective lengths after padding (only for initial placement)
        eff_L = np.maximum([Lx, Ly, Lz] - 2.0 * padding, 1e-9)
        origin = padding
        # Choose grid divisions to make cell sizes as isotropic as possible
        nx = ny = nz = 1
        # Increment along the axis with the largest current cell size until we have enough cells
        while nx * ny * nz < total_N:
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
        positions_list = grid[:total_N]

        # Prepare a type assignment list according to counts
        type_id_seq: List[int] = []
        name_seq: List[str] = []
        charge_seq: List[float] = []
        mass_seq: List[float] = []
        for t_name in type_names:
            n_t = counts[t_name]
            if n_t <= 0:
                continue
            type_id = topology.type_name_to_id.get(t_name, 0)
            spec = atom_types_spec[t_name]
            mass = float(spec.get("mass", 1.0))
            charge = float(spec.get("charge", 0.0))
            name = str(spec.get("name", t_name))
            type_id_seq.extend([type_id] * n_t)
            name_seq.extend([name] * n_t)
            charge_seq.extend([charge] * n_t)
            mass_seq.extend([mass] * n_t)

        # If order matters, currently we place in type order. Could shuffle lightly to avoid type clusters.
        for i in range(total_N):
            system.add_atom(
                position=positions_list[i],
                atom_type=type_id_seq[i],
                name=name_seq[i],
                charge=charge_seq[i],
                mass=mass_seq[i],
            )
        return

    # Default: random placement with min distance
    box = system.box
    eff_L = np.maximum(box - 2.0 * padding, 1e-9)
    origin = padding
    max_attempts = 5000
    for t_name in type_names:
        n_t = counts[t_name]
        if n_t <= 0:
            continue
        type_id = topology.type_name_to_id.get(t_name, 0)
        spec = atom_types_spec[t_name]
        mass = float(spec.get("mass", 1.0))
        charge = float(spec.get("charge", 0.0))
        name = str(spec.get("name", t_name))

        for _ in range(n_t):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                # Generate position within effective length
                position = origin + np.random.random(3) * eff_L

                if system.N > 0:
                    distances = system.get_all_distances(
                        position, system.positions[: system.N]
                    )
                    too_close = np.any(distances < min_distance)
                else:
                    too_close = False
                if not too_close:
                    system.add_atom(
                        position=position,
                        atom_type=type_id,
                        name=name,
                        charge=charge,
                        mass=mass,
                    )
                    placed = True
                attempts += 1
            if not placed:
                # fallback: place anyway (may be close)
                position = origin + np.random.random(3) * eff_L
                system.add_atom(
                    position=position,
                    atom_type=type_id,
                    name=name,
                    charge=charge,
                    mass=mass,
                )


def _configure_pair_force(force_name: str, params: dict, units: Units):
    name = force_name.lower()
    if name in ("lj", "lennardjones", "lennard_jones"):
        if params.get("sigma") is not None and params.get("epsilon") is not None:
            return LennardJones(
                sigma=float(params.get("sigma")),
                epsilon=float(params.get("epsilon")),
                units=units,
            )
        else:
            raise ValueError(f"LJ parameters not provided for {force_name}")
    # if name in ("coulomb", "electrostatic"):
    #     return Coulomb(units=units)
    if name in ("ewald", "ewald_sum", "ewald_summation"):
        # The EwaldReal class doesn't need parameters here - it gets them from the system
        return EwaldReal(units=units)
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
    return None


def _configure_one_body_force(field_cfg: dict, units: Units):
    kind = str(field_cfg.get("kind", "uniform")).lower()
    if kind == "uniform":
        g = np.array(field_cfg.get("gradient", [0.0, 0.0, 0.0]), dtype=float)
        return ExternalUniformField(g, units=units)
    if kind in ("wall", "external_wall", "wall_potential"):
        style = field_cfg.get("style", "bottom")
        return ExternalWallPotential(style, units=units)
    raise ValueError(f"Unknown one-body field kind: {kind}")


def create_topology(cfg, units: Units) -> Topology:
    """Create topology with force field from configuration (new schema)."""
    debug = bool(cfg.get("debug", False))
    profiler = Profiler(enabled=debug)
    topology = Topology(units=units, profiler=profiler)

    types_cfg = cfg.get("types", {})
    if not types_cfg:
        raise ValueError("No 'types' section in configuration")

    # Load per-type files and register types
    type_order = list(types_cfg.keys())
    per_type_params: Dict[int, dict] = {}
    for type_id, tname in enumerate(type_order):
        topology.register_type(type_id, tname)
        tcfg = types_cfg[tname] or {}
        # Optional per-type file that may contain LJ and other properties
        mass = tcfg.get("mass")
        charge = tcfg.get("charge")
        per_type = {"mass": mass, "charge": charge}
        per_type_params[type_id] = per_type
        topology.set_type_properties(type_id, per_type)

    # Build forces per pair
    pairs_cfg = cfg.get("pairs", [])
    if pairs_cfg:
        # Explicit pairs config; require two-type entries only for pairwise interactions
        type_name_to_id = topology.type_name_to_id
        for pair in pairs_cfg:
            tnames = pair.get("types")
            if not tnames:
                raise ValueError("Each pairs entry must have exactly two 'types'")
            # Accept list-like from OmegaConf
            tnames = list(tnames)
            if len(tnames) != 2:
                raise ValueError("Each pairs entry must have exactly two 'types'")
            forces_cfg = pair.get("forces", [])

            t1 = type_name_to_id[tnames[0]]
            t2 = type_name_to_id[tnames[1]]

            force_objs = []
            for fcfg in forces_cfg:
                f = _configure_pair_force(fcfg.get("kind", ""), fcfg, units)
                if f is None:
                    raise ValueError(f"Unknown force kind in pair {tnames}: {fcfg}")
                # Set base per-type params for mixing
                # if isinstance(f, LennardJones):
                #     p = {}
                #     if (
                #         fcfg.get("sigma") is not None
                #         and fcfg.get("epsilon") is not None
                #     ):
                #         p[(min(t1, t2), max(t1, t2))] = {
                #             "sigma": float(fcfg.get("sigma")),
                #             "epsilon": float(fcfg.get("epsilon")),
                #         }
                #     f.set_parameters(p)
                # elif isinstance(f, Coulomb):
                #     # Allow per-type dielectric and per-pair override
                #     p = {}
                #     for tid, props in per_type_params.items():
                #         if "dielectric" in types_cfg[topology.type_id_to_name[tid]]:
                #             p[tid] = {
                #                 "dielectric": float(
                #                     types_cfg[topology.type_id_to_name[tid]].get(
                #                         "dielectric"
                #                     )
                #                 )
                #             }
                #     o = overrides.get("Coulomb") or overrides.get("coulomb")
                #     if o is not None:
                #         p[(min(t1, t2), max(t1, t2))] = {
                #             "dielectric": float(
                #                 o.get("dielectric", fcfg.get("dielectric", 1.0))
                #             )
                #         }
                #     if p:
                #         f.set_parameters(p)
                # elif isinstance(f, HugginsMayer):
                #     p = {}
                #     if (
                #         fcfg.get("sigma") is not None
                #         and fcfg.get("b") is not None
                #         and fcfg.get("B") is not None
                #         and fcfg.get("c") is not None
                #         and fcfg.get("d") is not None
                #         and fcfg.get("cutoff") is not None
                #     ):
                #         p[(min(t1, t2), max(t1, t2))] = {
                #             "sigma": float(fcfg.get("sigma")),
                #             "b": float(fcfg.get("b")),
                #             "B": float(fcfg.get("B")),
                #             "c": float(fcfg.get("c")),
                #             "d": float(fcfg.get("d")),
                #             "cutoff": float(fcfg.get("cutoff")),
                #         }
                #     f.set_parameters(p)
                force_objs.append(f)

            # If multiple forces on same pair, we can just add each; Topology sums them.
            for f in force_objs:
                topology.add_interaction(t1, t2, f)

    # One-body fields (new dedicated section)
    for entry in cfg.get("one_body", []) or []:
        tname = entry.get("type")
        if tname is None:
            continue
        t_id = topology.type_name_to_id[tname]
        forces_list = entry.get("forces", [])
        # Accept OmegaConf ListConfig as a list-like container
        if not isinstance(forces_list, (list, tuple)) and not (
            "ListConfig" in globals() and isinstance(forces_list, ListConfig)
        ):
            forces_list = [forces_list]
        for fcfg in forces_list:
            if isinstance(fcfg, str):
                fcfg = {"kind": fcfg}
            if not fcfg.get("enabled", True):
                continue
            field = _configure_one_body_force(fcfg, units)
            topology.add_one_body_force(t_id, field)

    return topology


def create_sampler(
    system: System, topology: Topology, cfg, logger: Logger, units: Units
) -> Sampler:
    """Create Monte Carlo sampler from configuration."""
    actions = list(cfg.sampler.get("actions", []))
    if not actions:
        actions = [{"action": "translate", "probability": 1.0}]

    max_displacement = cfg.sampler.get("max_displacement", 0.1)
    debug = bool(cfg.get("debug", False))
    force_recompute = bool(cfg.sampler.get("force_recompute", False))

    return Sampler(
        system=system,
        topology=topology,
        actions=actions,
        max_displacement=max_displacement,
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
