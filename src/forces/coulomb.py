import numpy as np
from typing import Dict, Tuple
from ..units import Units
from .basic_force import Force


class Coulomb(Force):
    def __init__(
        self,
        units: Units = None,
        exclude_intermol: bool = False,
        cutoff: float = None,
        nx_images: int = 1,
        ny_images: int = 1,
        nz_images: int = 1,
    ):
        super().__init__(units)
        self.cutoff = cutoff
        self.nx_images = nx_images
        self.ny_images = ny_images
        self.nz_images = nz_images
        self.exclude_intermol = exclude_intermol
        self.exclude_self_interaction = True

        assert self.nx_images >= 0 and self.ny_images >= 0 and self.nz_images >= 0, (
            "nx_images, ny_images, nz_images must be positive"
        )

        # assert self.cutoff is None or (
        #     self.nx_images == 1 and self.ny_images == 1 and self.nz_images == 1
        # ), "Either cutoff or nx_images, ny_images, nz_images must be provided, not both"

    def __call__(
        self,
        pos: np.ndarray,
        r_ij: np.ndarray,
        charge1: float,
        charge2: float,
        system=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Handle interaction in the primary cell
        N = r_ij.shape[0]
        energies = np.zeros(N)
        forces = np.zeros((N, 3))

        # Calculate energy and forces for direct interactions (primary cell)
        # r = np.linalg.norm(r_ij, axis=1)

        # with np.errstate(divide="ignore", invalid="ignore"):
        #     if self.cutoff is not None:
        #         mask = (r > 1e-12) & (r < self.cutoff)
        #     else:
        #         mask = r > 1e-12

        #     if np.any(mask):
        #         inv_r = 1.0 / r[mask]
        #         inv_r3 = inv_r**3

        #         # Energy calculation - this is for the primary cell
        #         energies[mask] = charge1 * charge2 * inv_r

        #         # Force calculation
        #         force_mag = charge1 * charge2 * inv_r3
        #         force_vec = force_mag[:, np.newaxis] * r_ij[mask]
        #         forces[mask] = force_vec

        # For periodic systems with images
        # if (
        #     self.nx_images > 0 or self.ny_images > 0 or self.nz_images > 0
        # ) and system is not None:
        # Calculate interactions with periodic images
        image_energy = 0.0
        image_force = np.zeros(3)

        # 1. Interaction with other particles through images
        # This handles i != j interactions across periodic boundaries
        for i in range(-self.nx_images, self.nx_images + 1):
            for j in range(-self.ny_images, self.ny_images + 1):
                for k in range(-self.nz_images, self.nz_images + 1):
                    image_shift = np.array([i, j, k]) * system.box

                    # Calculate shifted distances with minimum image convention
                    r_shifted = r_ij + image_shift[np.newaxis, :]
                    r_norm = np.linalg.norm(r_shifted, axis=1)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        mask = r_norm > 1e-9
                        if np.any(mask):
                            inv_r = 1.0 / r_norm[mask]
                            inv_r3 = inv_r**3

                            # Energy contribution
                            energies[mask] = charge1 * charge2 * inv_r

                            # Force contribution
                            force_mag = charge1 * charge2 * inv_r3
                            force_vec = force_mag[:, np.newaxis] * r_shifted[mask]
                            forces[mask] += force_vec

        # 2. Self-interaction with images (i interacting with its own image)
        # Important: Apply 0.5 factor to avoid double counting as in the example code
        for i in range(-self.nx_images, self.nx_images + 1):
            for j in range(-self.ny_images, self.ny_images + 1):
                for k in range(-self.nz_images, self.nz_images + 1):
                    # Skip the primary cell (no self-interaction)
                    if i == 0 and j == 0 and k == 0:
                        continue

                    image_shift = np.array([i, j, k]) * system.box
                    r_norm = np.linalg.norm(image_shift)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        if r_norm > 1e-9:
                            inv_r = 1.0 / r_norm
                            inv_r3 = inv_r**3

                            # Use 0.5 factor for self-image interaction as in the example code
                            energy_contrib = 0.5 * charge1 * charge1 * inv_r
                            image_energy += energy_contrib

                            # Force contribution
                            force_mag = charge1 * charge1 * inv_r3
                            force_vec = force_mag * image_shift
                            image_force += force_vec

        # # Add image contributions to the energy of the primary particle
        energies[0] += image_energy
        forces[0] += image_force

        # Apply reduced Bjerrum length scaling for LJ units
        if hasattr(system, "ewald_handler") and system.ewald_handler is not None:
            energies *= system.ewald_handler.lB_star
            forces *= system.ewald_handler.lB_star
        elif hasattr(system, "lB_star") and system.lB_star is not None:
            energies *= system.lB_star
            forces *= system.lB_star
        else:
            raise ValueError("No Ewald handler or Bjerrum length scaling found")

        return np.sum(energies), forces
