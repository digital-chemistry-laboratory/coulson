"""Hückel calculator."""
from __future__ import annotations

from collections.abc import Iterable
import itertools
from typing import Sequence

import numpy as np

from coulson.graph import rings_from_connectivity
from coulson.typing import (
    Array1DFloat,
    Array1DInt,
    Array2DFloat,
    Array2DInt,
    ArrayLike2D,
)


class HuckelCalculator:
    """Calculator for Hückel molecular orbital theory.

    Args:
        huckel_matrix: Hückel matrix
        electrons: Number of electrons contributed by each atom
        charge: Molecular charge
        multiplicity: Multiplicity
        n_dec_degen: How many decimals to consider for degerate orbitals

    Attributes:
        bo_matrix: Bond order matrix
        charges: Atomic charges
        coefficients: Coefficients
        degeneracies: Degeneracies
        energies: Energies
        multiplicity: Multiplicity
        electrons: Number of electrons contributed by each atom
        n_electrons: Number of electrons
        n_occupied: Number of occupied orbitals
        n_orbitals: Number of orbitals
        n_unpaired: Number of unpaired electrons
        occupations: Orbital occupations
        spin_densities: Spin densities
        spin_occupations: Orbitals occupations of unpaired electrons
    """

    bo_matrix: Array2DFloat
    charges: Array1DFloat
    coefficients: Array2DFloat
    degeneracies: Array1DInt
    energies: Array1DFloat
    multiplicity: int
    electrons: Iterable[int]
    n_electrons: int
    n_occupied: int
    n_orbitals: int
    n_unpaired: int
    occupations: Array1DFloat
    spin_densities: Array1DFloat
    spin_occupations: Array1DFloat
    _D: Array2DFloat
    _D_spin: Array2DFloat

    def __init__(
        self,
        huckel_matrix: ArrayLike2D,
        electrons: Iterable[int],
        charge: int = 0,
        multiplicity: int | None = None,
        n_dec_degen: int = 3,
    ) -> None:
        huckel_matrix: Array2DFloat = np.array(huckel_matrix)
        self.huckel_matrix = huckel_matrix

        # Set up number of electrons
        self.electrons = electrons
        n_electrons = sum(electrons) - charge

        # Create connectivity matrix
        connectivity_matrix: Array2DInt = np.array(huckel_matrix)
        np.fill_diagonal(connectivity_matrix, 0)
        connectivity_matrix[connectivity_matrix.nonzero()] = 1
        self.connectivity_matrix = connectivity_matrix

        # Get number of orbitals
        self.n_orbitals = huckel_matrix.shape[0]

        # Set multiplicity
        if multiplicity is None:
            multiplicity = (n_electrons % 2) + 1

        # Multiplicity check
        if (n_electrons % 2) == (multiplicity % 2):
            raise Exception(
                f"Combination of number of electrons {n_electrons} and "
                f"multiplicity {multiplicity} not possible!"
            )
        self.n_excess_spin = multiplicity - 1
        self.multiplicity = multiplicity
        self.n_electrons = n_electrons
        self.n_dec_degen = n_dec_degen

        # Make calculations
        self._solve()
        self._set_occupations()
        self._calculate_bond_orders()

    def bond_order(self, i: int, j: int) -> float:
        """Returns bond order between two atoms.

        Args:
            i: Index of atom 1 (1-indexed)
            j: Index of atom 2 (1-indexed)

        Returns:
            bo: Bond order
        """
        bo: float = self.bo_matrix[i - 1, j - 1]

        return bo

    def calculate_i_ring(
        self,
        indices: Iterable[int],
        normalize: bool = True,
        permute: bool = False,
        norm_method: str = "solà",
    ) -> float:
        """Returns the I_ring or MCI indices.

        Args:
            indices: Indices of ring sequential order as bonded (1-indexed)
            normalize: Whether to normalize with respect to ring size
            permute: Whether to do all permutations of ring indices to get MCI
            norm_method: Method to use for normalization: 'mandado' or 'solà'

        Returns:
            index: I_ring or MCI index

        Raises:
            ValueError: When norm_method is not correct.
        """
        # Calculate I_ring
        indices = tuple(indices)
        n_atoms = len(indices)

        # Set up list of indices to calculate
        if permute is True:
            i_ring_indices = list(itertools.permutations(indices, n_atoms))
        else:
            i_ring_indices = [indices]

        # Calculate I_ring
        i_rings: list[float] = []
        for indices in i_ring_indices:
            i_ring = 1
            for (i, j) in zip(indices, indices[1:] + indices[:1]):
                i_ring *= self.bo_matrix[i - 1, j - 1]
            i_rings.append(i_ring)
        index = sum(i_rings)

        # Normalize
        if normalize:
            if norm_method not in ["mandado", "solà"]:
                raise ValueError("Choose 'mandado' or 'solà' as norm_metod.")
            if norm_method == "mandado":
                index /= n_atoms
            if np.sign(index) < 0:
                index = -abs(index) ** (1 / n_atoms)
            else:
                index = index ** (1 / n_atoms)

        return index

    def get_rings(self) -> Sequence[Sequence[int]]:
        """Get rings from connectivity.

        Returns:
            rings: Rings
        """
        rings = rings_from_connectivity(self.connectivity_matrix)
        rings = [[i + 1 for i in ring] for ring in rings]
        return rings

    def _solve(self) -> None:
        # Calcuate eigenvalues (energies) and eigenvectors (coefficients)
        eigenvalues, eigenvectors = np.linalg.eigh(self.huckel_matrix)
        self.coefficients = eigenvectors.T[::-1, :]
        self.energies = eigenvalues[::-1]

    def _set_occupations(self) -> None:
        # Determine number of singly and doubly occupied orbitals.
        n_doubly = int((self.n_electrons - self.n_excess_spin) / 2)
        n_singly = self.n_excess_spin

        # Make list of electrons to distribute in orbitals
        all_electrons = [2] * n_doubly + [1] * n_singly

        # Set up occupation numbers
        occupations = np.zeros(self.n_orbitals)
        spin_occupations = np.zeros(self.n_orbitals)

        # Loop over unique rounded orbital energies and degeneracies and fill with
        # electrons
        energies_rounded = self.energies.round(self.n_dec_degen)
        unique_energies_rounded: Array1DFloat
        degeneracies: Array1DInt
        unique_energies_rounded, indices, degeneracies = np.unique(
            energies_rounded, return_index=True, return_counts=True
        )
        unique_energies = np.flip(self.energies[indices])
        unique_energies_rounded = np.flip(unique_energies_rounded)
        degeneracies = np.flip(degeneracies)
        for energy, degeneracy in zip(unique_energies_rounded, degeneracies):
            if len(all_electrons) == 0:
                break

            # Determine number of electrons with and without excess spin.
            electrons = 0
            spin_electrons = 0
            for _ in range(degeneracy):
                if len(all_electrons) > 0:
                    pop_electrons = all_electrons.pop(0)
                    electrons += pop_electrons
                    if pop_electrons == 1:
                        spin_electrons += 1

            # Divide electrons evenly among orbitals
            occupations[np.where(energies_rounded == energy)] += electrons / degeneracy
            spin_occupations[np.where(energies_rounded == energy)] += (
                spin_electrons / degeneracy
            )

        n_occupied: int = np.count_nonzero(occupations)
        n_unpaired = int(
            np.sum(occupations[:n_occupied][occupations[:n_occupied] != 2])
        )

        # Set shell occupations
        shell_occupations = np.zeros_like(unique_energies)
        i = 0
        for j, degeneracy in enumerate(degeneracies):
            shell_occupations[j] = sum(occupations[i : i + degeneracy]) / degeneracy
            i += degeneracy

        self.occupations = occupations
        self.shell_occupations_avg = shell_occupations
        self.spin_occupations = spin_occupations
        self.n_occupied = n_occupied
        self.n_unpaired = n_unpaired
        self.unique_energies = unique_energies
        self.degeneracies = degeneracies

    def _calculate_bond_orders(self) -> None:
        # Set up density matrices
        D = np.zeros((self.n_orbitals, self.n_orbitals))
        D_spin = np.zeros((self.n_orbitals, self.n_orbitals))

        # Add contributions from each orbital to matrix
        for i, (occ, spin_occ) in enumerate(
            zip(
                self.occupations[: self.n_occupied],
                self.spin_occupations[: self.n_occupied],
            )
        ):
            outer_product = np.outer(self.coefficients[i], self.coefficients[i])
            D += outer_product * occ
            D_spin += outer_product * spin_occ

        # Set up attributes
        self._D = D
        self._D_spin = D_spin
        self.bo_matrix = D
        self.charges = self.electrons - np.diag(D)
        self.spin_densities = np.diag(D_spin)
