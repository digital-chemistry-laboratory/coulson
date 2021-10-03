"""Hückel calculator."""
import itertools
from typing import Sequence

import numpy as np

from coulson.utils import rings_from_connectivity


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

    def __init__(
        self,
        huckel_matrix: Sequence[Sequence],
        electrons: int,
        charge: int = 0,
        multiplicity: int = None,
        n_dec_degen: int = 3,
    ) -> None:
        huckel_matrix = np.array(huckel_matrix)
        self.huckel_matrix = huckel_matrix

        # Set up number of electrons
        self.electrons = electrons
        n_electrons = sum(electrons) - charge

        # Create connectivity matrix
        connectivity_matrix = np.array(huckel_matrix)
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
        self.n_occupied = None
        self.energies = None
        self.occupations = None
        self.spin_occupations = None
        self.charges = None
        self.spin_densities = None
        self.bo_matrix = None
        self._D = None
        self._D_spin = None
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
        bo = self.bo_matrix[i - 1, j - 1]

        return bo

    def calculate_i_ring(
        self,
        indices: Sequence,
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
        indices = list(indices)
        n_atoms = len(indices)

        # Set up list of indices to calculate
        if permute is True:
            i_ring_indices = itertools.permutations(indices, n_atoms)
        else:
            i_ring_indices = [indices]

        # Calculate I_ring
        i_rings = []
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
        unique_energies, degeneracies = np.unique(energies_rounded, return_counts=True)
        for energy, degeneracy in zip(np.flip(unique_energies), np.flip(degeneracies)):
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

        n_occupied = np.count_nonzero(occupations)
        n_unpaired = int(
            np.sum(occupations[:n_occupied][occupations[:n_occupied] != 2])
        )

        self.occupations = occupations
        self.spin_occupations = spin_occupations
        self.n_occupied = n_occupied
        self.n_unpaired = n_unpaired

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
