"""PPP calculator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
from typing import Any

import numpy as np
import scipy.constants
import scipy.misc
import scipy.spatial

from coulson.ci import calculate_matrix_element, generate_excitations
from coulson.data import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE
from coulson.huckel import HuckelCalculator, InputData, prepare_huckel_matrix
from coulson.parameters import (
    BEVERIDGE_HINZE,
    CHARGES,
    EA_BH,
    EA_CRC,
    ELECTRON_CONFIGURATIONS,
    IP_BH,
    IP_CRC,
    N,
    N_STAR,
    VE_COEFFICIENTS,
    Z_EFF,
)
from coulson.slater import (
    exp_slater_from_z_eff,
    get_exp_bv,
    slater_overlap,
    slater_overlap_grad,
    z_eff_slater,
)
from coulson.typing import Array1DFloat, Array2DFloat, Array2DInt, Array4DFloat


def get_gamma_xx(atom_type: str, data: str = "beveridge-hinze") -> float:
    """Returns the one-center electron repulsion integral.

    Args:
        atom_type: Atom type
        data: Data set for valence state ionization potential: 'beveridge-hinze'
            (default) or 'crc'

    Returns:
        gamma_xx: Electron repulsion integral (a.u.)
    """
    if data == "beveridge-hinze":
        IP, EA = IP_BH, EA_BH
    elif data == "crc":
        IP, EA = IP_CRC, EA_CRC

    gamma_xx = IP[atom_type] - EA[atom_type]
    gamma_xx *= EV_TO_HARTREE

    return gamma_xx


def get_gamma_xy(
    r: float,
    gamma_11: float,
    gamma_22: float,
    method: str = "beveridge-hinze",
) -> float:
    """Returns the two-center electron repulsion integral.

    Args:
        gamma_11: One-center repulsion integral for atom 1
        gamma_22: One-center repulsion integral for atom 2
        r: Distance between atoms (Å)
        method: Method of calculation: 'beveridge-hinze' (default), 'mataga-nishimoto'
            or 'ohno'

    Returns:
        gamma_xy: Electron repulsion integral (a.u.)

    Raises:
        ValueError: If method is not supported.
    """
    a = 2 / (gamma_11 + gamma_22)

    if method == "mataga-nishimoto":
        gamma_xy = 1 / (r + a)
    elif method == "ohno":
        gamma_xy = 1 / np.sqrt(r**2 + a**2)
    elif method == "beveridge-hinze":
        gamma_xy = 1 / (r + a * np.exp(-(r**2) / (2 * a**2)))
    else:
        raise ValueError(
            "Choose method from beveridge-hinze', 'mataga-nishimoto' or 'ohno'"
        )

    return gamma_xy


def get_beta_bh(
    r: float,
    gamma_xy: float,
    overlap: float,
    charge_1: int,
    charge_2: int,
    C: float = 0.545,
) -> float:
    """Returns two-center core resonance integral according to Beveridge-Hinze formula.

    Args:
        r: Distance between centers (Å)
        gamma_xy: Two-center electron repulsion integral (a.u.)
        overlap: Overlap between Slater orbitals
        charge_1: Core charge of atom 1.
        charge_2: Core charge of atom 2.
        C: Empirical tuning parameter.

    Returns:
        beta: Core resonance integral (a.u.)
    """
    beta = (charge_1 + charge_2) * overlap * (gamma_xy - 2 * C / r) / 2

    return beta


def get_beta_l(
    r: float,
    n_1: int,
    n_2: int,
    exp_1: float,
    exp_2: float,
    overlap_method: str = "formula",
) -> float:
    """Returns two-center core resonance integral according to Linderberg formula.

    Reference: 10.1016/0009-2614(67)80061-7.

    Args:
        r: Distance (a.u.)
        n_1: Principal quantum number of atom 1
        n_2: Principal quantum number of atom 2
        exp_1: Slater exponent of atom 1
        exp_2: Slater exponent of atom 2
        overlap_method: Method for overlap integral gradient: 'formula' or
            'interpolation'

    Returns:
        beta: Core resonance integral (a.u.)
    """
    overlap_grad = slater_overlap_grad(r, n_1, n_2, exp_1, exp_2, method=overlap_method)
    beta = overlap_grad / r

    return beta


def get_beta_jug(
    r: float,
    alpha_1: float,
    alpha_2: float,
    z_eff_1: float,
    z_eff_2: float,
    n_1: int,
    n_2: int,
    exp_1: float,
    exp_2: float,
    overlap_method: str = "formula",
) -> float:
    """Return two-center core resonance integral according to Jug formula.

    References:
        10.1007/BF00526431
        10.1007/BF00529308

    Args:
        r: Distance (a.u.)
        alpha_1: Core resonance integral of atom 1 (a.u.)
        alpha_2: Core resonance integral of atom 2 (a.u.)
        z_eff_1: Effective charge of atom 1
        z_eff_2: Effective charge of atom 2
        n_1: Principal quantum number of atom 1
        n_2: Principal quantum number of atom 2
        exp_1: Slater exponent of atom 1
        exp_2: Slater exponent of atom 2
        overlap_method: Method for overlap integral and overlap integral gradient:
            'formula' or 'interpolation'

    Returns:
        beta: Core resonance integral (a.u.)
    """
    # Calculate center of charge and distance from atom 1 to center of charge
    r_0 = (r * z_eff_1) / (z_eff_1 + z_eff_2)
    delta_r = r - 2 * r_0

    # Calculate overlap and its gradient
    overlap = slater_overlap(r, n_1, n_2, exp_1, exp_2, method=overlap_method)
    overlap_grad = slater_overlap_grad(r, n_1, n_2, exp_1, exp_2, method=overlap_method)

    t_1: float = (
        overlap / 2 / np.sqrt(1 - overlap**2) * delta_r / r * (alpha_1 - alpha_2)
    )
    t_2 = overlap_grad / r
    beta = t_1 + t_2

    return beta


class PPPCalculator:
    """Calculator Pariser-Parr-Pople molecular orbital theory.

    Args:
        input_data: Input data object
        charge: Molecular charge
        multiplicity: Multiplicity
        parametrization: Parametrization. Defaults to Beveridge-Hinze
        overlap_method: Method for calculating orbital overlap: 'formula' or
            'interpolation'

    Attributes:
        atom_types: Atom types
        charge: Molecular charge
        ci_coefficients: Configuration interaction coefficients
        ci_energies: State energies from configuration interaction (a.u.)
        ci_matrix: Configuration interaction matrix (a.u.)
        coefficients: Molecular orbital coefficients
        connectivity_matrix: Connectivity matrix
        converged: Whether SCF converged
        coordinates: Coordinates (Å)
        core_matrix: Core matrix (a.u.)
        density_matrix: Density matrix
        distance_matrix: Distance matrix (a.u.)
        electronic_energy: Electronic energy (a.u.)
        electrons: Electrons contributed by each atom
        excitations: Info on excitations
        exponents: Slater-type orbital exponents
        fock_matrix_mo: Fock matrix in molecular orbital basis (a.u.)
        fock_matrix: Fock matrix in atomic orbital basis (a.u.)
        gamma_matrix: Electron repulsion integrals in atomic orbital basis (a.u.)
        hc: Hückel calculator
        homo_idx: Index of HOMO
        ips: Valence state ionization potentials (eV)
        iter_densities: Density RMSD for each SCF iteration
        iter_energies: Energies for each SCF iteration (a.u.)
        lumo_idx: Index of LUMO
        mo_integrals: Electron repulsion integrals in molecular orbital basis (a.u.)
        multiplicity: Multiplicity
        n_atoms: Number of atoms
        n_electrons: Number of electrons
        n_excitations: Number of excitations (including ground state)
        n_iter: Number of SCF iterations done
        n_occupied: Number of occupied orbitals
        n_orbitals: Number of orbitals
        n_states: Number of lowest states requested in the configuration interaction
        n_virtual: Number of virtual orbitals
        ns: Principal quantum numbers of atomic orbitals
        occupations: Occupation number of each orbital
        oscillator_strengths: Oscillator strengths for excitation from ground state
        overlap_matrix: Overlap matrix for Slater-type orbitals
        parametrization: Parametrization
        transition_densities: Transition densities
        transition_dipole_moments: Transition dipole moments (a.u.)
    """

    atom_types: Sequence[str]
    charge: int
    ci_coefficients: Array2DFloat
    ci_energies: Array1DFloat
    ci_matrix: Array2DFloat
    coefficients: Array2DFloat
    connectivity_matrix: Array2DInt
    converged: bool
    coordinates: Array2DFloat
    core_matrix: Array2DFloat
    density_matrix: Array2DFloat
    distance_matrix: Array2DFloat
    electronic_energy: float
    electrons: list[int]
    excitations: list[tuple[tuple[int, ...], tuple[int, ...], str]]
    exponents: Array1DFloat
    fock_matrix_mo: Array2DFloat
    fock_matrix: Array2DFloat
    gamma_matrix: Array2DFloat
    hc: HuckelCalculator
    homo_idx: int
    ips: Array1DFloat
    iter_densities: list[float]
    iter_energies: list[float]
    lumo_idx: int
    mo_integrals: Array4DFloat
    multiplicity: int
    n_atoms: int
    n_electrons: int
    n_excitations: int
    n_iter: int
    n_occupied: int
    n_orbitals: int
    n_states: int
    n_virtual: int
    ns: list[int]
    occupations: list[int]
    oscillator_strengths: Array1DFloat
    overlap_matrix: Array2DFloat
    parametrization: Mapping[str, Any]
    transition_densities: Array1DFloat
    transition_dipole_moments: Array1DFloat
    _iter_populations: list[Array1DFloat]
    _populations_old: Array1DFloat | None

    def __init__(
        self,
        input_data: InputData,
        charge: int = 0,
        multiplicity: int | None = None,
        parametrization: Mapping[str, Any] | None = None,
        overlap_method: str = "interpolation",
    ) -> None:
        # Process input data
        if parametrization is None:
            parametrization = BEVERIDGE_HINZE
        atom_types = input_data.atom_types
        n_atoms = len(atom_types)
        coordinates = input_data.coordinates
        connectivity_matrix = input_data.connectivity_matrix
        twist_angles = input_data.twist_angles
        if twist_angles is None:
            twist_angles = np.zeros((n_atoms, n_atoms))
        self.twist_angles = twist_angles
        self._input_data = input_data
        sigma_charges = input_data.sigma_charges
        if sigma_charges is None:
            sigma_charges = np.zeros(n_atoms)
        self.sigma_charges = sigma_charges
        self.overlap_method = overlap_method

        # Get Hückel matrix and number of electrons
        huckel_matrix, electrons = prepare_huckel_matrix(
            atom_types, connectivity_matrix
        )

        # Calculate number of electrons
        n_electrons = sum(electrons) - charge

        # Set up Hückel calculator
        hc = HuckelCalculator(
            huckel_matrix, electrons, charge=charge, multiplicity=multiplicity
        )
        self.density_matrix = hc.bo_matrix

        # Calculate distance matrix
        distance_matrix = (
            scipy.spatial.distance_matrix(coordinates, coordinates) * ANGSTROM_TO_BOHR
        )

        self.electrons = electrons
        self.multiplicity = hc.multiplicity
        self.n_electrons = n_electrons
        self.n_orbitals = hc.n_orbitals
        self.n_occupied = hc.n_occupied
        self.n_virtual = self.n_orbitals - self.n_occupied
        self.homo_idx = self.n_occupied - 1
        self.lumo_idx = self.homo_idx + 1
        self.occupations = hc.occupations
        self.charge = charge
        self.atom_types = atom_types
        self.connectivity_matrix: Array2DInt = np.array(connectivity_matrix)
        self.distance_matrix = distance_matrix
        self.coordinates: Array2DFloat = np.array(coordinates)
        self.n_atoms = n_atoms
        self.hc = hc
        self.parametrization = parametrization

        # Set up
        self._setup_ips()
        self.ns = [N[atom_type] for atom_type in atom_types]
        self.n_stars = [N_STAR[n] for n in self.ns]
        self._setup_z_effs()

        # Calculate matrices
        self._setup_gamma_matrix()
        self._setup_exponents()
        self._setup_overlap_matrix()

        # Sigma charge adjustment
        if np.count_nonzero(self.sigma_charges) > 0:
            self._sigma_ve_adjustment()

    def _setup_ips(self) -> None:
        """Set up valence state ionization potentials."""
        # Use Beveridge-Hinze values
        if self.parametrization["data"] == "beveridge-hinze":
            self.ips = [IP_BH[atom_type] for atom_type in self.atom_types]
        # Use CRC handbook values
        elif self.parametrization["data"] == "crc":
            self.ips = [IP_CRC[atom_type] for atom_type in self.atom_types]

    def _setup_z_effs(self) -> None:
        #        if np.count_nonzero(self.sigma_charges) > 0:
        #            z_effs = []
        #            for atom_type, sigma_charge in zip(self.atom_types, self.sigma_charges):
        #                z_eff = z_eff_slater(
        #                    ELECTRON_CONFIGURATIONS[atom_type],
        #                    charge=CHARGES[atom_type],
        #                    population=1 - sigma_charge,
        #                )
        #                z_effs.append(z_eff)
        #        else:
        z_effs = [Z_EFF[atom_type] for atom_type in self.atom_types]
        self.z_effs = z_effs

    def _setup_exponents(self) -> None:
        """Calculate Slater orbital exponents."""
        # Use Beveridge-Hinze formula
        if self.parametrization["exponent"] == "beveridge-hinze":
            exponents = [get_exp_bv(gamma) for gamma in self.gamma_matrix.diagonal()]
        # Use Slater's rules
        elif self.parametrization["exponent"] == "slater":
            exponents = [
                exp_slater_from_z_eff(z_eff, n_star)
                for z_eff, n_star in zip(self.z_effs, self.n_stars)
            ]
        self.exponents: Array1DFloat = np.array(exponents)

    def _get_beta(self, i: int, j: int) -> float:
        """Calculate beta.

        Args:
            i: Index for atom 1
            j: Index for atom 2

        Returns:
            beta: Two-center core resonance integral.
        """
        r = self.distance_matrix[i, j]
        # Use Beveridge-Hinze formula
        if self.parametrization["beta"] == "beveridge-hinze":
            c = self.parametrization["parameters"]["c"]
            charge_i, charge_j = self.electrons[i], self.electrons[j]
            overlap = self.overlap_matrix[i, j]
            gamma = self.gamma_matrix[i, j]
            beta = get_beta_bh(r, gamma, overlap, charge_i, charge_j, c)
        # Use Linderberg formula
        elif self.parametrization["beta"] == "linderberg":
            overlap = self.overlap_matrix[i, j]
            n_i, n_j = self.ns[i], self.ns[j]
            exp_i, exp_j = self.exponents[i], self.exponents[j]
            beta = get_beta_l(
                r, n_i, n_j, exp_i, exp_j, overlap_method=self.overlap_method
            )
        # Use Jug formula
        elif self.parametrization["beta"] == "jug":
            alpha_i, alpha_j = self.core_matrix[i, i], self.core_matrix[j, j]
            z_eff_i, z_eff_j = self.z_effs[i], self.z_effs[j]
            n_i, n_j = self.ns[i], self.ns[j]
            exp_i, exp_j = self.exponents[i], self.exponents[j]
            beta = get_beta_jug(
                r,
                alpha_i,
                alpha_j,
                z_eff_i,
                z_eff_j,
                n_i,
                n_j,
                exp_i,
                exp_j,
                overlap_method=self.overlap_method,
            )

        # Adjust beta according to twist angle
        beta *= np.cos(np.deg2rad(self.twist_angles[i, j]))

        return beta

    def _setup_gamma_matrix(self) -> None:
        """Calculate electron repulsion integrals in atomic orbital basis."""
        gamma_matrix = np.zeros((self.n_atoms, self.n_atoms))

        # Set up diagonal elements
        diag_indices = np.diag_indices_from(gamma_matrix)
        for i, i in zip(*diag_indices):
            atom_type = self.atom_types[i]
            gamma = get_gamma_xx(atom_type, data=self.parametrization["data"])
            gamma_matrix[i, i] = gamma

        # Set up off-diagonal elements
        tril_indices = np.tril_indices_from(gamma_matrix, k=-1)
        for i, j in zip(*tril_indices):
            r = self.distance_matrix[i, j]
            gamma_ii = gamma_matrix[i, i]
            gamma_jj = gamma_matrix[j, j]
            gamma = get_gamma_xy(
                r, gamma_ii, gamma_jj, method=self.parametrization["gamma_xy"]
            )
            gamma_matrix[i, j] = gamma

        # Form symmetrix matrix
        gamma_matrix = gamma_matrix + gamma_matrix.T - np.diag(gamma_matrix.diagonal())

        self.gamma_matrix = gamma_matrix

    def _setup_overlap_matrix(self) -> None:
        """Calculate overlap matrix."""
        overlap_matrix = np.zeros((self.n_atoms, self.n_atoms))
        tril_indices = np.tril_indices_from(overlap_matrix)
        for i, j in zip(*tril_indices):
            if self.connectivity_matrix[i, j] == 0:
                continue
            r = self.distance_matrix[i, j]
            overlap = slater_overlap(
                r,
                self.ns[i],
                self.ns[j],
                self.exponents[i],
                self.exponents[j],
                method=self.overlap_method,
            )
            overlap_matrix[i, j] = overlap
        overlap_matrix = (
            overlap_matrix + overlap_matrix.T - np.diag(overlap_matrix.diagonal())
        )
        self.overlap_matrix = overlap_matrix

    def _setup_fock_matrix(self) -> None:
        """Calculate Fock matrix."""
        # Set up fock matrix and core matrix
        fock_matrix = np.zeros((self.n_atoms, self.n_atoms))
        core_matrix = np.zeros((self.n_atoms, self.n_atoms))

        # Calculate diagonal elements of Fock matrix
        diag_indices = np.diag_indices_from(fock_matrix)

        for i, j in zip(*diag_indices):
            U = -self.ips[i] * EV_TO_HARTREE
            gamma = self.gamma_matrix[i, i]
            p = self.density_matrix[i, i]
            summed = 0
            h_summed = 0
            for k in range(self.n_atoms):
                if k != i:
                    gamma_ik = self.gamma_matrix[i, k]
                    h_summed += -self.electrons[k] * gamma_ik
                    summed += (self.density_matrix[k, k]) * gamma_ik
            h = U + h_summed
            f = U + gamma * p / 2 + summed + h_summed
            fock_matrix[i, j] = f
            core_matrix[i, j] = h

        # Store matrices as they might be used to calculate off-diagonal elements
        self.fock_matrix = fock_matrix
        self.core_matrix = core_matrix

        # Calculate off-diagonal elements
        tril_indices = np.tril_indices_from(fock_matrix, k=-1)
        for i, j in zip(*tril_indices):
            gamma = self.gamma_matrix[i, j]
            if self.connectivity_matrix[i, j] != 0:
                beta = self._get_beta(i, j)
            else:
                beta = 0
            p = self.density_matrix[i, j]
            f = beta - gamma * p / 2
            fock_matrix[i, j] = f
            core_matrix[i, j] = beta

        fock_matrix = fock_matrix + fock_matrix.T - np.diag(fock_matrix.diagonal())
        core_matrix = core_matrix + core_matrix.T - np.diag(core_matrix.diagonal())

        self.fock_matrix = fock_matrix
        self.core_matrix = core_matrix

        if self.parametrization["sane_correction"] is True:
            self._sane_correction()

    def _sane_correction(self) -> None:
        """Perform correction to core matrix according to Sane and Sane."""
        correction = np.sum(
            self.core_matrix * self.connectivity_matrix * self.overlap_matrix, axis=1
        )
        np.fill_diagonal(self.fock_matrix, self.fock_matrix.diagonal() - correction)
        np.fill_diagonal(self.core_matrix, self.core_matrix.diagonal() - correction)

    def _eig(self) -> None:
        """Diagonalize fock matrix to get orbital energies and coefficients."""
        energies, coefficients = np.linalg.eigh(self.fock_matrix)
        self.orbital_energies = energies
        self.coefficients = coefficients.T

    def _setup_density_matrix(self) -> None:
        """Calculate density matrix from coefficients and occupations."""
        density_matrix = (
            self.coefficients.T @ np.diag(self.occupations) @ self.coefficients
        )
        self.density_matrix = density_matrix

    def _calculate_total_energy(self) -> float:
        energy: float = (
            np.sum(self.density_matrix * (self.fock_matrix + self.core_matrix)) / 2
        )
        return energy

    def scf(  # noqa: C901
        self,
        conv_e: float = 1e-6,
        conv_d: float = 1e-4,
        conv_ve: float = 1e-3,
        max_iter: int = 100,
        ve: bool = False,
        ve_damping: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """Do SCF calculation.

        Args:
            conv_e: Energy difference convergence threshold (a.u.)
            conv_d: Density matrix RMSD convergence threshold
            conv_ve: Charge difference convergence threshold
            max_iter: Maximum number of iterations
            ve: Whether to do variable electronegativity SCF
            ve_damping: Damping factor for VESCF iterations
            verbose: Whether to print for each iteration
        """
        delta_e = 1.0
        rmsd_d = 1.0
        n_iter = 0
        n_iter_ve = 0
        energy = 0.0
        iter_energies = []
        iter_densities = []
        iter_energies_ve = []
        iter_densities_ve = []

        if verbose is True:
            print(f"{'Iter':>5s}{'Total energy':>15s}{'D RSMD':>15s}{'Delta E':>15s}")

        converged_scf = False
        converged_ve = False

        if ve is True:
            self.z_effs_orig = self.z_effs
            self.exponents_orig = self.exponents
            self.gamma_matrix_orig = self.gamma_matrix
            self.ips_orig = self.ips
            self._populations_old = None
            self._iter_populations = []

        while (converged_ve is False) and (n_iter < max_iter):
            # Variable electronegativity adjustment
            if ve is True:
                if n_iter_ve > 0:
                    self._ve_adjustment(damping=ve_damping)
                charges_old = self.density_matrix.diagonal()
                density_matrix_old_ve = self.density_matrix
                energy_old_ve = energy
            while (converged_scf is False) and (n_iter < max_iter):
                # Do an iteration
                self._setup_fock_matrix()
                self._eig()

                # Update density matrix and calculate rmsd
                density_matrix_old = self.density_matrix
                self._setup_density_matrix()
                rmsd_d = np.sqrt(
                    np.sum((self.density_matrix - density_matrix_old) ** 2)
                )

                # Calculate total energy
                energy_old = energy
                energy = self._calculate_total_energy()
                delta_e_signed = energy - energy_old
                delta_e = abs(delta_e_signed)

                # Print out iteration
                if verbose is True and n_iter > 0:
                    print(
                        f"{n_iter:5d}{energy:15.6f}{rmsd_d:15.6E}{delta_e_signed:15.6E}"
                    )

                if n_iter > 0:
                    iter_energies.append(delta_e_signed)
                    iter_densities.append(rmsd_d)

                if (rmsd_d < conv_d) and (delta_e < conv_e):
                    converged_scf = True

                n_iter += 1

            if ve is True:
                rmsd_d_ve = np.sqrt(
                    np.sum((self.density_matrix - density_matrix_old_ve) ** 2)
                )
                delta_e_signed_ve = energy - energy_old_ve
                charges = self.density_matrix.diagonal()
                delta_max_charges = np.max(charges - charges_old)
                if abs(delta_max_charges) < conv_ve:
                    converged_ve = True
                else:
                    converged_scf = False
                if n_iter_ve > 0:
                    iter_energies_ve.append(delta_e_signed_ve)
                    iter_densities_ve.append(rmsd_d_ve)
            else:
                converged_ve = True

            n_iter_ve += 1

        self.n_iter = n_iter
        self.n_iter_ve = n_iter_ve
        self.iter_energies = iter_energies
        self.iter_densities = iter_densities
        self.iter_energies_ve = iter_energies_ve
        self.iter_densities_ve = iter_densities_ve
        self.electronic_energy = energy

    def _setup_mo_integrals(self) -> None:
        """Calculate MO integrals."""
        C = self.coefficients.T
        self.mo_integrals = np.einsum(
            "ui,uj,ak,al,ua->ijkl", C, C, C, C, self.gamma_matrix, optimize=True
        )

    def _setup_fock_matrix_mo(self) -> None:
        """Set up Fock matrix in molecular orbital basis."""
        self.fock_matrix_mo = self.coefficients @ self.fock_matrix @ self.coefficients.T

    def _sigma_ve_adjustment(self) -> None:
        """Do adjustment based on sigma framework."""
        self.z_effs_orig = self.z_effs
        self.gamma_matrix_orig = self.gamma_matrix

        # Calculate new effective potentials and scale IPs
        z_effs = []
        ips = []
        for atom_type, sigma_charge in zip(self.atom_types, self.sigma_charges):
            coefficients = VE_COEFFICIENTS[atom_type]
            electrons = ELECTRON_CONFIGURATIONS[atom_type]
            charge = CHARGES[atom_type]
            z_eff = z_eff_slater(electrons, charge=charge, population=1 - sigma_charge)
            z_effs.append(z_eff)
            ip = (
                coefficients[0] * z_eff**2 + coefficients[1] * z_eff + coefficients[2]
            )
            ips.append(ip)
        z_effs: Array1DFloat = np.array(z_effs)
        ips: Array1DFloat = np.array(ips)

        self.ips = ips
        self.z_effs = z_effs

        # Scale the gamma matrix and set up new orbital overlap and exponents.
        self._scale_gamma_matrix()
        self._setup_exponents()
        self._setup_overlap_matrix()

    def _ve_adjustment(self, damping: float = 0.5) -> None:
        """Do variable electronegativity adjustments.

        Args:
            damping: Damping factor for population changes.
        """
        # Set up new populations
        populations_new = self.density_matrix.diagonal()
        if self._populations_old is None:
            populations = populations_new
        else:
            populations = self._populations_old + damping * (
                populations_new - self._populations_old
            )
        self._populations_old = populations
        self._iter_populations.append(populations)

        # Calculate new effective charges and scale IPs
        ips = []
        z_effs = []
        for atom_type, population, sigma_charge in zip(
            self.atom_types, populations, self.sigma_charges
        ):
            coefficients = VE_COEFFICIENTS[atom_type]
            electrons = ELECTRON_CONFIGURATIONS[atom_type]
            charge = CHARGES[atom_type]
            z_eff = z_eff_slater(
                electrons, charge=charge, population=population - sigma_charge
            )
            z_effs.append(z_eff)
            ip = (
                coefficients[0] * z_eff**2 + coefficients[1] * z_eff + coefficients[2]
            )
            ips.append(ip)

        self.ips = ips
        self.z_effs = z_effs

        # Scale gamma matrix and overlap matrix.
        self._scale_gamma_matrix()
        self._setup_exponents()
        self._setup_overlap_matrix()

    def _scale_gamma_matrix(self) -> None:
        """Scale gamma matrix based on effective charges."""
        gamma_xx = self.gamma_matrix_orig.diagonal() * self.z_effs / self.z_effs_orig

        gamma_matrix = np.zeros((self.n_atoms, self.n_atoms))
        tril_indices = np.tril_indices_from(gamma_matrix, k=-1)
        for i, j in zip(*tril_indices):
            r = self.distance_matrix[i, j]
            gamma_ii = gamma_xx[i]
            gamma_jj = gamma_xx[j]
            gamma = get_gamma_xy(
                r,
                gamma_ii,
                gamma_jj,
                method=self.parametrization["gamma_xy"],
            )
            gamma_matrix[i, j] = gamma
        gamma_matrix = gamma_matrix + gamma_matrix.T
        np.fill_diagonal(gamma_matrix, gamma_xx)

        self.gamma_matrix = gamma_matrix

    def ci(
        self,
        n_states: int | None = None,
        multiplicity: str = "singlet",
        calculate_f: bool = True,
        f_type: str = "length",
    ) -> None:
        """Do configuration interaction with single and or double excitations.

        Formulas for double excitations are probably erroneous so results are not
        completely reliable.

        Args:
            n_states: Number of states to solve for, including the ground state
            multiplicity: Total multiplicity 'singlet' (default) or 'triplet'
            calculate_f: Whether to calculate oscillator strengths.
            f_type: Approximation for oscillator strenghts: 'length' (default) or
                'velocity'

        Raises:
            ValueError: When n_states exceeds number of possible states.
        """
        # Enumerate excitations
        excitations = generate_excitations(
            self.n_occupied,
            self.n_virtual,
        )

        # Setup fock matrix and electron repulsion integrals in MO basis
        self._setup_fock_matrix()
        self._setup_fock_matrix_mo()
        self._setup_mo_integrals()

        # Calculate matrix elements of CI matrix
        n_excitations = len(excitations)
        if n_states is None:
            n_states = n_excitations
        if n_states > n_excitations:
            raise ValueError(f"{n_states} requested but only {n_excitations} possible.")
        ci_matrix = np.zeros((n_excitations, n_excitations))
        tril_indices = np.tril_indices_from(ci_matrix)
        for i, j in zip(*tril_indices):
            excitation_1 = excitations[i]
            excitation_2 = excitations[j]
            matrix_element = calculate_matrix_element(
                excitation_1,
                excitation_2,
                self.electronic_energy,
                self.fock_matrix_mo,
                self.mo_integrals,
                multiplicity=multiplicity,
            )
            ci_matrix[i, j] = matrix_element
        ci_matrix = ci_matrix + ci_matrix.T - np.diag(ci_matrix.diagonal())

        # Diagonalized CI matrix to get state energies and coefficients
        if n_states == n_excitations:
            ci_energies, ci_coefficients = np.linalg.eigh(ci_matrix)
        else:
            ci_energies, ci_coefficients = scipy.sparse.linalg.eigsh(
                ci_matrix, k=n_states
            )

        ci_coefficients = ci_coefficients.T
        self.ci_matrix = ci_matrix
        self.ci_energies = ci_energies
        self.ci_coefficients = ci_coefficients
        self.excitations = excitations
        self.n_excitations = n_excitations
        self.n_states = n_states

        # Calculate oscillator strengths
        if calculate_f is True:
            self.calculate_oscillator_strengths(f_type=f_type)

    def calculate_oscillator_strengths(self, f_type: str = "length") -> None:
        """Calculate oscillator strengths.

        Formulas for dipole length and dipole velocity versions of the oscillator
        strength as in 10.1002/qua.560040606.

        Args:
            f_type: Approximation for oscillator strenghts: 'length' (default) or
                'velocity'
        """
        oscillator_strengths = []

        # According to dipole length formula
        if f_type == "length":
            # Calculate transition densities
            transition_densities_csf = []
            for excitation in self.excitations[1:]:
                i, j, _ = excitation
                transition_density = self.coefficients[i] * self.coefficients[j]
                transition_densities_csf.append(transition_density)
            transition_densities_csf: Array2DFloat = np.vstack(transition_densities_csf)

            # Calculate oscillator strength, transition density and transition dipole
            # moment
            transition_dipole_moments = []
            transition_densities = []
            for k in range(1, self.n_states):
                transition_density = np.sum(
                    transition_densities_csf
                    * self.ci_coefficients[k, 1:].reshape(-1, 1),
                    axis=0,
                )
                transition_dip_moment = (
                    np.sqrt(2)
                    * np.sum(
                        transition_density.reshape(-1, 1) * self.coordinates, axis=0
                    )
                    * ANGSTROM_TO_BOHR
                )
                excitation_energy = self.ci_energies[k] - self.ci_energies[0]
                oscillator_strength = (
                    2
                    / 3
                    * excitation_energy
                    * np.linalg.norm(transition_dip_moment) ** 2
                )
                oscillator_strengths.append(oscillator_strength)
                transition_dipole_moments.append(transition_dip_moment)
                transition_densities.append(transition_density)
            transition_dipole_moments: Array1DFloat = np.array(
                transition_dipole_moments
            )
            transition_densities: Array1DFloat = np.array(transition_densities)

            self.transition_dipole_moments = transition_dipole_moments
            self.transition_densities = transition_densities
            self.transition_densities_csf = transition_densities_csf
        # Calculate according to dipole velocity formula
        elif f_type == "velocity":
            # Calculate distances
            distances = self.coordinates[:, None, :] - self.coordinates[None, :, :]
            distances = distances.swapaxes(0, 1) * ANGSTROM_TO_BOHR

            # Calculate transition dipole velocity moments
            ps = []
            for excitation in self.excitations[1:]:
                i, j, _ = excitation
                p = (
                    np.sum(
                        (
                            np.outer(self.coefficients[i], self.coefficients[j])
                            * self.core_matrix
                        )[:, :, None]
                        * distances,
                        axis=(0, 1),
                    )
                    * 1j
                )
                ps.append(p)
            ps: Array1DFloat = np.array(ps)

            # Calculate oscillator strength
            for k in range(1, self.n_states):
                p = np.sqrt(2) * np.sum(
                    ps * self.ci_coefficients[k, 1].reshape(-1, 1),
                    axis=0,
                )
                excitation_energy = self.ci_energies[i] - self.ci_energies[0]
                oscillator_strength = 2 / 3 * np.linalg.norm(p) ** 2 / excitation_energy
                oscillator_strengths.append(oscillator_strength)
        oscillator_strengths: Array1DFloat = np.array(oscillator_strengths)

        self.oscillator_strengths = oscillator_strengths

    def calculate_mo_integral(self, i: int, j: int, k: int, l: int) -> float:
        """Calculate electron repulsion integral in molecular orbital basis.

        Args:
            i: Index of orbital 1
            j: Index of orbital 2
            k: Index of orbital 3
            l: Index of orbital 4

        Returns:
            mo_integral: Electron repulsion integral in molecular orbital basis.
        """
        mo_integral: float = np.sum(
            np.outer(
                self.coefficients[j] * self.coefficients[i],
                self.coefficients[k] * self.coefficients[l],
            )
            * self.gamma_matrix
        )
        return mo_integral

    def get_huckel_matrix(self, method: str = "spectroscopic") -> Array2DFloat:
        """Generate Hückel matrix from PPP fock matrix.

        Following the procedure in 10.1007/BF00528229. Hückel matrix needs to be
        converted to standard format in a separate procedure.

        Args:
            method: Method 'spectroscopic' (default) or 'additive'

        Returns:
            huckel_matrix: Hückel matrix (a.u.)

        Raises:
            ValueError: If method is not supported.
        """
        huckel_matrix: Array2DFloat = np.array(self.fock_matrix)

        # Use spectroscopic method
        if method == "spectroscopic":
            pass
        # Use energy additive method
        elif method == "additive":
            corr_matrix = np.zeros(huckel_matrix.shape)
            for i, j in zip(*np.tril_indices_from(huckel_matrix)):
                if i == j:
                    corr_matrix[i, i] -= (
                        1 / 4 * self.density_matrix[i, i] * self.gamma_matrix[i, i]
                    )
                    for k in range(self.n_atoms):
                        if i != k:
                            corr_matrix[i, i] -= (
                                1
                                / 2
                                * self.density_matrix[k, k]
                                * self.gamma_matrix[i, k]
                            )
                else:
                    corr_matrix[i, j] += (
                        1 / 4 * self.distance_matrix[i, j] * self.gamma_matrix[i, j]
                    )
            corr_matrix = corr_matrix + corr_matrix.T - np.diag(corr_matrix.diagonal())
            huckel_matrix += corr_matrix
        else:
            raise ValueError("Available methods are: 'spectroscopic' and 'additive'.")

        # Set matrix elements between non-neighbors to zero
        huckel_matrix *= self.connectivity_matrix + np.eye(self.n_atoms)

        return huckel_matrix

    def _get_c_matrix(self, state: int) -> Array2DFloat:
        if state < 1:
            raise ValueError(f"Specified state is {state} but needs to be > 0.")
        elif state > self.n_states:
            raise ValueError(
                f"Specified state is {state} but only {self.n_states} possible."
            )
        c_matrix = np.zeros((self.n_occupied, self.n_virtual))
        for idx, excitation in enumerate(self.excitations[1:]):
            i = excitation[0][0]
            a = excitation[1][0] - self.n_occupied
            c_matrix[i, a] = self.ci_coefficients[state][1:][idx]
        return c_matrix

    def get_ntos(self, state: int) -> tuple[Array2DFloat, Array2DFloat, Array1DFloat]:
        """Calculate natural transition orbitals for state.

        Args:
            state: State

        Returns:
            ntos_hole: Hole natural transition orbitals
            ntos_particle: Particle natural transition orbitals
            occupation numbers: Occupation numbers
        """
        # Calculate coefficients matrix for state in question
        occ_orbs = self.coefficients[: self.n_occupied].T
        virt_orbs = self.coefficients[self.n_occupied :].T
        c_matrix = self._get_c_matrix(state)
        tdm = c_matrix
        U, s, V_t = np.linalg.svd(tdm, full_matrices=True)
        V = V_t.T
        ntos_hole = (occ_orbs @ U).T
        ntos_particle = (virt_orbs @ V).T
        occupation_numbers = s**2

        return ntos_hole, ntos_particle, occupation_numbers

    def get_ct_analysis(self, state: int) -> dict[str, Any]:
        """Perform charge transfer analysis for state.

        Args:
            state: State of interest

        Returns:
            results: Dictionary of results
        """
        c_matrix = self._get_c_matrix(state)
        occ_orbs = self.coefficients[: self.n_occupied].T
        virt_orbs = self.coefficients[self.n_occupied :].T

        # Calculate attachment and detachment density matrices in MO basis
        ddm_mo = np.einsum("ia,ja->ij", c_matrix, c_matrix)
        adm_mo = np.einsum("ia,ib->ab", c_matrix, c_matrix)

        # Concert to AO basis
        ddm_ao = np.einsum("pq,up,vq->uv", ddm_mo, occ_orbs, occ_orbs)
        adm_ao = np.einsum("pq,up,vq->uv", adm_mo, virt_orbs, virt_orbs)

        # Get attachment and detachment atomic densities
        detachment_densities = ddm_ao.diagonal()
        attachment_densities = adm_ao.diagonal()

        # Calculate density difference matrix and positive and negative densities
        delta_p_ao = adm_ao - ddm_ao
        delta_p_densities = delta_p_ao.diagonal()
        delta_p_plus_densities = delta_p_densities.clip(0)
        delta_p_minus_densities = delta_p_densities.clip(-np.inf, 0)

        # Do real-space CT analysis according to Ciofini
        centroid_plus = np.sum(
            delta_p_plus_densities.reshape(-1, 1) * self.coordinates, axis=0
        ) / np.sum(delta_p_plus_densities)
        centroid_minus = np.sum(
            delta_p_minus_densities.reshape(-1, 1) * self.coordinates, axis=0
        ) / np.sum(delta_p_minus_densities)
        d_ct_vector = centroid_plus - centroid_minus
        d_ct = np.linalg.norm(d_ct_vector)
        d_ct_a = (
            np.sum(
                np.outer(delta_p_minus_densities, delta_p_plus_densities)
                * self.distance_matrix
            )
            / np.outer(delta_p_minus_densities, delta_p_plus_densities).sum()
            * BOHR_TO_ANGSTROM
        )

        # Do Hilbert space CT analysis according to Etienne
        delta = (np.sum(attachment_densities) + np.sum(detachment_densities)) / 2
        phi_s = np.sum(np.sqrt(attachment_densities * detachment_densities)) / delta
        dd_centroid = (
            np.sum(detachment_densities.reshape(-1, 1) * self.coordinates, axis=0)
            / delta
        )
        ad_centroid = (
            np.sum(attachment_densities.reshape(-1, 1) * self.coordinates, axis=0)
            / delta
        )
        zeta_vector = ad_centroid - dd_centroid
        zeta = np.linalg.norm(zeta_vector)
        zeta_a = (
            np.sum(
                np.outer(attachment_densities, detachment_densities)
                * self.distance_matrix
            )
            / np.outer(attachment_densities, detachment_densities).sum()
            * BOHR_TO_ANGSTROM
        )

        # Set up results
        results = {
            "attachment_density_matrix": adm_ao,
            "detachment_density_matrix": ddm_ao,
            "density_difference_matrix": delta_p_ao,
            "d_ct_vector": d_ct_vector,
            "d_ct": d_ct,
            "d_ct_average": d_ct_a,
            "positive_barycenter": centroid_plus,
            "negative_barycenter": centroid_minus,
            "phi_s": phi_s,
            "zeta_vector": zeta_vector,
            "zeta": zeta,
            "zeta_average": zeta_a,
            "attachment_centroid": ad_centroid,
            "detachment_centroid": dd_centroid,
        }

        return results


def homo_lumo_overlap(ppp: PPPCalculator) -> float:
    """Calculate HOMO-LUMO spatial overlap.

    Args:
        ppp: PPPCalculator object

    Returns:
        overlap: Orbital overlap
    """
    overlap: float = np.dot(
        np.abs(ppp.coefficients[ppp.homo_idx]), np.abs(ppp.coefficients[ppp.lumo_idx])
    )
    return overlap


def calculate_exchange(ppp: PPPCalculator) -> float:
    """Calculate exchange integral between HOMO and LUMO.

    Args:
        ppp: PPPCalculator object

    Returns:
        exchange: Exchange integral (a.u.)
    """
    homo_idx = ppp.homo_idx
    lumo_idx = ppp.lumo_idx
    exchange = ppp.calculate_mo_integral(homo_idx, lumo_idx, homo_idx, lumo_idx)
    return exchange


def calculate_dsp(
    ppp: PPPCalculator,
    ci: bool = False,
    energy_s_1: float = None,
    energy_t_1: float = None,
) -> float:
    """Calculate dynamic spin polarization difference with perturbation theory.

    Spin polarization difference between singlet and triplet HOMO->LUMO excited
    states. Approach from 10.1007/BF00549021. Perturbation theory on CIS states
    following 10.1016/0009-2614(94)00070-0. Negative values indicate singlet is
    more stabilized than triplet.

    Args:
        ppp: PPPCalculator object
        ci: Calculate DSP relative to the CIS states
        energy_s_1: CIS energy of S1 state (a.u.)
        energy_t_1: CIS energy of T1 state (a.u.)

    Returns:
        dsp: Stabilization of singlet over triplet (a.u.)
        excitations: DSP values for each excitaiton (a.u.)
    """
    # Set up variables
    n_occupied = ppp.n_occupied
    n_virtual = ppp.n_virtual
    homo_idx = ppp.homo_idx
    lumo_idx = ppp.lumo_idx

    # Calculate MO integrals and Fock matrix in MO basis
    ppp._setup_mo_integrals()
    ppp._setup_fock_matrix_mo()

    if ci is True:
        if energy_s_1 is None:
            ppp.ci(n_states=3, multiplicity="singlet")
            energy_s_1 = ppp.ci_energies[1] - ppp.ci_energies[0]
        if energy_t_1 is None:
            ppp.ci(n_states=3, multiplicity="triplet")
            energy_t_1 = ppp.ci_energies[1] - ppp.ci_energies[0]

    # Generate all single excitations
    single_excitations = list(
        itertools.product(
            range(n_occupied - 1), range(n_occupied + 1, n_occupied + n_virtual)
        )
    )

    # Do perturbation
    excitations = {}
    for i, j in single_excitations:
        k_x = ppp.mo_integrals[i, homo_idx, homo_idx, j]
        k_y = ppp.mo_integrals[i, lumo_idx, lumo_idx, j]
        if ci is True:
            gap_s = (
                ppp.fock_matrix_mo[j, j]
                + ppp.fock_matrix_mo[lumo_idx, lumo_idx]
                - ppp.fock_matrix_mo[homo_idx, homo_idx]
                - ppp.fock_matrix_mo[i, i]
                - energy_s_1
            )
            gap_t = (
                ppp.fock_matrix_mo[j, j]
                + ppp.fock_matrix_mo[lumo_idx, lumo_idx]
                - ppp.fock_matrix_mo[homo_idx, homo_idx]
                - ppp.fock_matrix_mo[i, i]
                - energy_t_1
            )
        else:
            gap_s = gap_t = ppp.fock_matrix_mo[j, j] - ppp.fock_matrix_mo[i, i]
        s_1 = (3 / 2) * (k_x - k_y) ** 2 / gap_s
        t_1 = (1 / 2) * (k_x - k_y) ** 2 / gap_t
        t_2 = (k_x + k_y) ** 2 / gap_t
        excitations[(i, j)] = {
            "s_1": s_1,
            "t_1": t_1,
            "t_2": t_2,
            "dsp": -s_1 + t_1 + t_2,
        }

    dsp = sum(excitation["dsp"] for excitation in excitations.values())

    return dsp, excitations
