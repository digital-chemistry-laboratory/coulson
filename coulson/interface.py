"""Interfaces to other programs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import functools
import itertools
import typing
from typing import Any, cast

import networkx as nx
import numpy as np
import scipy.spatial

from coulson.data import COV_RADII_PYYKKO, SYMBOLS_TO_NUMBERS
from coulson.matrix import InputData
from coulson.typing import (
    Array1DBool,
    Array1DFloat,
    Array1DInt,
    Array1DStr,
    Array2DFloat,
    Array2DInt,
    ArrayLike1D,
    ArrayLike2D,
)
from coulson.utils import Import, requires_dependency

if typing.TYPE_CHECKING:  # pragma: no cover
    import pyscf  # pragma: no cover
    from rdkit import Chem  # pragma: no cover

    from coulson.ppp import PPPCalculator  # pragma: no cover


@requires_dependency([Import(module="pyscf")], globals())
def get_pyscf_mf(ppp: PPPCalculator, spin: float | None = None) -> pyscf.scf.HF:
    """Converts PPP calculator object into PySCF SCF object.

    Args:
        ppp: PPP calculator
        spin: Number of unpaired electrons

    Returns:
        mf: PySCF SCF object
    """
    n = ppp.n_atoms
    ppp._setup_fock_matrix()

    # Create PySCF mol object
    mol = pyscf.gto.M()
    symbols = ["H" for atom_type in ppp.atom_types]
    atoms = list(zip(symbols, ppp.coordinates))
    mol = pyscf.gto.M(atom=atoms, basis="sto-3g", spin=None, charge=None)
    # mol.spin = None
    # mol.charge = None
    mol.incore_anyway = True
    mol.nelectron = ppp.n_electrons

    if spin is None:
        mol.spin = ppp.hc.n_unpaired
    else:
        mol.spin = spin
    mol.charge = ppp.charge

    # Create mf object and overwrite the default methods
    mf = pyscf.scf.HF(mol)
    overlap_matrix = np.eye(n)
    core_matrix = ppp.core_matrix
    fock_matrix = ppp.fock_matrix

    def energy_tot(
        self: pyscf.scf.HF,
        dm: Array2DFloat | None = None,
        h1e: Any | None = None,
        vhf: Any | None = None,
    ) -> float:
        """Modified total energy function that ignores nuclear interactions."""
        nuc = mf.energy_nuc()
        e_elec: float = mf.energy_elec(dm, h1e, vhf)[0]
        mf.scf_summary["nuc"] = nuc.real

        return e_elec

    mf.get_ovlp = lambda *args, **kwargs: overlap_matrix
    mf.get_hcore = lambda *args, **kwargs: core_matrix
    mf.get_fock = lambda *args, **kwargs: fock_matrix
    mf.energy_tot = functools.partial(energy_tot, mf)

    # Set up electron repulsion integrals
    M = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            M[i, i, j, j] = ppp.gamma_matrix[i, j]

    mf._eri = pyscf.ao2mo.restore(8, M, n)

    # Set initial guess from Hückel
    mf.init_guess = ppp.hc.bo_matrix

    return mf


def assign_atom_types(  # noqa: C901
    symbols: Iterable[str], degrees: Iterable[int]
) -> tuple[Array1DStr, Array1DInt]:
    """Assigns atom types based and which atoms to remove.

    Args:
        symbols: Atomic symbols
        degrees: Total connectivity degree of atom

    Returns:
        atom_types: Atom types
        to_remove: Atoms to remove

    Raises:
        NotImplementedError: When atom type does not have parameters.
    """
    atom_types: list[str | None] = []
    to_remove = []
    for i, (symbol, degree) in enumerate(zip(symbols, degrees)):
        if symbol == "H":
            atom_types.append("H")
            to_remove.append(i)
        elif symbol in ["N", "P"]:
            if degree == 1:
                atom_types.append(symbol + "1")
            elif degree == 2:
                atom_types.append(symbol + "1")
            elif degree == 3:
                atom_types.append(symbol + "2")
            else:
                raise NotImplementedError(
                    f"{symbol} with degree {degree} is not implemented!"
                )
        elif symbol in ["O", "S"]:
            if degree == 1:
                atom_types.append(symbol + "1")
            elif degree == 2:
                atom_types.append(symbol + "2")
            else:
                raise NotImplementedError(
                    f"{symbol} with degree {degree} is not implemented!"
                )
        elif symbol in ["C", "Si"]:
            if degree != 4:
                atom_types.append(symbol)
            else:
                to_remove.append(i)
                atom_types.append(None)
        elif symbol == "B":
            if degree == 3:
                atom_types.append(symbol)
            else:
                raise NotImplementedError(
                    f"{symbol} with degree {degree} is not implemented!"
                )
        elif symbol in ["F", "Cl"]:
            if degree == 1:
                atom_types.append(symbol)
            else:
                raise NotImplementedError(
                    f"{symbol} with degree {degree} is not implemented!"
                )
        else:
            raise NotImplementedError(f"Atom type {symbol} is not implemented!")
    to_remove: Array1DInt = np.array(to_remove, dtype=int)
    atom_types: Array1DStr = np.array(atom_types)

    return atom_types, to_remove


def connectivity_from_coordinates(
    symbols: Sequence[str], coordinates: ArrayLike2D, scale: float = 1.2
) -> Array2DInt:
    """Returns connectivity matrix from coordinates based on sum of covalent radii.

    Args:
        symbols: Symbols
        coordinates: Coordinates (Å)
        scale: Factor to multiply the sum of the covalent radii with.

    Returns:
        connectivity_matrix: Connectivity matrix
    """
    # Get atomi radii and add to matrix
    n_atoms = len(symbols)
    numbers = [SYMBOLS_TO_NUMBERS[symbol] for symbol in symbols]
    radii = [COV_RADII_PYYKKO[number] for number in numbers]
    radii = np.array(radii) * scale
    radii_matrix = radii[:, None] + radii[None, :]

    # Get distance matrix and check distances below the sum of the two radii
    distance_matrix = scipy.spatial.distance_matrix(coordinates, coordinates)
    connectivity_matrix: Array2DInt = ((distance_matrix - radii_matrix) < 0) - np.eye(
        n_atoms
    )

    return connectivity_matrix


def determine_angles(  # noqa: C901
    coordinates: ArrayLike2D,
    connectivity_matrix: ArrayLike2D,
    mask: Sequence[int],
    linear_threshold: float = 0.9,
    method: str = "atan2",
) -> Array2DFloat:
    """Returns twist angle matrix based on dihedrals.

    Linear bonds are assessed by the dot product between the bond vectors. Method with
    atan2 is numerically more stable.

    Atan2 procedure taken from:
    https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates # noqa: B950
    Cos procedure taken from: https://en.wikipedia.org/wiki/Dihedral_angle


    Args:
        coordinates: Coordinates (Å)
        connectivity_matrix: Connectivity matrix
        mask: Mask for which atoms that will be included in the calculation
        linear_threshold: Threshold for linear bond in terms of dot product.
        method: Method for determining angles 'cos' or 'atan2' (default).

    Returns:
        twist_angles: Twist angle matrix (degrees)
    """
    # Generate networkx graph
    G = nx.from_numpy_array(connectivity_matrix)
    mask: Array1DBool = np.asarray(mask)
    coordinates: Array2DFloat = np.asarray(coordinates)

    # Loop over all bonds
    twist_angles: Array2DFloat = np.zeros_like(connectivity_matrix)
    for edge in G.edges:
        i, j = edge

        # Skip bond if any of the atoms are not in the mask
        if not all(mask[[i, j]]):
            continue

        # Generate neighbor list
        neighbors_i = list(G.neighbors(i))
        neighbors_i.remove(j)
        neighbors_j = list(G.neighbors(j))
        neighbors_j.remove(i)

        # Generate combinations of atoms to determine dihedral angle of bond
        combinations = list(itertools.product(neighbors_i, neighbors_j))

        # Allow perfect conjugation if an atom does not have any neighbors
        if len(combinations) == 0:
            if method == "cos":
                cos = 1
            elif method == "atan2":
                angle = 0
        # Determine the twist angle as the average over all combinations
        else:
            if method == "cos":
                cosines: list[float] = []
            elif method == "atan2":
                angles: list[float] = []
            linear = False

            for k, l in combinations:
                # Construct vectors to define the two planes
                v_ik = coordinates[k] - coordinates[i]
                v_ik /= np.linalg.norm(v_ik)
                v_ij = coordinates[j] - coordinates[i]
                v_ij /= np.linalg.norm(v_ij)
                v_jl = coordinates[l] - coordinates[j]
                v_jl /= np.linalg.norm(v_jl)
                v_ji = -v_ij

                if (
                    abs(np.dot(v_ik, v_ij)) > linear_threshold
                    or abs(np.dot(v_jl, v_ij)) > linear_threshold
                ):
                    linear = True
                    break
                if method == "cos":
                    # Get normal vectors and determine dihedral angle
                    # TODO: Remove type ignores when https://github.com/numpy/numpy/pull/21216 is released  # noqa: B950
                    n_i: Array1DFloat = np.cross(v_ik, v_ij)
                    n_i /= np.linalg.norm(n_i)  # type: ignore
                    n_j: Array1DFloat = np.cross(v_jl, v_ji)
                    n_j /= np.linalg.norm(n_j)

                    cos = np.dot(n_i, n_j)
                    cosines.append(abs(cos))
                elif method == "atan2":
                    # Atan2 version
                    b_1 = -v_ik / np.linalg.norm(v_ik)
                    b_2 = v_ij / np.linalg.norm(v_ij)
                    b_3 = v_jl / np.linalg.norm(v_jl)
                    n_1: Array1DFloat = np.cross(b_1, b_2)
                    n_2: Array1DFloat = np.cross(b_2, b_3)  # type: ignore
                    m_1: Array1DFloat = np.cross(n_1, b_2)
                    x = np.dot(n_1, n_2)
                    y = np.dot(m_1, n_2)
                    angle = np.arctan2(abs(y), abs(x))
                    angles.append(angle)

            if linear is True:
                if method == "cos":
                    cos = 1
                elif method == "atan2":
                    angle = 0
            else:
                if method == "cos":
                    cos = np.mean(cosines)
                elif method == "atan2":
                    angle = np.mean(angles)
        if method == "cos":
            angle = np.rad2deg(np.arccos(cos))
        elif method == "atan2":
            angle = np.rad2deg(angle)
        twist_angles[i, j] = twist_angles[j, i] = angle

    return twist_angles


def process_coordinates(
    symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    remove_isolated: bool = True,
    generate_twist_angles: bool = True,
    radii_scale: float = 1.2,
) -> tuple[InputData, Array1DBool]:
    """Returns atom types and connectivity matrix from coordinates.

    Args:
        symbols: Symbols
        coordinates: Coordinates (Å)
        remove_isolated: Whether to remove non-conjugated atoms
        generate_twist_angles: Whether to generate twist angles
        radii_scale: Scaling of covalent radii when determining connectivity

    Returns:
        atom_types: Atom types
        connectivity_matrix: Connectivity matrix
        mask: Mask for which atoms were kept
    """
    # Generate connectivity matrix and degrees
    connectivity_matrix = connectivity_from_coordinates(
        symbols, coordinates, scale=radii_scale
    )
    degrees = np.sum(connectivity_matrix, axis=0, dtype=int)

    # Generate input data
    input_data, mask = generate_input_data(
        symbols,
        degrees,
        connectivity_matrix,
        coordinates=coordinates,
        remove_isolated=remove_isolated,
        generate_twist_angles=generate_twist_angles,
    )

    return input_data, mask


def _get_isolated_mask(
    connectivity_matrix: ArrayLike2D, mask: ArrayLike1D
) -> Array1DBool:
    connectivity_matrix: Array2DInt = np.asarray(connectivity_matrix)
    mask: Array1DBool = np.asarray(mask)
    to_remove = np.where(connectivity_matrix[:, mask].sum(axis=1) == 0)[0]
    mask[to_remove] = False
    return mask


@requires_dependency([Import(module="rdkit", item="Chem")], globals())
def process_rdkit_mol(
    mol: Chem.Mol,
    coordinates: ArrayLike2D | None = None,
    remove_isolated: bool = True,
    generate_twist_angles: bool = True,
    coord_from_mol: bool = True,
) -> tuple[InputData, Array1DBool]:
    """Returns atom types and connectivity matrix from RDKit Mol.

    Args:
        mol: RDKit Mol object
        coordinates: Coordinates (Å)
        remove_isolated: Whether to remove non-conjugated atoms
        generate_twist_angles: Whether to generate twist angles
        coord_from_mol: Whether to try to generate coordinates from Mol object

    Returns:
        input_data: Input data object
        mask: Mask of which atoms were kept
    """
    # Get connectivity matrix
    connectivity_matrix = Chem.GetAdjacencyMatrix(mol)

    # Get atom types based on connectivity number
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    degrees = [atom.GetTotalDegree() for atom in mol.GetAtoms()]

    # Try to generate coordinates from Mol object
    if coordinates is None and coord_from_mol is True:
        if mol.GetNumConformers() > 0:
            coordinates = mol.GetConformer().GetPositions()

    # Generate input data object
    input_data, mask = generate_input_data(
        symbols,
        degrees,
        connectivity_matrix,
        coordinates=coordinates,
        remove_isolated=remove_isolated,
        generate_twist_angles=generate_twist_angles,
    )

    return input_data, mask


def generate_input_data(
    symbols: Sequence[str],
    degrees: Iterable[int],
    connectivity_matrix: ArrayLike2D,
    coordinates: ArrayLike2D | None = None,
    remove_isolated: bool = True,
    generate_twist_angles: bool = True,
) -> tuple[InputData, Array1DBool]:
    """Generate input data for Hückel and PPP calculation.

    Args:
        symbols: Symbols
        degrees: Total connectivity degrees
        connectivity_matrix: Connectivity matrix
        coordinates: Coordinates (Å)
        remove_isolated: Whether to remove non-conjugated atoms
        generate_twist_angles: Whether to generate twist angles

    Returns:
        input_data: Input data object
        mask: Mask of which atoms were kept
    """
    n_atoms = len(symbols)
    connectivity_matrix: Array2DInt = np.asarray(connectivity_matrix)

    atom_types, to_remove = assign_atom_types(symbols, degrees)

    # Set up mask to remove atoms without atom type and which are not conjugated
    mask: Array1DBool = np.ones(n_atoms, dtype=bool)
    mask[to_remove] = False
    if remove_isolated:
        mask = _get_isolated_mask(connectivity_matrix, mask)

    # Generate twist angles if coordinates are given and prune out masked atoms
    if coordinates is not None:
        coordinates = cast(Array2DFloat, coordinates)
        coordinates = np.asarray(coordinates)
        if generate_twist_angles is True:
            twist_angles = determine_angles(coordinates, connectivity_matrix, mask)
            twist_angles = twist_angles[mask, :][:, mask]
        coordinates = coordinates[mask]
    else:
        twist_angles = None

    connectivity_matrix = connectivity_matrix[mask, :][:, mask]
    atom_types = atom_types[mask]

    # Generate input data object
    input_data = InputData(
        atom_types=atom_types,
        coordinates=coordinates,
        connectivity_matrix=connectivity_matrix,
        twist_angles=twist_angles,
    )

    return input_data, mask
