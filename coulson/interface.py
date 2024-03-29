"""Interfaces to other programs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import functools
import itertools
import typing
from typing import Any, cast
import warnings

import networkx as nx
import numpy as np
import scipy.spatial

from coulson.data import COV_RADII_PYYKKO, SYMBOLS_TO_NUMBERS
from coulson.graph_utils import order_cycle
from coulson.huckel import InputData
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
    from bcwizard.calc.biotsavart import BiotSavart  # pragma: no cover
    from bcwizard.dummyman import gen_dummies  # pragma: no cover
    from bcwizard.mol import Atom  # pragma: no cover
    from bcwizard.mol import Molecule  # pragma: no cover
    import pyscf  # pragma: no cover
    from rdkit import Chem  # pragma: no cover
    from rdkit.Chem import AllChem, rdCoordGen, rdDetermineBonds

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
                cos = 1.0
            elif method == "atan2":
                angle = 0.0
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
                elif method == "cos":
                    # Get normal vectors and determine dihedral angle
                    n_i: Array1DFloat = np.cross(v_ik, v_ij)
                    n_i /= np.linalg.norm(n_i)
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
                    n_2: Array1DFloat = np.cross(b_2, b_3)
                    m_1: Array1DFloat = np.cross(n_1, b_2)
                    x = np.dot(n_1, n_2)
                    y = np.dot(m_1, n_2)
                    angle = np.arctan2(abs(y), abs(x))
                    angles.append(angle)

            if linear is True:
                if method == "cos":
                    cos = 1.0
                elif method == "atan2":
                    angle = 0.0
            else:
                if method == "cos":
                    cos = np.mean(np.array(cosines))
                elif method == "atan2":
                    angle = np.mean(np.array(angles))
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


@requires_dependency(
    [
        Import(module="rdkit", item="Chem"),
        Import(module="rdkit.Chem", item="rdDetermineBonds"),
    ],
    globals(),
)
def mol_from_xyz(filename: str, charge: int = 0) -> Chem.Mol:
    """Generate RDKit Mol object from xyz file.

    Args:
        filename: XYZ filename
        charge: Molecular charge

    Returns:
        mol: RDKit Mol object
    """
    mol = Chem.MolFromXYZFile(filename)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    return mol


@requires_dependency(
    [
        Import(module="rdkit", item="Chem"),
        Import(module="rdkit.Chem", item="AllChem"),
        Import(module="rdkit.Chem", item="rdCoordGen"),
    ],
    globals(),
)
def gen_coords_for_mol(
    mol: Chem.Mol, bond_length: float = 1.4, coordgen: bool = True
) -> Array2DFloat:
    """Generates 2D coordinates for RDKit Mol and returns them.

    Args:
        mol: RDKit Mol object. Changed in the process
        bond_length: Bond length
        coordgen: Whether to use CoordGen or RDKit's default algorithm.

    Returns:
        coordinates: 2D coordinates (Å)

    Warns:
        If Mol has coordinates already
    """
    n_conformers = mol.GetNumConformers()
    if n_conformers > 0:
        warnings.warn(
            f"Mol already has coordinates: {n_conformers} conformers", stacklevel=2
        )
    if coordgen is True:
        ps = rdCoordGen.CoordGenParams()
        ps.minimizerPrecision = ps.sketcherBestPrecision
        rdCoordGen.AddCoords(mol, ps)
        tm = np.zeros((4, 4))
        for i in range(3):
            tm[i, i] = bond_length
        tm[3, 3] = 1.0
        AllChem.TransformMol(mol, tm)
    else:
        AllChem.Compute2DCoords(mol, bondLength=bond_length)
    coordinates = mol.GetConformer().GetPositions()

    return coordinates


@requires_dependency([Import(module="rdkit", item="Chem")], globals())
def cycles_from_mol(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """Returns cycles from RDKit Mol object.

    Args:
        mol: RDKit Mol object

    Returns:
        cycles: Cycles
    """
    ri = mol.GetRingInfo()
    G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(mol))
    cycles = [order_cycle(cycle, G) for cycle in ri.AtomRings()]

    return cycles


@requires_dependency([Import(module="rdkit", item="Chem")], globals())
def mask_mol(mol: Chem.Mol, mask: ArrayLike1D) -> Chem.Mol:
    """Returns RDKit Mol where atom with False indices in mask have been removed.

    Args:
        mol: RDKit Mol object
        mask: Boolean mask

    Returns:
        masked_mol: Masked Mol object
    """
    mask: Array1DBool = np.asarray(mask)
    rw_mol = Chem.RWMol(mol)
    for i in reversed(np.where(~mask)[0]):
        rw_mol.RemoveAtom(int(i))
    masked_mol = rw_mol.GetMol()
    Chem.SanitizeMol(
        masked_mol,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
    )
    return masked_mol


@requires_dependency(
    [
        Import(module="rdkit", item="Chem"),
        Import(module="rdkit.Chem", item="AllChem"),
    ],
    globals(),
)
def scale_rdkit_mol(mol: Chem.Mol, bond_length: float = 1.4, n_dec: int = 3) -> None:
    """Scales RDKit mol object to desired bond length.

    Args:
        mol: RDKit Mol object
        bond_length: Desired bond length
        n_dec: Number of decimals for equal bond length

    Raises:
        ValueError: When bond lengths are not the same
    """
    # Check if there is only one bond length
    dm = Chem.Get3DDistanceMatrix(mol)
    distances = [
        dm[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()
    ]
    unique_dists: Array1DFloat = np.unique(np.round(distances, n_dec))
    if len(unique_dists) != 1:
        raise ValueError("More than one unique distance. Cannot scale distances.")

    # Scale bond lengths to desired value
    tm = np.zeros((4, 4))
    scale_factor = bond_length / unique_dists
    np.fill_diagonal(tm, scale_factor)
    tm[3, 3] = 1.0
    AllChem.TransformMol(mol, tm)


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
        else:
            twist_angles = None
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


@requires_dependency(
    [
        Import(module="bcwizard.calc.biotsavart", item="BiotSavart"),
        Import(module="bcwizard.dummyman", item="gen_dummies"),
        Import(module="bcwizard.mol", item="Atom"),
        Import(module="bcwizard.mol", item="Molecule"),
    ],
    globals(),
)
def nics_from_bcwizard(
    symbols: Iterable[str],
    coordinates: Iterable[Iterable[float]],
    bcs: Mapping[tuple[int, int], float],
    z_setoff: float = 1.15,
    probe_coordinates: ArrayLike2D | None = None,
    benzene_current: float = 11.5,
    n_bond_points: int = 100,
) -> Array1DFloat:
    """Calculates NICS values with BC-Wizard.

    Install BC-Wizard from https://gitlab.com/porannegroup/bcwizard. The molecule is
    assumed to be in the xy-plane and the NICS values will be calculated for the z
    direction.

    Args:
        symbols: Atom symbols
        coordinates: Coordinates (Å)
        bcs: Bond currents
        z_setoff: Setoff of NICS probes in the Z direction.
        probe_coordinates: Optional set of probe coordinates. If None, BC-Wizard will
            be used to generate probe coordinates at the ring centers.
        benzene_current: Reference current for benzene (nT)
        n_bond_points: Number of points per bond

    Returns:
        nics: NICS values (ppm)
    """
    atoms = {
        i: Atom(atomID=i, element=symbol, x=x, y=y, z=z)
        for i, (symbol, (x, y, z)) in enumerate(zip(symbols, coordinates))
    }
    molecule = Molecule(atoms)
    bc_graph = nx.from_edgelist(bcs.keys()).to_directed()
    molecule.bcGraph = bc_graph
    attrs = {}
    for i, j in molecule.bcGraph.edges():
        I = bcs.get((i, j))
        if I is None:
            I = bcs.get((j, i))
        else:
            I = -I
        attrs[(i, j)] = {"weight": I}
    nx.set_edge_attributes(molecule.bcGraph, attrs)

    setoff: Array1DFloat = np.array([0, 0, z_setoff])
    if probe_coordinates is None:
        probe_coordinates = gen_dummies(molecule, setoff)
    else:
        probe_coordinates = np.asarray(probe_coordinates)
        probe_coordinates += setoff
    nics_component = "z"

    bs_calculator = BiotSavart(
        molecule, benzene_current * 1e-9, n_points=n_bond_points, unit="au"
    )

    nics = bs_calculator.calc_nics(probe_coordinates, component=nics_component)
    return nics
