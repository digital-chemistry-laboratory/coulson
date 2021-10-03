"""Code related to electronegativity equilibration."""

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from rdkit import Chem


@dataclass
class Orbital:
    """Orbital object."""

    idx: int
    atom: object
    bond: object
    b_0: float
    b_1: float
    c: float
    en: float


@dataclass
class Atom:
    """Atom object."""

    idx: int
    bonds: List["Bond"] = field(default_factory=list)
    orbitals: List["Orbital"] = field(default_factory=list)


@dataclass
class Bond:
    """Bond object."""

    idx: int
    atoms: List["Atom"] = field(default_factory=list)
    orbitals: List["Orbital"] = field(default_factory=list)


def get_hyb_str(symbol: str, hybridization: "Chem.HybridizationType") -> str:
    """Get hybridization string from symbol and RDKit hybridization type.

    Args:
        symbol: Symbol
        hybridization: RDKit hybridization type

    Returns:
        hyb_str: Hybridization string
    """
    if symbol == "H":
        hyb_str = "s"
    elif symbol in ["F", "Cl", "Br", "I"]:
        hyb_str = "p"
    else:
        hyb_str = HYB_TO_HYB_STR[hybridization]

    return hyb_str


def _get_atoms_bonds(
    mol: "Chem.Mol", sigma_only: bool = True
) -> Tuple[Sequence["Atom"], Sequence["Bond"]]:
    """Get list of atoms and bonds with associated bond orbitals.

    Args:
        mol: RDKit mol object
        sigma_only: Whether to calculate charges based on sigma framework only

    Returns:
        atoms: Atom objects
        bonds: Bond objects
    """
    n_atoms = mol.GetNumAtoms()
    bonds = []
    atoms = [Atom(idx=i) for i in range(n_atoms)]
    for a, bond in enumerate(mol.GetBonds()):
        # Determine atom symbols and indices
        atom_i, atom_j = bond.GetBeginAtom(), bond.GetEndAtom()
        symbol_i, symbol_j = atom_i.GetSymbol(), atom_j.GetSymbol()
        i, j = atom_i.GetIdx(), atom_j.GetIdx()

        # Get bond order
        bo = BOND_TYPE_TO_ORDER[bond.GetBondType()]
        if sigma_only is True and bo > 1:
            bo -= 1
        bo_str = BOND_ORDER_TO_STR[bo]

        # Get hybridization states and degrees
        hyb_i = get_hyb_str(symbol_i, atom_i.GetHybridization())
        hyb_j = get_hyb_str(symbol_j, atom_j.GetHybridization())
        degree_i = atom_i.GetTotalDegree()
        degree_j = atom_j.GetTotalDegree()

        # Get bond types
        bond_type_i = f"{symbol_i}_{hyb_i}_{degree_i}_{bo_str}"
        bond_type_j = f"{symbol_j}_{hyb_j}_{degree_j}_{bo_str}"

        # Setup bond, atom and orbital objects
        bond = Bond(idx=a)
        atom_i = atoms[i]
        atom_j = atoms[j]

        idx = 2 * (a + 1)
        orb_i = Orbital(
            idx=idx - 1,
            atom=atom_i,
            bond=bond,
            b_0=B_0[bond_type_i],
            b_1=B_1[bond_type_i],
            c=C[bond_type_i],
            en=EN[bond_type_i],
        )
        orb_j = Orbital(
            idx=idx,
            atom=atom_j,
            bond=bond,
            b_0=B_0[bond_type_j],
            b_1=B_1[bond_type_j],
            c=C[bond_type_j],
            en=EN[bond_type_j],
        )

        atom_i.bonds.append(bond)
        atom_i.orbitals.append(orb_i)

        atom_j.bonds.append(bond)
        atom_j.orbitals.append(orb_j)

        bond.atoms.extend([atom_i, atom_j])
        bond.orbitals.extend([orb_i, orb_j])

        bonds.append(bond)

    return atoms, bonds


def _get_charge_displacements(
    bonds: Sequence["Bond"], orbital_order: List[List[int]]
) -> Sequence[float]:
    """Returns charge displacement for bonds.

    Args:
        bonds: Bond objects
        orbital_order: Order of bond orbital pairs

    Returns:
        delta_qs: Charge displacements
    """
    # Set up linear equations
    en_diffs = []
    all_delta_qs = []
    for bond in bonds:
        delta_qs = np.zeros(len(orbital_order))
        orb_j, orb_i = bond.orbitals
        atom_j, atom_i = bond.atoms

        delta_q = 2 * (orb_i.c + orb_j.c)
        idx = orbital_order.index([orb_i.idx, orb_j.idx])
        delta_qs[idx] = delta_q

        for bond_i in atom_i.bonds:
            if bond_i == bond:
                continue
            if orb_i.atom == bond_i.atoms[0]:
                orb_k, orb_l = bond_i.orbitals
            else:
                orb_l, orb_k = bond_i.orbitals
            k, l = orb_k.idx, orb_l.idx
            delta_q = np.sign(k - l) * orb_i.b_1
            idx = orbital_order.index(sorted([k, l], reverse=True))
            delta_qs[idx] += delta_q

        for bond_j in atom_j.bonds:
            if bond_j == bond:
                continue
            if orb_j.atom == bond_j.atoms[0]:
                orb_k, orb_l = bond_j.orbitals
            else:
                orb_l, orb_k = bond_j.orbitals
            k, l = orb_k.idx, orb_l.idx
            delta_q = np.sign(k - l) * orb_j.b_1
            idx = orbital_order.index(sorted([k, l], reverse=True))
            delta_qs[idx] -= delta_q

        all_delta_qs.append(delta_qs)
        en_diff = orb_j.en - orb_i.en
        en_diffs.append(en_diff)
    delta_q_matrix = np.vstack(all_delta_qs)

    # Solve equations
    delta_qs = np.linalg.solve(delta_q_matrix, en_diffs)

    return delta_qs


def get_sigma_charges(mol: "Chem.Mol", sigma_only: bool = True) -> Sequence[float]:
    """Get charges in sigma framework from electronengativity equalization.

    Uses the procedure and data from Bergmann and Hinze: 978-3-540-17740-1.

    Args:
        mol: RDKit mol object
        sigma_only: Whether to calculate charges based on sigma framework only.

    Returns:
        atom_charges: Atom charges
    """
    # Get atoms and bonds
    atoms, bonds = _get_atoms_bonds(mol, sigma_only=sigma_only)

    # Establish orbital order for reference
    orbital_order = [
        list(reversed([orb.idx for orb in bond.orbitals])) for bond in bonds
    ]

    # Calculate charge displacements
    delta_qs = _get_charge_displacements(bonds, orbital_order)

    # Calculate atom charges
    atom_charges = []
    for atom in atoms:
        # Sum charge displacements over bonds
        delta_qs_atom = []
        for bond in atom.bonds:
            j, i = [orb.idx for orb in bond.orbitals]
            idx = orbital_order.index([i, j])
            if bond.orbitals[1] in atom.orbitals:
                sign = np.sign(i - j)
            else:
                sign = np.sign(j - i)
            delta_q = delta_qs[idx] * sign
            delta_qs_atom.append(delta_q)
        atom_charges.append(sum(delta_qs_atom))
    atom_charges = np.array(atom_charges)

    return atom_charges


BOND_ORDER_TO_STR = {
    1: "s",
    2: "d",
    3: "t",
}
"""dict: Bond orders as keys and bond strings as values."""


HYB_TO_HYB_STR = {
    Chem.HybridizationType.SP: "di",
    Chem.HybridizationType.SP2: "tr",
    Chem.HybridizationType.SP3: "te",
    Chem.HybridizationType.UNSPECIFIED: "s",
}
"""dict: RDKit hybridization types as keys and hybridization strings as values."""

BOND_TYPE_TO_ORDER = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 2,
}
"""dict: RDKit bond types as keys and bond orders as values."""


B_0 = {
    "H_s_1_s": 20.021,
    "B_tr_3_s": 16.266,
    "C_tr_3_s": 22.400,
    "C_tr_3_d": 19.5355,
    "C_te_4_s": 21.185,
    "C_di_1_s": 26.160,
    "C_di_1_d": 21.527,
    "C_di_2_s": 24.409,
    "C_di_2_d": 20.5405,
    "C_di_2_t": 19.251,
    "N_te_3_s": 26.399,
    "N_tr_3_s": 27.195,
    "N_tr_2_s": 28.419,
    "N_di_1_d": 26.383,
    "N_di_1_t": 24.428,
    "N_di_2_s": 29.804,
    "N_di_2_d": 24.986,
    "O_te_2_s": 33.545,
    "O_tr_1_s": 36.249,
    "O_tr_1_d": 30.786,
    "O_tr_2_s": 35.565,
    "F_p_1_s": 29.595,
    "S_te_2_s": 20.861,
    "S_tr_1_s": 21.775,
    "S_tr_2_s": 21.662,
}
"""dict: b⁰ values from Hinze and Bergmann."""

B_1 = {
    "H_s_1_s": 0.0,
    "B_tr_3_s": 8.717,
    "C_te_4_s": 10.996,
    "C_tr_3_s": 10.984,
    "C_tr_3_d": 11.056,
    "C_di_1_s": 12.024,
    "C_di_1_d": 11.8835,
    "C_di_2_s": 11.020,
    "C_di_2_d": 11.0695,
    "C_di_2_t": 11.086,
    "N_te_3_s": 12.826,
    "N_tr_3_s": 12.467,
    "N_tr_2_s": 12.890,
    "N_di_1_d": 13.2105,
    "N_di_1_t": 13.1977,
    "N_di_2_s": 12.331,
    "N_di_2_d": 12.713,
    "O_te_2_s": 15.393,
    "O_tr_1_s": 15.956,
    "O_tr_1_d": 15.6395,
    "O_tr_2_s": 15.132,
    "F_p_1_s": 0.0,
    "S_te_2_s": 11.347,
    "S_tr_1_s": 11.026,
    "S_tr_2_s": 11.601,
}
"""dict: b¹ values from Hinze and Bergmann."""

C = {
    "H_s_1_s": 6.422,
    "B_tr_3_s": 4.978,
    "C_te_4_s": 6.565,
    "C_tr_3_s": 6.766,
    "C_tr_3_d": 6.1325,
    "C_di_1_s": 7.234,
    "C_di_1_d": 6.404,
    "C_di_2_s": 6.972,
    "C_di_2_d": 6.2225,
    "C_di_2_t": 5.9727,
    "N_te_3_s": 7.456,
    "N_tr_3_s": 7.466,
    "N_tr_2_s": 7.800,
    "N_di_1_d": 7.326,
    "N_di_1_t": 6.9923,
    "N_di_2_s": 7.694,
    "N_di_2_d": 6.8705,
    "O_te_2_s": 9.150,
    "O_tr_1_s": 9.595,
    "O_tr_1_d": 8.6085,
    "O_tr_2_s": 9.419,
    "F_p_1_s": 8.726,
    "S_te_2_s": 5.361,
    "S_tr_1_s": 5.445,
    "S_tr_2_s": 5.389,
}
"""dict: c values from Hinze and Bergmann."""

EN = {
    "H_s_1_s": 7.18,
    "B_tr_3_s": 6.31,
    "C_te_4_s": 8.06,
    "C_tr_3_s": 8.87,
    "C_tr_3_d": 7.27,
    "C_di_1_s": 11.69,
    "C_di_1_d": 8.72,
    "C_di_2_s": 10.47,
    "C_di_2_d": 8.1,
    "C_di_2_t": 7.31,
    "N_te_3_s": 11.49,
    "N_tr_3_s": 12.26,
    "N_tr_2_s": 12.82,
    "N_di_1_d": 11.73,
    "N_di_1_t": 10.4433,
    "N_di_2_s": 14.42,
    "N_di_2_d": 11.245,
    "O_te_2_s": 15.25,
    "O_tr_1_s": 17.06,
    "O_tr_1_d": 13.57,
    "O_tr_2_s": 16.73,
    "F_p_1_s": 12.14,
    "S_te_2_s": 10.14,
    "S_tr_1_s": 10.89,
    "S_tr_2_s": 10.88,
}
"""dict: Mulliken orbital electronegativity values from Hinze and Bergmann."""
