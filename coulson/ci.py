"""Code related to configuration interaction."""


import itertools
from typing import Sequence, Tuple

import numpy as np


def generate_excitations(  # noqa: C901
    n_occupied: int,
    n_virtual: int,
) -> Sequence[Tuple[Tuple[int, ...], Tuple[int, ...], str]]:
    """Generate excitations with specific truncation.

    Args:
        n_occupied: Number of occupied orbitals
        n_virtual: Number of virtual orbitals

    Returns:
        excitaitons: Sequence of excitations
    """
    excitations = []

    # Generate ground state dummy excitation
    excitations.append(((), (), "g"))

    # Add single excitations
    single_excitations_pre = list(
        itertools.product(
            itertools.combinations(range(n_occupied), 1),
            itertools.combinations(range(n_occupied, n_occupied + n_virtual), 1),
        )
    )
    single_excitations = []
    for occupied, virtual in single_excitations_pre:
        label = "s"
        single_excitations.append((occupied, virtual, label))
    excitations.extend(single_excitations)

    return excitations


def calculate_matrix_element(  # noqa: C901
    excitation_1: Tuple[Tuple[int, ...], Tuple[int, ...], str],
    excitation_2: Tuple[Tuple[int, ...], Tuple[int, ...], str],
    e_0: float,
    fock_matrix_mo: Sequence[Sequence[float]],
    mo_integrals: Sequence[Sequence[Sequence[Sequence[float]]]],
    multiplicity: str = "singlet",
) -> float:
    """Returns matrix element between two state configuration functions (CSFs).

    Uses formulas from 10.1016/0584-8539(72)80159-4.

    Args:
        excitation_1: Information on excitation 1
        excitation_2: Information on excitation 2
        e_0: Ground state energy (a.u.)
        fock_matrix_mo: Fock matrix in molecular orbital basis (a.u.)
        mo_integrals: Electron repulsion integrals in molecular orbital basis (a.u.)
        multiplicity: State multiplicity: 'singlet' (default) or 'triplet'

    Returns:
        matrix_element: Matrix element between CSFs
    """
    # Set up convenient names
    E_0 = e_0
    F = fock_matrix_mo
    mo_int = mo_integrals
    delta = np.equal

    J = mo_integrals.diagonal(axis1=0, axis2=1).diagonal(axis1=0, axis2=1)
    K = mo_integrals.diagonal(axis1=0, axis2=2).diagonal(axis1=0, axis2=1)

    # Sort the excitations according to the formula in the paper.
    sort_order = [
        "g",
        "s",
    ]
    excitation_1, excitation_2 = sorted(
        [excitation_1, excitation_2], key=lambda x: sort_order.index(x[2])
    )
    occupied_1, virtual_1, label_1 = excitation_1
    occupied_2, virtual_2, label_2 = excitation_2

    # Label the indices according to the formula in the paper.
    if label_1 == "s":
        k = occupied_1[0]
        m = virtual_1[0]
    if label_2[0] == "s" and label_1[0] == "s":
        j = occupied_2[0]
        r = virtual_2[0]
    elif label_2[0] == "s" and label_1[0] == "g":
        k = occupied_2[0]
        m = virtual_2[0]

    labels = (label_1, label_2)

    # Loop over all possible matrix elements
    if multiplicity == "singlet":
        if labels == ("g", "g"):
            matrix_element = E_0
        elif labels == ("g", "s"):
            matrix_element = np.sqrt(2) * F[k, m]
        elif labels == ("s", "s"):
            if k == j and m == r:
                matrix_element = E_0 + F[m, m] - F[k, k] + 2 * K[k, m] - J[k, m]
            else:
                matrix_element = (
                    delta(k, j) * F[m, r]
                    - delta(m, r) * F[k, j]
                    + 2 * mo_int[k, m, j, r]
                    - mo_int[k, j, m, r]
                )
    elif multiplicity == "triplet":
        if labels == ("g", "g"):
            matrix_element = E_0
        elif labels == ("g", "s"):
            matrix_element = 0
        elif labels == ("s", "s"):
            if k == j and m == r:
                matrix_element = E_0 + F[m, m] - F[k, k] - J[k, m]
            else:
                matrix_element = (
                    delta(k, j) * F[m, r] - delta(m, r) * F[k, j] - mo_int[k, j, m, r]
                )

    return matrix_element
