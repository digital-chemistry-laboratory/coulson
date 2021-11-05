"""Creating Hückel matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from coulson.parameters import beta_from_r, N_ELECTRONS, PARAMETER_SETS


@dataclass
class InputData:
    """Class for handling input to Hückel and PPP calculations."""

    atom_types: Sequence[str]
    coordinates: Sequence[Sequence[float]]
    connectivity_matrix: Sequence[Sequence[int]]
    twist_angles: Sequence[Sequence[float]] = None
    sigma_charges: Sequence = None


def prepare_huckel_matrix(
    atom_types: Sequence[str],
    connectivity_matrix: Sequence[Sequence[int]],
    parametrization: str = "van-catledge",
    var_cc: bool = False,
    var_cc_method: str = "hlhs",
    distance_matrix: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[Sequence[Sequence[float]], Sequence[int]]:
    """Prepare Hückel matrix from atom types and connectivity.

    Args:
        atom_types: Atom types
        connectivity_matrix: Connectivity matrix
        parametrization: Parametrization: 'hess-schaad', 'streitwieser' or
            'van-catledge' (default)
        var_cc: Whether to adjust C-C β values by bond length.
        var_cc_method: Method for variable β: 'hlhs' (default) or 'hssh'
        distance_matrix: Distance matrix (Å)

    Returns:
        huckel_matrix: Hückel matrix
        electrons: Number of electrons contributed by each atom
    """
    # Build huckel matrix
    huckel_matrix = np.array(connectivity_matrix, dtype=float)
    parameters = PARAMETER_SETS[parametrization]

    # Set k_xy values
    for i, j in zip(*np.nonzero(connectivity_matrix)):
        # Set variable betas from distance matrix
        if var_cc is True and (atom_types[i], atom_types[j]) == ("C", "C"):
            k_xy = beta_from_r(distance_matrix[i, j], method=var_cc_method)
        else:
            k_xy = parameters.get_k_xy(atom_types[i], atom_types[j])
        huckel_matrix[i, j] = k_xy

    # Set h_x values
    h_x_values = [parameters.get_h_x(atom_type) for atom_type in atom_types]
    huckel_matrix[np.diag_indices_from(huckel_matrix)] = h_x_values

    # Calculate number of electrons
    electrons = [N_ELECTRONS[atom_type] for atom_type in atom_types]

    return huckel_matrix, electrons
