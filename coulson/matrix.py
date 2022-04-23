"""Creating Hückel matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from coulson.parameters import beta_from_r, N_ELECTRONS, PARAMETER_SETS
from coulson.typing import Array1DFloat, Array2DFloat, ArrayLike2D


@dataclass
class InputData:
    # TODO rename to something better
    """Class for handling input to Hückel and PPP calculations."""

    atom_types: Sequence[str]
    coordinates: Array2DFloat
    connectivity_matrix: Array2DFloat
    twist_angles: Array2DFloat | None = None
    sigma_charges: Array1DFloat | None = None


def prepare_huckel_matrix(
    atom_types: Sequence[str],
    connectivity_matrix: ArrayLike2D,
    parametrization: str = "van-catledge",
    var_cc: bool = False,
    var_cc_method: str = "hlhs",
    distance_matrix: ArrayLike2D | None = None,
) -> Tuple[Array2DFloat, list[int]]:
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

    Raises:
        ValueError: If var_cc requested but distance_matrix not given.
    """
    # Build huckel matrix
    huckel_matrix: Array2DFloat = np.asarray(connectivity_matrix, dtype=float)
    parameters = PARAMETER_SETS[parametrization]

    # Set k_xy values
    for i, j in zip(*np.nonzero(connectivity_matrix)):
        # Set variable betas from distance matrix
        if var_cc is True and (atom_types[i], atom_types[j]) == ("C", "C"):
            if distance_matrix is not None:
                distance_matrix = np.asarray(distance_matrix)
            else:
                raise ValueError("Distance matrix needed.")
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
