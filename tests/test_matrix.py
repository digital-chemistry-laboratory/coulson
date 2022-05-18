"""Tests for Hückel matrix generation."""


from numpy.testing import assert_array_equal
import pytest

from coulson.huckel import prepare_huckel_matrix


@pytest.mark.parametrize(
    "atom_types,connectivity_matrix,huckel_matrix,n_electrons",
    [
        (
            ["C", "C", "C", "C"],
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            [1, 1, 1, 1],
        ),
        (
            ["N1", "C", "C", "C", "C", "C"],
            [
                [0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0],
            ],
            [
                [0.51, 1.02, 0.0, 0.0, 0.0, 1.02],
                [1.02, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [1.02, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            [1, 1, 1, 1, 1, 1],
        ),
    ],
)
def test_prepare_huckel_matrix(
    atom_types, connectivity_matrix, huckel_matrix, n_electrons
):
    """Test preparation of Hückel matrix."""
    ref_huckel_matrix = huckel_matrix
    ref_n_electrons = n_electrons

    huckel_matrix, n_electrons = prepare_huckel_matrix(atom_types, connectivity_matrix)
    assert_array_equal(huckel_matrix, ref_huckel_matrix)

    assert n_electrons == ref_n_electrons
