"""Test utility functions and classes."""
import sys

import numpy as np
import pytest

from coulson.typing import Array2DInt
from coulson.utils import Import, requires_dependency, rings_from_connectivity


def test_requires_depedency_module():
    """Test import of optional dependency."""

    @requires_dependency([Import("rdkit")], globals())
    def f():
        pass

    f()

    assert "rdkit" in sys.modules and "rdkit.Chem" in sys.modules


def test_requires_depedency_item():
    """Test import of optional dependency."""

    @requires_dependency([Import("rdkit", item="Chem")], globals())
    def f():
        pass

    f()

    assert "rdkit" in sys.modules and "rdkit.Chem" in sys.modules


def test_requires_depedency_alias():
    """Test import of optional dependency."""

    @requires_dependency(
        [Import("rdkit", item="Chem", alias="AllChem")],
        globals(),
    )
    def f():
        pass

    f()

    assert "rdkit" in sys.modules and "rdkit.Chem" in sys.modules


def test_requires_depedency_fail():
    """Test failed import of optional dependency."""

    @requires_dependency([Import("does.not", item="exist")], globals())
    def f():
        pass  # pragma: no cover

    with pytest.raises(ImportError):
        f()


def test_requires_depedency_item_fail():
    """Test failed import of optional dependency."""

    @requires_dependency([Import("rdkit", item="doesnotexist")], globals())
    def f():
        pass  # pragma: no cover

    with pytest.raises(ImportError):
        f()


def test_rings_from_connectivity():
    """Test generation of rings from connectivity matrix."""
    connectivity_matrix: Array2DInt = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        ]
    )
    ref_rings = [[0, 9, 4, 3, 2, 1], [4, 9, 8, 7, 6, 5], [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
    rings = rings_from_connectivity(connectivity_matrix)

    assert ref_rings == rings
