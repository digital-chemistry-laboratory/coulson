"""Test utility functions and classes."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
import sys

import numpy as np
import pytest

from coulson.graph_utils import get_simple_cycles
from coulson.typing import Array2DInt
from coulson.utils import Import, requires_dependency


def is_cyclic_permutation(a: Sequence[int], b: Sequence[int]) -> bool:
    """Test if two cycles are permutations of each other.

    Taken from https://github.com/networkx/networkx/blob/main/networkx/algorithms/tests/test_cycles.py # noqa: B950

    Args:
        a: First cycle
        b: Second cycle

    Returns:
        True if cycles are permutations of each other
    """
    n = len(a)
    if len(b) != n:
        return False
    l = a + a
    return any(l[i : i + n] == b for i in range(n))


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


def test_get_simple_cycles():
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
    ref_rings = [(0, 9, 4, 3, 2, 1), (4, 9, 8, 7, 6, 5), (0, 9, 8, 7, 6, 5, 4, 3, 2, 1)]
    rings = get_simple_cycles(connectivity_matrix)

    checks = []
    for rings_permuted in itertools.permutations(rings):
        checks.append(map(is_cyclic_permutation, ref_rings, rings_permuted))
    assert any(checks)
