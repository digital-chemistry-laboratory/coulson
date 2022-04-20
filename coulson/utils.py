"""Helper functions and classes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, List, NoReturn, Sequence

import networkx as nx
import numpy as np

from coulson.typing import Array1DInt, ArrayLike2D


def get_multiplicity(n_electrons: int) -> int:
    """Set multiplicity based on number of electrons.

    Args:
        n_electrons: Number of electrons

    Returns:
        multiplicity: Multiplicity

    Raises:
        ValueError: If number of electrons and multiplicity does not match
    """
    multiplicity = (n_electrons % 2) + 1

    if (n_electrons % 2) == (multiplicity % 2):
        raise ValueError(
            f"Combination of number of electrons {n_electrons} "
            f"and multiplicity {multiplicity} not possible!"
        )
    return multiplicity


def occupations_from_multiplicity(
    n_electrons: int, n_orbitals: int, multiplicity: int
) -> Array1DInt:
    """Simple occupation calculator which does not take degeneracies into account.

    Args:
        n_electrons: Number of electrons
        n_orbitals: Number of orbitals
        multiplicity: Multiplicity

    Returns:
        occupations: Orbital occupations
    """
    n_singly = multiplicity - 1
    n_doubly = int((n_electrons - n_singly) / 2)
    n_empty = n_orbitals - n_doubly - n_singly
    occupations: Array1DInt = np.array([2] * n_doubly + [1] * n_singly + [0] * n_empty)
    return occupations


def rings_from_connectivity(
    connectivity_matrix: ArrayLike2D,
) -> List[List[int]]:
    """Return rings in graph sorted by length.

    Args:
        connectivity_matrix: Connectivity matrix

    Returns:
        rings: Rings as list of list
    """
    # Create graph
    G = nx.convert_matrix.from_numpy_array(connectivity_matrix)

    # Loop over rings and keep unique ones
    already_seen = set()
    rings = []
    for i in list(nx.simple_cycles(G.to_directed())):
        if len(i) > 2:
            if frozenset(i) not in already_seen:
                rings.append(i)
                already_seen.add(frozenset(i))
    rings.sort(key=len)

    return rings


@dataclass
class Import:
    """Class for handling optional dependency imports."""

    module: str
    item: str | None = None
    alias: str | None = None


def requires_dependency(imports: Sequence, _globals: dict) -> Callable:  # noqa: C901
    """Decorator factory to control optional dependencies.

    Args:
        imports: Imports as Import objects.
        _globals: Global symbol table from calling module.

    Returns:
        decorator (function): Either 'noop_decorator' returns the original
            function or 'error_decorator' which raises an ImportError and lists
            absent dependencies.
    """

    def noop_decorator(function: Callable) -> Callable:
        """Returns function unchanged."""
        return function

    def error_decorator(function: Callable) -> Callable:
        """Raises error."""

        def error(*args, **kwargs) -> NoReturn:
            error_msg = "Install extra requirements to use this function:"
            for e in import_errors:
                error_msg += f" {e.name}"
            raise ImportError(error_msg)

        return error

    import_errors = []
    for imp in imports:
        # Import module
        try:
            module = import_module(imp.module)

            # Try to import item as attribute
            if imp.item is not None:
                try:
                    item = getattr(module, imp.item)
                except AttributeError:
                    item = import_module(f"{imp.module}.{imp.item}")
                name = imp.item
            else:
                item = module
                name = imp.module

            # Convert item name to alias
            if imp.alias is not None:
                name = imp.alias

            _globals[name] = item
        except ImportError as import_error:
            import_errors.append(import_error)

    return error_decorator if len(import_errors) > 0 else noop_decorator
