"""Graph utility functions."""

from __future__ import annotations

import itertools
import typing
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from coulson.typing import Array2DInt, ArrayLike2D


def cycle_edges_from_basis(
    cycle_basis: Iterable[Iterable[int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Calculates simple cycles from given cycle basis.

    Uses algorithm from J. Chem. Inf. Comput. Sci. 1975, 15 (3), 140-147, 10.1021/ci60003a003.

    Args:
        cycle_basis: Cycles in basis

    Returns:
        simple_cycles: Cycles derived from basis
    """
    cycle_basis = [
        frozenset(frozenset(pair) for pair in nx.utils.pairwise(cycle, cyclic=True))
        for cycle in cycle_basis
    ]
    simple_cycles: set[frozenset[frozenset[int]]] = set()
    all_cycles: set[frozenset[frozenset[int]]] = set()

    for base_cycle in cycle_basis:
        new_simple_cycles: set[frozenset[frozenset[int]]] = set()
        new_all_cycles: set[frozenset[frozenset[int]]] = set()

        # Create combinations with previously created cycles
        for old_cycle in all_cycles:
            new_cycle = old_cycle.symmetric_difference(base_cycle)
            new_all_cycles.add(new_cycle)

            # If there is no intersection, cycle is disconnected and not simple
            if len(old_cycle.intersection(base_cycle)) == 0:
                continue

            # Prune out cycles
            rm_simple_cycles = set()
            for check_cycle in new_simple_cycles:
                if new_cycle < check_cycle:  # Proper subset
                    rm_simple_cycles.add(check_cycle)
                if check_cycle < new_cycle:  # Proper subset
                    rm_simple_cycles.add(new_cycle)
                    break
            new_simple_cycles.add(new_cycle)
            new_simple_cycles.difference_update(rm_simple_cycles)
        simple_cycles.update(new_simple_cycles)
        all_cycles.update(new_all_cycles)
        all_cycles.add(base_cycle)
        simple_cycles.add(base_cycle)
    simple_cycles = tuple(
        tuple(tuple(pair) for pair in cycle) for cycle in simple_cycles
    )
    simple_cycles = typing.cast(tuple[tuple[tuple[int, int], ...], ...], simple_cycles)

    return simple_cycles


def edges_are_disconnected(edges: Iterable[Iterable[int]]) -> bool:
    """Check whether edges form disconnected components.

    Args:
        edges: Edges of graph

    Returns:
        disconnected: Whether components are disconnected.
    """
    # Create connectivity matrix from edges
    G = nx.from_edgelist(edges)
    n_connected_components: int = nx.number_connected_components(G)
    disconnected = n_connected_components > 1

    return disconnected


def edges_to_cycle(edges: Iterable[Iterable[int]]) -> tuple[int, ...]:
    """Converts unordered sequeunce of cycle edges to ordered sequence of nodes.

    Args:
        edges: Unordered sequence of edges

    Returns:
        cycle: Ordered sequence of nodes

    Raises:
        ValueError: If wrong number of cycles found
    """
    # Create connectivity matrix
    G = nx.from_edgelist(edges)
    cycles = get_simple_cycles(G)
    if len(cycles) == 0:
        raise ValueError("No cycle found.")
    elif len(cycles) > 1:
        raise ValueError(f"One cycle expected, {len(cycles)} found.")
    cycle = tuple(cycles[0])

    return cycle


def get_simple_cycles(
    graph: nx.Graph | ArrayLike2D,
    excluded: Iterable[Iterable[int]] | None = None,
) -> list[tuple[int, ...]]:
    """Get simple cycles with potential exclusion of base cycles.

    Args:
        graph: NetworkX graph or connectivity matrix
        excluded: Cycles to exclude from cycle basis

    Returns:
        cycles: Cycles

    Raises:
        ValueError: If excluded cycle not in basis
    """
    if not isinstance(graph, nx.Graph):
        connectivity_matrix: Array2DInt = np.asarray(graph)
        G = nx.from_numpy_array(connectivity_matrix)
    else:
        G = graph

    # For excluded cycles, create simple cycles from reduced set of basis cycles
    if excluded is not None:
        excluded = [frozenset(cycle) for cycle in excluded]
        cycle_basis = []
        excluded_cycles = []
        for cycle in minimum_cycle_basis(connectivity_matrix):
            if frozenset(cycle) in excluded:
                excluded_cycles.append(cycle)
            else:
                cycle_basis.append(cycle)
        if len(excluded_cycles) != len(excluded):
            raise ValueError(
                f"Excluded cycle not in cycle basis: {set(excluded).difference(excluded_cycles)}"
            )

        # Prune out disconnected cycles
        cycle_edges = cycle_edges_from_basis(cycle_basis)
        cycle_edges = [
            edges for edges in cycle_edges if edges_are_disconnected(edges) is False
        ]

        # Convert edges to cycles and order
        cycle_indices = [
            tuple(set(itertools.chain.from_iterable(edges))) for edges in cycle_edges
        ]
        cycles = [order_cycle(indices, G) for indices in cycle_indices]
    else:
        cycles = _simple_cycles(G)

    return cycles


def minimum_cycle_basis(
    graph: nx.Graph | ArrayLike2D, ordered: bool = True
) -> list[tuple[int, ...]]:
    """Calculate minimum cycle basis from graph.

    Does not return cycles of length one for graphs with one-node loops.

    Args:
        graph: NetworkX graph or connectivity matrix
        ordered: Whether to order the cycle indices

    Returns:
        cycle_basis: Cycle basis
    """
    if not isinstance(graph, nx.Graph):
        connectivity_matrix: Array2DInt = np.asarray(graph)
        G = nx.from_numpy_array(connectivity_matrix)
    else:
        G = graph
    cycle_basis = [
        tuple(cycle) for cycle in nx.minimum_cycle_basis(G) if len(cycle) > 1
    ]
    if ordered is True:
        cycle_basis = [order_cycle(cycle, G) for cycle in cycle_basis]

    return cycle_basis


def order_cycle(cycle: Iterable[int], G: nx.Graph) -> tuple[int, ...]:
    """Orders cycle based on reference graph.

    Args:
        cycle: Cycle
        G: Reference graph

    Returns:
        cycle_ordered: Ordered cycle

    Raises:
        ValueError: If no cycle found or too many cycles found.
    """
    indices = list(cycle)
    subgraph = G.subgraph(cycle)
    cycles = [
        simple_cycle
        for simple_cycle in get_simple_cycles(subgraph)
        if len(simple_cycle) == len(indices)
    ]
    if len(cycles) == 0:
        raise ValueError("No cycle found.")
    elif len(cycles) > 1:
        raise ValueError(f"One cycle expected, {len(cycles)} found.")
    cycle_ordered = tuple(cycles[0])

    return cycle_ordered


def _simple_cycles(G: nx.Graph) -> list[tuple[int, ...]]:
    """Get simple cycles excluding those with size below 2.

    Args:
        G: NetworkX graph object

    Returns:
        simple_cycles: Simple cycles
    """
    already_seen = set()
    simple_cycles: list[Sequence[int]] = []
    cycle: Sequence[int]
    for cycle in list(nx.simple_cycles(G.to_directed())):
        if len(cycle) > 2:
            if frozenset(cycle) not in already_seen:
                simple_cycles.append(cycle)
                already_seen.add(frozenset(cycle))
    simple_cycles.sort(key=len)
    simple_cycles = [tuple(cycle) for cycle in simple_cycles]

    return simple_cycles
