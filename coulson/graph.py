"""Graph theoretical theory of aromaticity."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import itertools

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from coulson.typing import (
    Array1DFloat,
    Array1DInt,
    Array2DFloat,
    ArrayLike1D,
    ArrayLike2D,
)
from coulson.utils import get_multiplicity, occupations_from_multiplicity


def calculate_bre(
    huckel_matrix: ArrayLike2D,
    n_electrons: int,
    indices: Iterable[tuple[int, int]],
    multiplicity: int | None = None,
) -> float:
    """Calculate bond resonance energy according to recipe of Aihara.

    Args:
        huckel_matrix: Hückel matrix
        n_electrons: Number of electrons
        indices: Tuples of atom indices (1-indexed)
        multiplicity: Multiplicity

    Returns:
        bre: Bond resonance energy
    """
    huckel_matrix: Array2DFloat = np.array(huckel_matrix)
    n_atoms = huckel_matrix.shape[0]

    # Set multiplicity
    if multiplicity is None:
        multiplicity = get_multiplicity(n_electrons)

    # Set occupations
    occupations = occupations_from_multiplicity(n_electrons, n_atoms, multiplicity)

    # Calculate reference roots
    ref_roots, _ = np.linalg.eigh(huckel_matrix)
    huckel_matrix = huckel_matrix.astype(complex)
    for i, j in indices:
        huckel_matrix[i - 1, j - 1] *= 1j
        huckel_matrix[j - 1, i - 1] *= -1j
    bre_roots, _ = np.linalg.eigh(huckel_matrix)
    bre = sum((bre_roots - ref_roots) * occupations)

    return bre


def calculate_tre(
    huckel_matrix: ArrayLike2D,
    n_electrons: int,
    multiplicity: int | None = None,
) -> tuple[float, float]:
    """Calculate topological resonance energy according to the recipe of Aihara.

    See BCSJ 2016, 89 (12), 1425-1454 for further information.

    Args:
        huckel_matrix: Hückel matrix
        n_electrons: Number of electrons in the molecule
        multiplicity: Multiplicity

    Returns:
        tre: Topological resonance energy
        p_tre: Percentage topological resonance energy
    """
    huckel_matrix: Array2DFloat = np.asarray(huckel_matrix)
    n_atoms = huckel_matrix.shape[0]

    # Set multiplicity
    if multiplicity is None:
        multiplicity = get_multiplicity(n_electrons)

    # Set occupations
    occupations = occupations_from_multiplicity(n_electrons, n_atoms, multiplicity)

    # Set up graph and set weights
    G = nx.Graph(huckel_matrix)

    # Calculate charateristic polynomial
    cp = np.poly(huckel_matrix)
    cp_roots = np.sort(np.roots(cp))[::-1]
    cp_energy = np.sum(cp_roots * occupations)

    # Calculate matching polynomial
    mp = matching_polynomial(G)
    mp_roots = mp.roots()[::-1]  # type: ignore
    mp_energy = np.sum(mp_roots * occupations)

    # Calculate topological resonance energy
    tre = cp_energy - mp_energy
    p_tre = tre / mp_energy * 100

    return tre, p_tre


def n_neighbors_edge(g: nx.Graph, edge: tuple[int, int]) -> int:
    """Returns the number of neighbors for the two nodes connected by an edge.

    Args:
        g: Graph
        edge: Edge indices

    Returns:
        n_neighbors: Number of neighbors
    """
    n_neighbors = 0
    for i in edge:
        neighbors = set(g.neighbors(i))
        neighbors.discard(i)
        n_neighbors += len(neighbors) - 1
    return n_neighbors


def matching_polynomial(g: nx.Graph) -> np.polynomial.Polynomial:
    """Calculate the matching polynomial of a graph.

    Uses the algorithm of J. Am. Chem. Soc. 1977, 99 (6), 1692-1704

    Args:
        g: Graph

    Returns:
        mp: Matching polynomial
    """
    # TODO: Create dictionary of fragments to avoid calculating multiple times
    # Instantiate initial list of graphs with a factor of one
    graphs = [(g, 1)]

    # Cut graph into linear pieces and evaluate their matching polynomials
    mp = np.polynomial.Polynomial(0)  # type: ignore
    while graphs:
        # Create a working graph
        w_g, factor = graphs.pop()

        cycles = [i for i in nx.cycle_basis(w_g) if len(i) > 1]
        if len(cycles) > 0:
            # Cut graph at non-bridge sites
            bridges = list(nx.bridges(w_g))
            edge_candidates = list(set(w_g.edges()).difference(bridges))
            edge_candidates = [(i, j) for i, j in edge_candidates if i != j]

            # Pick edge that does not have vertices with weights
            weights = [
                (w_g.has_edge(i, i), w_g.has_edge(j, j)) for i, j in edge_candidates
            ]
            n_weights = np.sum(weights, axis=1)
            n_neighbors: Array1DInt = np.array(
                [n_neighbors_edge(w_g, edge) for edge in edge_candidates]
            )
            idx = np.lexsort((n_neighbors, n_weights), axis=0)[0]

            # Remove vertex weights if no candidates found
            if np.all(n_weights != 0):
                # Find vertex with highest connectivity and lowest number of vertex weights
                edge = edge_candidates[idx]
                vertex = edge[np.argsort(weights[idx])[-1]]
                vertex_weight = w_g.get_edge_data(vertex, vertex)["weight"]

                # Cut graph
                n_g_1 = w_g.copy()
                n_g_1.remove_edge(vertex, vertex)
                n_g_2 = w_g.copy()
                n_g_2.remove_node(vertex)

                graphs.append((n_g_1, 1 * factor))
                graphs.append((n_g_2, -1 * vertex_weight * factor))
                continue

            # Cut graph
            i, j = edge_candidates[idx]
            n_g_1 = w_g.copy()
            n_g_2 = w_g.copy()
            e_weight = w_g.get_edge_data(i, j)["weight"]
            n_g_1.remove_edge(i, j)
            n_g_2.remove_node(i)
            n_g_2.remove_node(j)
            graphs.append((n_g_1, 1 * factor))
            graphs.append((n_g_2, -1 * factor * e_weight**2))
        else:
            # Create matching polynomial directly from array for acyclic structure
            poly = np.polynomial.Polynomial(np.poly(nx.to_numpy_array(w_g))[::-1])  # type: ignore
            mp += factor * poly

    return mp


def poly_div_deriv(  # noqa: C901
    power: int, dp: int, p: ArrayLike1D, dq: int, q: ArrayLike1D
) -> float:
    """Evaluate derivative of polynomial division.

    Following the procedure of Chemistry 2021, 3 (4), 1138-1156, 10.3390/chemistry3040083.

    Args:
        power: Number of times to differentiate
        dp: Degree of polynomial p
        p: Polynomial coefficients of p
        dq: Degree of polynomial q
        q: Polynomial coefficients of q

    Returns:
        ans: Derivative
    """
    p: Array1DFloat = np.asarray(p)
    q: Array1DFloat = np.asarray(q)
    r: Array1DFloat = np.zeros_like(p)
    if power == 0:
        if dp < dq:
            limit = dq
        else:
            limit = dp
        ans = 1.0
        for i in range(limit):
            if i < dp:
                top = p[i]
            else:
                top = 1
            if i < dq:
                bottom = q[i]
            else:
                bottom = 1
            ans *= top / bottom
        return ans
    ans = 0.0
    if dp > 0:
        if dp == 1:
            r[0] = 1
            ans += poly_div_deriv(power - 1, dp - 1, r, dq, q)
        else:
            for i in range(dp):
                pos = 0
                for j in range(dp):
                    if i != j:
                        r[pos] = p[j]
                        pos += 1
                ans += poly_div_deriv(power - 1, dp - 1, r, dq, q)
    for i in range(dq):
        r[i] = q[i]
    for i in range(dq):
        r[dq] = q[i]
        ans -= poly_div_deriv(power - 1, dp, p, dq + 1, r)
    return ans


def cycles_from_basis(
    cycle_basis: Iterable[Iterable[tuple[int, ...]]],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Calculates simple cycles from given cycle basis.

    Uses algorithm from J. Chem. Inf. Comput. Sci. 1975, 15 (3), 140-147, 10.1021/ci60003a003.

    Args:
        cycle_basis: Cycles in basis

    Returns:
        simple_cycles: Cycles derived from basis
    """
    cycle_basis = [
        frozenset(frozenset(pair) for pair in cycle) for cycle in cycle_basis
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

    return simple_cycles


def rings_from_connectivity(
    connectivity_matrix: ArrayLike2D,
) -> tuple[tuple[int, ...], ...]:
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
    cycle: list[int]
    for cycle in list(nx.simple_cycles(G.to_directed())):
        if len(cycle) > 2:
            if frozenset(cycle) not in already_seen:
                rings.append(cycle)
                already_seen.add(frozenset(cycle))
    rings.sort(key=len)
    rings = tuple(tuple(ring) for ring in rings)

    return rings


def is_disconnected(edges: Sequence[tuple[int, int]]) -> bool:
    """Check whether edges form disconnected components.

    Args:
        edges: Edges of graph

    Returns:
        disconnected: Whether components are disconnected.
    """
    # Create connectivity matrix from edges
    size = max(itertools.chain.from_iterable(edges)) + 1
    cm = np.zeros((size, size))
    for i, j in edges:
        cm[i, j] = cm[j, i] = 1
    mask = cm.any(axis=0)
    cm = cm[:, mask][mask, :]

    # Calculate number of connected components
    n: int
    n, _ = connected_components(csgraph=csr_matrix(cm), directed=False)
    disconnected = n > 1

    return disconnected
