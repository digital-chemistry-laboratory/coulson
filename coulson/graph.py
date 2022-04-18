"""Graph theoretical theory of aromaticity."""

from __future__ import annotations

import networkx as nx
import numpy as np

from coulson.typing import ArrayLike2D


def calculate_tre(
    huckel_matrix: ArrayLike2D,
    n_electrons: int,
    multiplicity: int = None,
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

    Raises:
        ValueError: If number of electrons and multiplicity does not match
    """
    huckel_matrix = np.asarray(huckel_matrix)
    n_atoms = huckel_matrix.shape[0]

    # Set multiplicity
    if multiplicity is None:
        multiplicity = (n_electrons % 2) + 1

    # Multiplicity check
    if (n_electrons % 2) == (multiplicity % 2):
        raise ValueError(
            f"Combination of number of electrons {n_electrons} "
            f"and multiplicity {multiplicity} not possible!"
        )
    # Set occupations
    # TODO: Might set better occupations for degenerate orbitals
    n_singly = multiplicity - 1
    n_doubly = int((n_electrons - n_singly) / 2)
    n_empty = n_atoms - n_doubly - n_singly
    occupations = np.array([2] * n_doubly + [1] * n_singly + [0] * n_empty)

    # Set up graph and set weights
    G = nx.Graph(huckel_matrix)
    # nx.set_node_attributes(
    #    G, {i: j for i, j in enumerate(np.diag(huckel_matrix))}, "weight"
    # )

    # Calculate charateristic polynomial
    cp = np.poly(huckel_matrix)
    cp_roots = np.sort(np.roots(cp).real)[::-1]
    cp_energy = np.sum(cp_roots * occupations)

    # Calculate matching polynomial
    mp = matching_polynomial(G)
    mp_roots = mp.roots().real[::-1]
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
    mp = np.polynomial.Polynomial(0)
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
            n_neighbors = np.array(
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
            poly = np.polynomial.Polynomial(np.poly(nx.to_numpy_array(w_g))[::-1])
            mp += factor * poly

    return mp
