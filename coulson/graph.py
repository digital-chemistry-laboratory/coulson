"""Graph theoretical theory of aromaticity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import itertools
import math
import typing

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from coulson.typing import (
    Array1DFloat,
    Array1DInt,
    Array2DFloat,
    Array2DInt,
    ArrayLike1D,
    ArrayLike2D,
)
from coulson.utils import Import, occupations_from_multiplicity, requires_dependency

if typing.TYPE_CHECKING:  # pragma: no cover
    from shapely import Polygon  # pragma: no cover


@dataclass
class Circuit:
    """Class for storing circuit data."""

    indices: tuple[int, ...]
    A: Array1DFloat
    I: Array1DFloat
    X: Array1DFloat
    area: float
    ccw: bool


def calculate_bre(
    huckel_matrix: ArrayLike2D,
    occupations: ArrayLike1D,
    indices: Iterable[Iterable[int]],
) -> float:
    """Calculate bond resonance energy according to recipe of Aihara.

    Args:
        huckel_matrix: Hückel matrix
        occupations: Orbital occupations
        indices: Tuples of atom indices

    Returns:
        bre: Bond resonance energy

    Raises:
        ValueError: If matrix elements are zero
    """
    huckel_matrix: Array2DFloat = np.array(huckel_matrix)
    occupations: Array1DFloat = np.asarray(occupations)

    # Check if connected or not
    for i, j in indices:
        if huckel_matrix[i, j] == 0 or huckel_matrix[j, i] == 0:
            raise ValueError(f"Matrix element {i}, {j} is zero.")

    # Calculate reference roots
    ref_roots = np.linalg.eigvalsh(huckel_matrix)[::-1]
    huckel_matrix = huckel_matrix.astype(complex)
    for i, j in indices:
        huckel_matrix[i, j] *= 1j
        huckel_matrix[j, i] *= -1j
    bre_roots = np.linalg.eigvalsh(huckel_matrix)[::-1]
    bre = sum((ref_roots - bre_roots) * occupations)

    return bre


def calculate_tre(
    huckel_matrix: ArrayLike2D, n_electrons: int, multiplicity: int = 1
) -> tuple[float, float]:
    """Calculate topological resonance energy according to the recipe of Aihara.

    See BCSJ 2016, 89 (12), 1425-1454 for further information.

    Args:
        huckel_matrix: Hückel matrix
        n_electrons: Number of electrons
        multiplicity: Multiplicity

    Returns:
        tre: Topological resonance energy
        p_tre: Percentage topological resonance energy
    """
    huckel_matrix: Array2DFloat = np.asarray(huckel_matrix)
    n_orbitals = len(huckel_matrix)
    occupations = occupations_from_multiplicity(n_electrons, n_orbitals, multiplicity)

    # Set up graph and set weights
    G = nx.Graph(huckel_matrix)

    # Calculate charateristic polynomial
    cp_roots = np.linalg.eigvalsh(huckel_matrix)[::-1]
    cp_energy = np.sum(cp_roots * occupations)

    # Calculate matching polynomial
    mp = matching_polynomial(G)
    mp_roots = mp.roots().real[::-1]  # type: ignore
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


@requires_dependency(
    [
        Import(module="shapely.geometry", item="Polygon"),
    ],
    globals(),
)
def calculate_circuits(
    cycles: Iterable[Iterable[int]],
    huckel_matrix: ArrayLike2D,
    shell_energies: Iterable[float],
    shell_occupations: Iterable[float],
    degeneracies: Iterable[int],
    coordinates: ArrayLike2D,
    multiplicity: float = 1,
    ref_area: float = 5.092229374252501,
) -> list[Circuit]:
    """Calculate circuits.

    Args:
        cycles: Cycles
        huckel_matrix: Hückel matrix
        shell_energies: Energies of each shell
        shell_occupations: Occupations of each shell
        degeneracies: Degeneracies of each shell
        coordinates: coordinates (Å)
        multiplicity: Multiplicity
        ref_area: Reference area of benzene ring (Å²)

    Returns:
        circuits: Circuits

    Raises:
        ValueError: If number of unpaired electrons does not match multiplicity.
    """
    # Check if not open-shell
    shell_occupations: Array1DFloat = np.asarray(shell_occupations)
    degeneracies: Array1DInt = np.asarray(degeneracies)
    mask = np.logical_and(
        ~np.isclose(shell_occupations, 2), ~np.isclose(shell_occupations, 0)
    )
    unique: Array1DFloat
    unique, indices, counts = np.unique(
        shell_occupations[mask], return_counts=True, return_index=True
    )
    non_one = unique[unique != 1]
    if len(non_one) > 0:
        raise ValueError(f"Open shell layers with occupations: {non_one}")
    if len(unique) > 0:
        n_unpaired = counts[0] * degeneracies[mask][indices[0]]
        if n_unpaired + 1 != multiplicity:
            raise ValueError(
                f"Number of unpaired electrons does {n_unpaired} not match "
                f"multiplicity {multiplicity}. Open-shell calculation not supported."
            )

    huckel_matrix: Array2DFloat = np.asarray(huckel_matrix)
    coordinates: Array2DFloat = np.asarray(coordinates)

    # Calculate characteristic polynomial
    roots_cp = np.linalg.eigvalsh(huckel_matrix)
    cp = np.polynomial.Polynomial.fromroots(roots_cp)  # type: ignore

    circuits = []
    for indices in cycles:
        indices = tuple(indices)

        # Calculate reference polynomial
        mask = np.ones(len(huckel_matrix), dtype=bool)
        mask[list(indices)] = False
        submatrix = huckel_matrix[mask, :][:, mask]
        if len(submatrix) == 0:
            rp = np.polynomial.Polynomial(1.0)  # type: ignore
            roots_rp = rp.roots()  # type: ignore
        else:
            roots_rp = np.linalg.eigvalsh(submatrix)
            rp = np.polynomial.Polynomial.fromroots(roots_rp)  # type: ignore

        # Calculate A
        A = []
        for energy, occupation, degeneracy in zip(
            shell_energies, shell_occupations, degeneracies
        ):
            if occupation == 0:
                continue
            elif degeneracy == 1:
                f_k = rp(energy) / cp.deriv()(energy)
            else:
                p = energy - roots_rp
                q = energy - roots_cp
                q = q[~np.isclose(roots_cp, energy)]
                dp = len(p)
                dq = len(q)
                max_deg = max(dp, dq) + degeneracy - 1
                p: Array1DFloat = np.pad(p, (0, max_deg - dp))
                q: Array1DFloat = np.pad(q, (0, max_deg - dq))
                f_k = (
                    1
                    / math.factorial(degeneracy - 1)
                    * poly_div_deriv(degeneracy - 1, dp, p, dq, q)
                )
            A.append(occupation * f_k)
        A: Array1DFloat = np.array(A)
        A *= 2

        # Multiply by k_st for each bond
        pairs = nx.utils.pairwise(indices, cyclic=True)
        k_sts = huckel_matrix[tuple(zip(*pairs))]
        A *= np.prod(k_sts)

        # Calculate current intensity and diamagnetic susceptibility
        polygon = Polygon(coordinates[list(indices)])
        I = 4.5 * A * polygon.area / ref_area
        X = 4.5 * A * (polygon.area / ref_area) ** 2

        circuit = Circuit(
            indices=indices,
            A=A,
            I=I,
            X=X,
            area=polygon.area,
            ccw=polygon.exterior.is_ccw,
        )
        circuits.append(circuit)

    return circuits


def minimum_cycle_basis(
    connectivity_matrix: ArrayLike2D,
) -> tuple[tuple[int, ...], ...]:
    """Calculate minimum cycle basis from graph.

    Does not return cycles of length one for graphs with one-node loops.

    Args:
        connectivity_matrix: Connectivity matrix

    Returns:
        cycle_basis: Cycle basis
    """
    connectivity_matrix: Array2DInt = np.asarray(connectivity_matrix)
    g = nx.from_numpy_array(connectivity_matrix)

    cycle_basis = tuple(
        tuple(cycle) for cycle in nx.minimum_cycle_basis(g) if len(cycle) > 1
    )
    return cycle_basis


def is_disconnected(edges: Iterable[Iterable[int]]) -> bool:
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


def bond_analysis(
    circuits: Iterable[Circuit],
) -> tuple[dict[tuple[int, int], float], dict[frozenset[int], float]]:
    """Analyze circuits with respect to bonds.

    Args:
        circuits: Circuits

    Returns:
        bcs, m_bres: Bond currents and magnetic bond resonance energies
    """
    bcs: defaultdict[tuple[int, int], float] = defaultdict(float)
    m_bres: defaultdict[frozenset[int], float] = defaultdict(float)
    for circuit in circuits:
        pairs = nx.utils.pairwise(circuit.indices, cyclic=True)
        for i, j in pairs:
            A = sum(circuit.A)
            I = sum(circuit.I)
            if circuit.ccw is False:
                I = -I
            if i > j:
                i, j = j, i
                I = -I
            bcs[(i, j)] += I
            m_bres[frozenset([i, j])] += A
    bcs = dict(bcs)
    m_bres = dict(m_bres)

    return bcs, m_bres


def global_analysis(circuits: Iterable[Circuit]) -> tuple[float, float]:
    """Analyze circuits with respect to global properties.

    Args:
        circuits: Circuits

    Returns:
        mre: Magnetic resonance energy
        sus: Diamagnetic susceptibility
    """
    mre = sum(itertools.chain.from_iterable(circuit.A for circuit in circuits))
    sus = sum(itertools.chain.from_iterable(circuit.X for circuit in circuits))

    return mre, sus


def calculate_m_sse(
    circuits: Iterable[Circuit],
    connectivity_matrix: ArrayLike2D,
    excluded: Iterable[Iterable[int]],
) -> float:
    """Calculates magnetic superaromatic stabilization energy.

    Args:
        circuits: Circuits
        excluded: Cycles to exclude from cycle basis
        connectivity_matrix: Connectivity matrix

    Returns:
        m_sse: Magnetic superaromatic stabilization energy
    """
    # Calculate total MRE
    mre_tot, _ = global_analysis(circuits)

    # Calculate reduced MRE
    reduced_cycles = cycles_from_connectivity(connectivity_matrix, excluded=excluded)
    reduced_circuits = [
        circuit for circuit in circuits if circuit.indices in reduced_cycles
    ]
    mre_reduced, _ = global_analysis(reduced_circuits)

    m_sse = mre_tot - mre_reduced

    return m_sse


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


def get_simple_cycles(G: nx.Graph) -> tuple[tuple[int, ...], ...]:
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
    simple_cycles = tuple(tuple(cycle) for cycle in simple_cycles)

    return simple_cycles


def cycles_from_connectivity(
    connectivity_matrix: ArrayLike2D,
    excluded: Iterable[Iterable[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Get cycles with potential exclusion of base cycles.

    Args:
        connectivity_matrix: Connectivity matrix
        excluded: Cycles to exclude from cycle basis

    Returns:
        cycles: Cycles

    Raises:
        ValueError: If excluded cycle not in basis
    """
    # Get all simple cycles
    G = nx.convert_matrix.from_numpy_array(connectivity_matrix)
    simple_cycles = get_simple_cycles(G)

    # Loop over rings and keep unique ones

    if excluded is not None:
        excluded = [frozenset(cycle) for cycle in excluded]
        ref_cycles = {frozenset(cycle): cycle for cycle in simple_cycles}
        cycle_basis = []
        excluded_cycles = []
        for cycle in minimum_cycle_basis(connectivity_matrix):
            if frozenset(cycle) in excluded:
                excluded_cycles.append(frozenset(cycle))
            else:
                cycle_basis.append(ref_cycles[frozenset(cycle)])
        if len(excluded_cycles) != len(excluded):
            raise ValueError(
                f"Excluded cycle not in cycle basis: {set(excluded).difference(excluded_cycles)}"
            )

        # Prune out disconnected cycles
        cycles = cycle_edges_from_basis(cycle_basis)
        cycles = tuple(cycle for cycle in cycles if is_disconnected(cycle) is False)
        cycles = tuple(
            ref_cycles.get(frozenset(itertools.chain.from_iterable(cycle)))
            for cycle in cycles
        )
        cycles = tuple(cycle for cycle in cycles if cycle is not None)
    else:
        cycles = simple_cycles

    return cycles
