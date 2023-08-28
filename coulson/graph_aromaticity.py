"""Graph theoretical theory of aromaticity."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import itertools
import math

import networkx as nx
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from coulson.graph_utils import get_simple_cycles, minimum_cycle_basis, order_cycle
from coulson.typing import (
    Array1DFloat,
    Array1DInt,
    Array2DFloat,
    ArrayLike1D,
    ArrayLike2D,
)
from coulson.utils import occupations_from_multiplicity


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
    n_electrons: int,
    indices: Iterable[Iterable[int]],
    multiplicity: int = 1,
) -> float:
    """Calculate bond resonance energy according to recipe of Aihara.

    Args:
        huckel_matrix: Hückel matrix
        n_electrons: Number of electrons
        indices: Tuples of atom indices
        multiplicity: Multiplicity

    Returns:
        bre: Bond resonance energy

    Raises:
        ValueError: If matrix elements are zero
    """
    huckel_matrix: Array2DFloat = np.array(huckel_matrix)
    n_orbitals = len(huckel_matrix)
    occupations = occupations_from_multiplicity(n_electrons, n_orbitals, multiplicity)

    # Check if connected or not
    for i, j in indices:
        if huckel_matrix[i, j] == 0 or huckel_matrix[j, i] == 0:
            raise ValueError(f"Matrix element {i}, {j} is zero.")

    # Calculate reference roots
    cp_roots: Array1DFloat = np.linalg.eigvalsh(huckel_matrix)[::-1]
    huckel_matrix = huckel_matrix.astype(complex)
    for i, j in indices:
        huckel_matrix[i, j] *= 1j
        huckel_matrix[j, i] *= -1j
    ref_roots: Array1DFloat = np.linalg.eigvalsh(huckel_matrix)[::-1]
    bre: float = np.sum((cp_roots - ref_roots) * occupations)

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


def n_neighbors_edge(G: nx.Graph, edge: tuple[int, int]) -> int:
    """Returns the number of neighbors for the two nodes connected by an edge.

    Args:
        G: Graph
        edge: Edge indices

    Returns:
        n_neighbors: Number of neighbors
    """
    n_neighbors = 0
    for i in edge:
        neighbors = set(G.neighbors(i))
        neighbors.discard(i)
        n_neighbors += len(neighbors) - 1
    return n_neighbors


def matching_polynomial(G: nx.Graph) -> np.polynomial.Polynomial:
    """Calculate the matching polynomial of a graph.

    Uses the algorithm of J. Am. Chem. Soc. 1977, 99 (6), 1692-1704

    Args:
        G: Graph

    Returns:
        mp: Matching polynomial
    """
    # TODO: Create dictionary of fragments to avoid calculating multiple times
    # Instantiate initial list of graphs with a factor of one
    graphs = [(G, 1)]

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
            poly = np.polynomial.Polynomial(np.poly(nx.to_numpy_array(w_g))[::-1])
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
        raise ValueError(
            f"Open shell layers with occupations different from one not supported: {non_one}"
        )
    if len(unique) > 0:
        n_unpaired = counts[0] * degeneracies[mask][indices[0]]
        if n_unpaired + 1 != multiplicity:
            raise ValueError(
                f"Number of unpaired electrons {n_unpaired} does not match "
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
            rp = np.polynomial.Polynomial(1.0)
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
    excluded: Iterable[Iterable[int]],
    connectivity_matrix: ArrayLike2D | None = None,
    coordinates: ArrayLike2D | None = None,
    method: str = "gibbs",
) -> float:
    """Calculates magnetic superaromatic stabilization energy.

    Args:
        circuits: Circuits
        excluded: Cycles to exclude from cycle basis
        connectivity_matrix: Connectivity matrix
        coordinates: Coordinates (Å)
        method: Method: 'gibbs' or 'graphical'

    Returns:
        m_sse: Magnetic superaromatic stabilization energy

    Raises:
        ValueError: When method is not supported or incompatabile with optional keywords
    """
    # Calculate total MRE
    mre_tot, _ = global_analysis(circuits)

    # Calculate the MRE from the reduced set of cycles
    if method == "gibbs":
        if connectivity_matrix is None:
            raise ValueError("Connectivity matrix needed.")
        reduced_cycles = get_simple_cycles(connectivity_matrix, excluded=excluded)
        reduced_circuits = [
            circuit for circuit in circuits if circuit.indices in reduced_cycles
        ]
    elif method == "graphical":
        if coordinates is None:
            raise ValueError("Coordinates needed.")
        else:
            coordinates = np.asarray(coordinates)
        polygons = [Polygon(coordinates[list(circuit.indices)]) for circuit in circuits]
        polygons_ref = [Polygon(coordinates[list(indices)]) for indices in excluded]
        mask = [
            any([polygon_ref.covered_by(polygon) for polygon_ref in polygons_ref])
            for polygon in polygons
        ]
        reduced_circuits = [
            circuit for i, circuit in enumerate(circuits) if mask[i] is False
        ]
    else:
        raise ValueError(
            f"Method not supported: {method}. Choose 'gibbs' or 'graphical'"
        )
    mre_reduced, _ = global_analysis(reduced_circuits)
    m_sse = mre_tot - mre_reduced

    return m_sse


def calculate_sse(  # noqa: C901
    huckel_matrix: ArrayLike2D,
    n_electrons: int,
    coordinates: ArrayLike2D,
    ring: Iterable[int],
    cycle_basis: Iterable[Iterable[int]] | None = None,
    multiplicity: int = 1,
) -> float:
    """Calculates superaromatic stabilization energy for a ring.

    Args:
        huckel_matrix: Hückel matrix
        n_electrons: Electrons per atom
        coordinates: Coordinates (Å)
        ring: Cycle to calculate SSE for
        cycle_basis: Cycle basis. Can be provided to save computational time if many
            cycles are computed
        multiplicity: Multiplicity

    Returns:
        sse: SSE value

    Raises:
        ValueError: If cycle_path not found
    """
    huckel_matrix: Array2DFloat = np.asarray(huckel_matrix)
    coordinates: Array2DFloat = np.array(coordinates)
    ring = list(ring)

    # Create connectivity matrix and graph
    connectivity_matrix = np.zeros_like(huckel_matrix)
    connectivity_matrix[huckel_matrix.nonzero()] = 1
    np.fill_diagonal(connectivity_matrix, 0)
    G = nx.from_numpy_array(connectivity_matrix)

    # Calculate cycle basis if needed
    if cycle_basis is None:
        cycle_basis = minimum_cycle_basis(G)

    # Do connected components analysis and consider only cycles from current component
    # indices = list(nx.node_connected_component(G, ring[0]))
    indices = [
        list(indices)
        for indices in nx.biconnected_components(G)
        if indices.issuperset(ring)
    ][0]
    polygon_cycles = [
        cycle for cycle in cycle_basis if len(set(cycle).intersection(indices)) > 0
    ]

    # Build up maximum polygon and take out the cycle indices
    polygons = []
    for cycle in polygon_cycles:
        indices = list(cycle)
        coords = coordinates[indices]
        polygon = Polygon(coords)
        polygons.append(polygon)

    max_polygon = unary_union(polygons)
    max_cycle_unordered = np.where(
        (coordinates[:, None] == max_polygon.boundary.coords).all(axis=-1).any(axis=-1)
    )[0]
    max_cycle = order_cycle(max_cycle_unordered, G)

    # Find intersection with target cycle and max cycle
    # - If two consecutive atoms on the perimeter, cycle already on the perimeter
    # - Otherwise, build path of cycles to perimeter
    max_pairs = set(
        frozenset(pair) for pair in nx.utils.pairwise(max_cycle, cyclic=True)
    )
    target_pairs = set(frozenset(pair) for pair in nx.utils.pairwise(ring, cyclic=True))
    intersecting_pairs = max_pairs.intersection(target_pairs)

    if len(intersecting_pairs) > 0:
        i_indices = [tuple(list(intersecting_pairs)[0])]
    else:
        # Determine shortest path from target to boundary
        shortest_paths = [
            nx.shortest_path(G, source, target)
            for source, target in itertools.product(ring, max_cycle)
        ]
        shortest_path = sorted(shortest_paths, key=lambda x: len(x))[0][::-1]

        # 4. Shortest path of length 4 or greater -> Special algorithm
        # Determine a cycle that border so both boundary and target along shortest path
        first_cycle = None
        last_cycle = None
        if len(shortest_path) == 2:
            for cycle in polygon_cycles:
                intersection_target = set(ring).intersection(cycle)
                intersection_max = set(max_cycle).intersection(cycle)
                intersection_path = set(shortest_path).intersection(cycle)
                if (
                    len(intersection_target) >= 2
                    and len(intersection_max) >= 2
                    and len(intersection_path) == 2
                ):
                    first_cycle = cycle
                    last_cycle = cycle
                    break
        # Determine first and last cycles from path
        else:
            for cycle in polygon_cycles:
                if len(set(cycle).intersection(shortest_path[:3])) == 3:
                    first_cycle = cycle
                if len(set(cycle).intersection(shortest_path[-3:])) == 3:
                    last_cycle = cycle
                if first_cycle is not None and last_cycle is not None:
                    break
        if first_cycle is None:
            raise ValueError("Could not find first cycle along path.")
        if last_cycle is None:
            raise ValueError("Could not find last cycle along path.")

        # Complete bond path with info from first and last cycle
        bond_path = list(shortest_path)
        pairs = nx.utils.pairwise(first_cycle, cyclic=True)
        for pair in pairs:
            if shortest_path[0] in pair and shortest_path[1] not in pair:
                first_pair = list(pair)
                first_pair.remove(shortest_path[0])
                idx_first = first_pair[0]
        pairs = nx.utils.pairwise(last_cycle, cyclic=True)

        for pair in pairs:
            if shortest_path[-1] in pair and shortest_path[-2] not in pair:
                last_pair = list(pair)
                last_pair.remove(shortest_path[-1])
                idx_last = last_pair[0]
        bond_path.insert(0, idx_first)
        bond_path.append(idx_last)

        # Search over quadruplets of atom indices to find cycle path
        cycles_path = [first_cycle]
        quadruplets = [bond_path[i : i + 4] for i in range(2, len(bond_path) - 4, 2)]
        previous = first_cycle
        for quadruplet in quadruplets:
            cycle = [
                cycle
                for cycle in polygon_cycles
                if len(set(cycle).intersection(quadruplet)) >= 3
                and len(set(cycle).intersection(previous)) >= 2
            ][0]
            previous = quadruplet
            cycles_path.append(cycle)
        if last_cycle not in cycles_path:
            cycles_path.append(last_cycle)

        # Take out indices of the cycles intersections along the path
        previous = None
        intersections = [
            tuple(set(cycle_pair[0]).intersection(cycle_pair[1]))
            for cycle_pair in nx.utils.pairwise(cycles_path)
        ]
        path_indices = [tuple(bond_path[:2])] + intersections + [tuple(bond_path[-2:])]

        # Construct the ordered set of bond indices for calculation of SSE
        cycle_pairs = list(nx.utils.pairwise(cycles_path[0], cyclic=True))
        pair_sets = [frozenset(pair) for pair in cycle_pairs]
        i, j = cycle_pairs[pair_sets.index(frozenset(path_indices[0]))]

        # Set the direction of first bond arbitrarily
        pairs = list(nx.utils.pairwise(path_indices))
        i_indices = [(i, j)]
        previous = j, i
        for cycle, ((i, j), (k, l)) in zip(cycles_path, pairs):
            # Determine orientation of each new bond to be opposite to the previous
            cycle_pairs = list(nx.utils.pairwise(cycle, cyclic=True))
            pair_sets = [frozenset(pair) for pair in cycle_pairs]
            i, j = cycle_pairs[pair_sets.index(frozenset((i, j)))]
            k, l = cycle_pairs[pair_sets.index(frozenset((k, l)))]
            # Check for different cw/ccw orientation of this ring compared to previous
            if previous == (i, j):
                k, l = l, k
            i_indices.append((l, k))
            previous = k, l

    sse = calculate_bre(
        huckel_matrix, n_electrons, indices=i_indices, multiplicity=multiplicity
    )

    return sse
