"""Drawing tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import io
import typing
from typing import BinaryIO, Sequence

import networkx as nx
import numpy as np

from coulson.graph_aromaticity import Circuit
from coulson.interface import gen_coords_for_mol
from coulson.typing import (
    Array1DAny,
    Array1DFloat,
    Array1DInt,
    Array2DFloat,
    ArrayLike1D,
    ArrayLike2D,
)
from coulson.utils import Import, requires_dependency

if typing.TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt  # pragma: no cover
    from rdkit import Chem  # pragma: no cover
    from rdkit.Chem.Draw import SimilarityMaps  # pragma: no cover
    from rdkit.Geometry import Point2D, Point3D


@requires_dependency(
    [
        Import(module="matplotlib", item="pyplot", alias="plt"),
    ],
    globals(),
)
def draw_orbital_energies(  # noqa: C901
    energies: ArrayLike1D,
    occupations: Iterable[int] | None = None,
    fig_size: tuple[float, float] = (8, 12),
) -> tuple[plt.Figure, plt.Axes]:
    """Draw orbital energy diagram.

    Args:
        energies: Orbital energies
        occupations: Orbital occupations
        fig_size: Figure size (inch)

    Returns:
        fig, ax: Matplotlib Figure and Axes objects.

    Raises:
        ValueError: When maximum degeneracy exceeds 2.
    """
    # Set up plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xticklabels([])
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.invert_yaxis()
    ax.set_ylabel(r"$E - \alpha$ ($\beta$)")

    # Calculate offsets
    energies = np.round(energies, 3)
    energies_unique: Array1DFloat
    degeneracies: Array1DInt
    energies_unique, degeneracies = np.unique(energies, return_counts=True)

    line_offsets: list[int] = []
    for degeneracy in degeneracies:
        if degeneracy == 1:
            line_offsets.extend([0])
        elif degeneracy == 2:
            line_offsets.extend([-1, 1])
        else:
            raise ValueError(
                f"Given degeneracy {degeneracy} > maximum supported degeneracy of 2."
            )

    # Draw orbital energy levels
    ax.eventplot(
        energies.reshape(-1, 1), lineoffsets=line_offsets, orientation="vertical"
    )

    # Draw electron arrows:
    if occupations is not None:
        for occupation, energy, offset in zip(occupations, energies, line_offsets):
            if occupation in [0.5, 1.0]:
                scale = occupation
                ax.arrow(
                    offset,
                    energy + 0.1 * scale,
                    0.0,
                    -0.2 * scale,
                    width=0.01,
                    fc="k",
                    length_includes_head=True,
                )
            elif occupation in [1.5, 2.0]:
                scale = occupation - 1
                ax.arrow(
                    offset - 0.1,
                    energy + 0.1,
                    0.0,
                    -0.2,
                    width=0.01,
                    fc="k",
                    length_includes_head=True,
                )
                ax.arrow(
                    offset + 0.1,
                    energy - 0.1 * scale,
                    0.0,
                    0.2 * scale,
                    width=0.01,
                    fc="k",
                    length_includes_head=True,
                )
            elif occupation == 0:
                pass
            else:
                raise ValueError(
                    f"Given occupation {occupation} not supported. "
                    "Use 0, 0.5, 1, 1.5, or 2"
                )
            ax.annotate(occupation, (offset - 0.7, energy))

    # Make energy labels
    for energy in energies_unique:
        ax.annotate(f"{energy:.03f}", (max(line_offsets) + 0.7, energy))
    plt.close()

    return fig, ax


@requires_dependency(
    [
        Import(module="rdkit", item="Chem"),
        Import(module="rdkit.Chem.Draw", item="SimilarityMaps"),
        Import(module="rdkit.Geometry", item="Point2D"),
        Import(module="rdkit.Geometry", item="Point3D"),
    ],
    globals(),
)
def draw_mol(  # noqa: C901
    mol: "Chem.Mol",
    mol_label: str | None = None,
    properties: Iterable[float] | None = None,
    atom_numbers: bool = False,
    atom_labels: Iterable[float] | None = None,
    bond_numbers: bool = False,
    bond_labels: Iterable[float] | None = None,
    rings: Iterable[Iterable[int]] | None = None,
    ring_labels: Iterable[float] | None = None,
    circle_values: Iterable[float] | None = None,
    circle_radius: float = 0.3,
    circle_cmap: str = "RdBu",
    circle_cmax: float | None = 0.25,
    highlighted_atoms: Iterable[int] | None = None,
    highlighted_bonds: Iterable[int] | None = None,
    size: tuple[int, int] = (400, 400),
    n_decimals: int = 3,
    img_format: str = "svg",
) -> str | bytes:
    """Draw molecule with RDKit.

    Args:
        mol: Molecule
        mol_label: Molecule label
        properties: Properties for contour plot
        atom_numbers: Whether to print atom numbers
        atom_labels: Labels to print for atoms
        bond_numbers: Whether to print bond numbers
        bond_labels: Labels to print for bonds
        rings: Ring indices
        ring_labels: Ring labels
        circle_values: Value to plot as colored circles in rings
        circle_radius: Circle radius
        circle_cmap: Color map for circles
        circle_cmax: Max value for color mapping for circles
        highlighted_atoms: Indices for atoms to highlight
        highlighted_bonds: Indices for bonds to highlight
        size: Size of RDKit draw object
        n_decimals: Number of decimals for properties and labels
        img_format: Image format: 'png' or 'svg'

    Returns:
        drawing_text: Drawing text as string or bytes. Can be used with IPython's SVG or
            PNG

    Raises:
        ValueError: When image format not supported.
    """
    # Set mutable default arguments
    if highlighted_atoms is None:
        highlighted_atoms = []
    if highlighted_bonds is None:
        highlighted_bonds = []

    # Copy mol object to prevent editing it in place
    mol = Chem.Mol(mol)

    # Generate 2D coordinates
    if mol.GetNumConformers() == 0:
        _ = gen_coords_for_mol(mol)

    # Create drawing object according to image format
    if img_format == "png":
        draw_method = Chem.Draw.MolDraw2DCairo
    elif img_format == "svg":
        draw_method = Chem.Draw.MolDraw2DSVG
    else:
        raise ValueError(
            f"Image format {img_format} not supported. Choose 'svg' or 'png'."
        )
    d2d = draw_method(*size)

    # Add atom and bond indices
    if atom_numbers is True:
        d2d.drawOptions().addAtomIndices = True
    if bond_numbers is True:
        d2d.drawOptions().addBondIndices = True

    # Add atom and bond labels
    if atom_labels is not None:
        for atom, label in zip(mol.GetAtoms(), atom_labels):
            atom.SetProp("atomNote", f"{label:.{n_decimals}f}")
    if bond_labels is not None:
        for bond, label in zip(mol.GetBonds(), bond_labels):
            bond.SetProp("bondNote", f"{label:.{n_decimals}f}")
    if ring_labels is not None and rings is not None:
        coordinates = mol.GetConformer().GetPositions()
        rw_mol = Chem.RWMol(mol)
        for ring, label in zip(rings, ring_labels):
            idx = rw_mol.AddAtom(Chem.Atom(0))
            rw_mol.GetAtomWithIdx(idx).SetProp("atomNote", f"{label:.{n_decimals}f}")
            coords = np.mean(coordinates[list(ring)], axis=0)
            point = Point3D(*coords)
            conformer = rw_mol.GetConformer()
            conformer.SetAtomPosition(idx, point)
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(
            mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )

    # Draw mol
    if mol_label is None:
        mol_label = ""
    Chem.Draw.PrepareAndDrawMolecule(
        d2d,
        mol,
        legend=mol_label,
        highlightAtoms=highlighted_atoms,
        highlightBonds=highlighted_bonds,
        kekulize=False,
    )

    # Draw properties with similarity map
    if properties is not None:
        SimilarityMaps.GetSimilarityMapFromWeights(mol, list(properties), draw2d=d2d)

    # Draw circles
    if circle_values is not None and rings is not None:
        coordinates = mol.GetConformer().GetPositions()
        if circle_cmax is None:
            max_val = max(circle_values)
        else:
            max_val = circle_cmax
        for ring, value in zip(rings, circle_values):
            center = np.mean(coordinates[list(ring)], axis=0)
            p1 = Point2D(center[0] - circle_radius, center[1] - circle_radius)
            p2 = Point2D(center[0] + circle_radius, center[1] + circle_radius)
            cmap = plt.cm.get_cmap(circle_cmap)
            rgb = cmap(value / abs(max_val) / 2 + 0.5)[:3]
            d2d.SetColour(rgb)
            d2d.DrawEllipse(p1, p2)

    # Finalize drawing
    d2d.FinishDrawing()
    drawing_text: str | bytes = d2d.GetDrawingText()

    return drawing_text


@requires_dependency(
    [
        Import(module="matplotlib", item="pyplot", alias="plt"),
    ],
    globals(),
)
def draw_bond_currents(
    bond_currents: Mapping[tuple[int, int], float],
    coordinates: ArrayLike2D,
    figsize: tuple[int, int] = (12, 12),
    arrow_scale: float = 0.005,
    marker_size: float = 100,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw bond currents.

    Args:
        bond_currents: Bond currents
        coordinates: Coordinates (Å)
        figsize: Figure size
        arrow_scale: Arrow scale factor
        marker_size: Marker size

    Returns:
        fig: Matplotlib Figure object
        ax: Matplotlib Axes objects
    """
    coordinates: Array2DFloat = np.asarray(coordinates)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    for (i, j), I in bond_currents.items():
        if I < 0:
            i, j = j, i
        x, y = coordinates[[i, j], 0], coordinates[[i, j], 1]
        dx, dy = x[1] - x[0], y[1] - y[0]
        width = np.sqrt(arrow_scale * abs(I))
        ax.plot(x, y, linewidth=width * 100, color="C1")
        ax.arrow(
            x[0],
            y[0],
            dx * 0.4,
            dy * 0.4,
            linewidth=width,
            head_width=width * 4,
            color="C1",
        )
        ax.annotate(f"{abs(I):.3f}", (x[0] + dx * 0.4 + 0.05, y[0] + dy * 0.4 + 0.05))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], color="C0", s=marker_size)
    plt.close()

    return fig, ax


@requires_dependency(
    [
        Import(module="matplotlib", item="pyplot", alias="plt"),
    ],
    globals(),
)
def draw_circuit_currents(
    circuits: Iterable[Circuit],
    coordinates: ArrayLike2D,
    figsize: tuple[int, int] = (12, 12),
    arrow_scale: float = 0.005,
    marker_size: float = 100,
) -> list[tuple[plt.Figure, plt.Axes]]:
    """Draws circuit currents.

    Args:
        circuits: Circuits
        coordinates: Coordinates (Å)
        figsize: Figure size
        arrow_scale: Arrow scale factor
        marker_size: Marker size

    Returns:
        plots: Plots as tuples of Figure and Axes
    """
    coordinates: Array2DFloat = np.asarray(coordinates)

    plots = []
    for circuit in circuits:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        indices = list(circuit.indices)
        I = sum(circuit.I)
        if (I > 0 and circuit.ccw is False) or (I < 0 and circuit.ccw is True):
            indices = list(reversed(indices))
        pairs = nx.utils.pairwise(indices, cyclic=True)
        for i, j in pairs:
            x, y = coordinates[[i, j], 0], coordinates[[i, j], 1]
            dx, dy = x[1] - x[0], y[1] - y[0]
            width = np.sqrt(arrow_scale * abs(I))
            ax.plot(x, y, linewidth=width * 100, color="C1")
            ax.arrow(
                x[0],
                y[0],
                dx * 0.4,
                dy * 0.4,
                linewidth=width,
                head_width=width * 6,
                color="C1",
            )
            # ax.quiver(x[0] + dx / 2, y[0] + dy / 2, dx * width, dy * width, color="C1")
        ax.scatter(coordinates[:, 0], coordinates[:, 1], color="C0", s=marker_size)
        plt.close()
        plots.append((fig, ax))

    return plots


def fig_to_bytesio(fig: plt.Figure) -> io.BytesIO:
    """Converts plt.Figure into io.BytesIO.

    Args:
        fig: Matplotlib Figure object

    Returns:
        bytes_io: BytesIO object for figure
    """
    bytes_io = io.BytesIO()
    fig.savefig(bytes_io, format="png")
    bytes_io.seek(0)

    return bytes_io


def draw_png_grid(
    images: Sequence[BinaryIO],
    n_columns: int = 5,
    figsize: tuple[float, float] = (18, 6),
    labels: Iterable[str] | None = None,
) -> tuple[plt.Figure, Array1DAny]:
    """Plot images in grid.

    Args:
        images: PNG images as bytes data, e.g., io.BytesIO
        n_columns: Number of columns
        figsize: Figure size
        labels: Labels for each subplot

    Returns:
        fig: Matplotlib Figure object
        ax: Matplotlib Axes objects
    """
    # Determine number of rows and columns
    n_images = len(images)
    n_columns = min([n_images, n_columns])
    n_rows = n_images // n_columns + np.clip(n_images % n_columns, 0, 1)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize)
    axes: Array1DAny = np.asarray(axes)
    if labels is None:
        labels = [""] * n_images
    for i, (img, label) in enumerate(zip(images, labels)):
        image = plt.imread(img)
        ax = axes.flatten()[i]
        ax.axis("off")
        ax.imshow(image)
        ax.set_title(label)
    plt.close()

    return fig, axes
