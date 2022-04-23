"""Drawing tools."""

from __future__ import annotations

from collections.abc import Iterable
import typing

import numpy as np

from coulson.typing import Array1DFloat, Array1DInt, ArrayLike1D
from coulson.utils import Import, requires_dependency

if typing.TYPE_CHECKING:  # pragma: no cover
    import matplotlib.pyplot as plt  # pragma: no cover
    from rdkit import Chem  # pragma: no cover
    from rdkit.Chem.Draw import SimilarityMaps  # pragma: no cover


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
    Chem.Draw.rdDepictor.SetPreferCoordGen(True)

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

    return fig, ax


@requires_dependency(
    [
        Import(module="rdkit", item="Chem"),
        Import(module="rdkit.Chem.Draw", item="SimilarityMaps"),
    ],
    globals(),
)
def draw_mol(  # noqa: C901
    mol: "Chem.Mol",
    properties: Iterable[float] | None = None,
    atom_numbers: bool = True,
    atom_labels: Iterable[float] | None = None,
    bond_numbers: bool = False,
    bond_labels: Iterable[float] | None = None,
    highlighted_atoms: Iterable[int] | None = None,
    highlighted_bonds: Iterable[int] | None = None,
    size: tuple[int, int] = (400, 400),
    n_decimals: int = 3,
    img_format: str = "svg",
) -> str | bytes:
    """Draw molecule with RDKit.

    Args:
        mol: Molecule
        properties: Properties for contour plot
        atom_numbers: Whether to print atom numbers
        atom_labels: Labels to print for atoms
        bond_numbers: Whether to print bond numbers
        bond_labels: Labels to print for bonds
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
    Chem.Draw.rdDepictor.SetPreferCoordGen(True)

    # Set mutable default arguments
    if highlighted_atoms is None:
        highlighted_atoms = []
    if highlighted_bonds is None:
        highlighted_bonds = []

    # Copy mol object to prevent editing it in place
    mol = Chem.Mol(mol)

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

    # Draw properties with similarity map
    if properties is not None:
        SimilarityMaps.GetSimilarityMapFromWeights(mol, list(properties), draw2d=d2d)

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

    # Finalize drawing
    d2d.DrawMolecule(
        mol, highlightAtoms=highlighted_atoms, highlightBonds=highlighted_bonds
    )

    d2d.FinishDrawing()
    drawing_text: str | bytes = d2d.GetDrawingText()

    return drawing_text
