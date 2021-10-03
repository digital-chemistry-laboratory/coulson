"""Test drawings tools."""

import matplotlib
import numpy as np
import pytest
from rdkit import Chem

from coulson.draw import draw_mol, draw_orbital_energies
from coulson.huckel import HuckelCalculator
from coulson.matrix import prepare_huckel_matrix

matplotlib.use("Agg")


@pytest.fixture
def huckel_input():
    """Fixture that creates Hückel matrix with number of electrons."""
    atom_types = ["C", "C", "C", "C"]
    connectivity_matrix = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    huckel_matrix, n_electrons = prepare_huckel_matrix(atom_types, connectivity_matrix)

    return huckel_matrix, n_electrons


def test_draw_orbital_energies(huckel_input):
    """Test tool to draw orbital energies."""
    huckel_matrix, n_electrons = huckel_input
    hc = HuckelCalculator(huckel_matrix, n_electrons)

    fig, ax = draw_orbital_energies(hc.energies)

    assert True


def test_draw_orbital_energies_occupations(huckel_input):
    """Test tool to draw orbital energies."""
    huckel_matrix, n_electrons = huckel_input
    hc = HuckelCalculator(huckel_matrix, n_electrons)

    fig, ax = draw_orbital_energies(hc.energies, occupations=hc.occupations)

    assert True


def test_draw_orbital_energies_raise_degeneracy(huckel_input):
    """Test tool to draw orbital energies."""
    huckel_matrix, n_electrons = huckel_input
    hc = HuckelCalculator(huckel_matrix, n_electrons)
    energies = np.append(hc.energies, hc.energies)
    occupations = np.append(hc.occupations, hc.occupations)

    with pytest.raises(ValueError):
        fig, ax = draw_orbital_energies(energies, occupations=occupations)

    assert True


def test_draw_orbital_energies_raise_occupations(huckel_input):
    """Test tool to draw orbital energies."""
    huckel_matrix, n_electrons = huckel_input
    hc = HuckelCalculator(huckel_matrix, n_electrons)
    occupations = hc.occupations
    occupations[0] = 3

    with pytest.raises(ValueError):
        fig, ax = draw_orbital_energies(hc.energies, occupations=occupations)

    assert True


def test_draw_mol_svg(huckel_input):
    """Test tool to draw molecule."""
    mol = Chem.MolFromSmiles("c1ccc1")
    huckel_matrix, n_electrons = huckel_input
    hc = HuckelCalculator(huckel_matrix, n_electrons)
    bond_orders = [
        hc.bond_order(bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1)
        for bond in mol.GetBonds()
    ]
    drawing_text = draw_mol(
        mol,
        properties=hc.coefficients[0],
        atom_numbers=True,
        atom_labels=hc.charges,
        bond_numbers=True,
        bond_labels=bond_orders,
        highlighted_atoms=[0],
        highlighted_bonds=[0],
        size=(200, 200),
        n_decimals=4,
        img_format="svg",
    )

    assert isinstance(drawing_text, str)


def test_draw_mol_png():
    """Test tool to draw molecule."""
    mol = Chem.MolFromSmiles("c1ccc1")

    drawing_text = draw_mol(mol, atom_numbers=False, img_format="png")

    assert isinstance(drawing_text, bytes)


def test_draw_mol_raises_img_format():
    """Test tool to draw molecule."""
    mol = Chem.MolFromSmiles("c1ccc1")

    with pytest.raises(ValueError):
        draw_mol(mol, img_format="wrong")
