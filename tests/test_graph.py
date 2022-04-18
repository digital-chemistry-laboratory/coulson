"""Test graph code."""

from __future__ import annotations

import csv
from pathlib import Path

from numpy.testing import assert_almost_equal
from rdkit import Chem

from coulson.graph import calculate_tre
from coulson.interface import process_rdkit_mol
from coulson.matrix import prepare_huckel_matrix

DATA_DIR = Path(__file__).parent / "data" / "graph"


def pytest_generate_tests(metafunc):
    """Generate test data from csv file."""
    if "pah_data" in metafunc.fixturenames:
        with open(DATA_DIR / "reference_pah.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("pah_data", records)
    elif "hetero_data" in metafunc.fixturenames:
        with open(DATA_DIR / "reference_hetero.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("hetero_data", records)
    elif "nonbenz_data" in metafunc.fixturenames:
        with open(DATA_DIR / "reference_nonbenzenoid.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("nonbenz_data", records)


def test_pah(pah_data):
    """Test against buried volume reference data."""
    data = pah_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    n_electrons = sum(electrons) - int(data["charge"])
    tre, p_tre = calculate_tre(huckel_matrix, n_electrons)

    assert_almost_equal(tre, float(data["tre"]), decimal=4)
    assert_almost_equal(p_tre, float(data["p_tre"]), decimal=3)


def test_nonbenz(nonbenz_data):
    """Test against buried volume reference data."""
    data = nonbenz_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    n_electrons = sum(electrons) - int(data["charge"])
    tre, p_tre = calculate_tre(huckel_matrix, n_electrons)

    assert_almost_equal(tre, float(data["tre"]), decimal=4)
    assert_almost_equal(p_tre, float(data["p_tre"]), decimal=3)


def test_hetero(hetero_data):
    """Test against buried volume reference data."""
    data = hetero_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    n_electrons = sum(electrons) - int(data["charge"])
    tre, p_tre = calculate_tre(huckel_matrix, n_electrons)

    assert_almost_equal(tre, float(data["tre"]), decimal=4)
    assert_almost_equal(p_tre, float(data["p_tre"]), decimal=3)
