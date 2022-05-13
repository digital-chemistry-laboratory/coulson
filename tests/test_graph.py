"""Test graph code."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from rdkit import Chem

from coulson.graph import (
    bond_analysis,
    calculate_bre,
    calculate_circuits,
    calculate_tre,
    cycles_from_connectivity,
    global_analysis,
)
from coulson.huckel import HuckelCalculator, prepare_huckel_matrix
from coulson.interface import coords_2D_from_rdkit, process_rdkit_mol, scale_rdkit_mol
from coulson.typing import Array1DFloat

DATA_DIR = Path(__file__).parent / "data" / "graph"


def pytest_generate_tests(metafunc):  # noqa: C901
    """Generate test data from csv file."""
    records: list[dict[Any, Any]]
    if "pah_data" in metafunc.fixturenames:
        with open(DATA_DIR / "pah.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("pah_data", records)
    if "pah_charged_data" in metafunc.fixturenames:
        with open(DATA_DIR / "pah_charged.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("pah_charged_data", records)
    elif "hetero_data" in metafunc.fixturenames:
        with open(DATA_DIR / "hetero.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("hetero_data", records)
    elif "nonbenz_data" in metafunc.fixturenames:
        with open(DATA_DIR / "nonbenzenoid.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        metafunc.parametrize("nonbenz_data", records)
    elif "bre_data" in metafunc.fixturenames:
        with open(DATA_DIR / "bre.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        with open(DATA_DIR / "bre_values.csv") as f:
            reader = csv.DictReader(f)
            records_values = list(reader)
        for record in records:
            values = [
                record_values
                for record_values in records_values
                if record_values["name"] == record["name"]
            ]
            record["values"] = values
        metafunc.parametrize("bre_data", records)
    elif "bre_charged_data" in metafunc.fixturenames:
        with open(DATA_DIR / "bre_charged.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        with open(DATA_DIR / "bre_charged_values.csv") as f:
            reader = csv.DictReader(f)
            records_values = list(reader)
        for record in records:
            values = [
                record_values
                for record_values in records_values
                if record_values["name"] == record["name"]
            ]
            record["values"] = values
        metafunc.parametrize("bre_charged_data", records)
    elif "bc_data" in metafunc.fixturenames:
        with open(DATA_DIR / "bond_currents.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        with open(DATA_DIR / "bond_currents_values.csv") as f:
            reader = csv.DictReader(f)
            records_values = list(reader)
        for record in records:
            values = [
                record_values
                for record_values in records_values
                if record_values["name"] == record["name"]
            ]
            record["values"] = values
        metafunc.parametrize("bc_data", records)
    elif "cc_data" in metafunc.fixturenames:
        with open(DATA_DIR / "cycle_currents.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        with open(DATA_DIR / "cycle_currents_values.csv") as f:
            reader = csv.DictReader(f)
            records_values = list(reader)
        for record in records:
            values = [
                record_values
                for record_values in records_values
                if record_values["name"] == record["name"]
            ]
            record["values"] = values
        metafunc.parametrize("cc_data", records)
    elif "sse_data" in metafunc.fixturenames:
        with open(DATA_DIR / "sse.csv") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        with open(DATA_DIR / "sse_values.csv") as f:
            reader = csv.DictReader(f)
            records_values = list(reader)
        for record in records:
            values = [
                record_values
                for record_values in records_values
                if record_values["name"] == record["name"]
            ]
            record["values"] = values
        metafunc.parametrize("sse_data", records)


def test_tre_pah(pah_data):
    """Test TRE for PAHs."""
    data = pah_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    tre, p_tre = calculate_tre(huckel_matrix, sum(electrons) - int(data["charge"]))

    assert_almost_equal(tre, float(data["tre"]), decimal=4)
    assert_almost_equal(p_tre, float(data["p_tre"]), decimal=3)


def test_tre_pah_charged(pah_charged_data):
    """Test TRE for PAHs."""
    data = pah_charged_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    tre, _ = calculate_tre(huckel_matrix, sum(electrons) - int(data["charge"]))

    assert_almost_equal(tre, float(data["tre"]), decimal=3)


def test_tre_nonbenz(nonbenz_data):
    """Test TRE for non-benzenoids."""
    data = nonbenz_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    tre, p_tre = calculate_tre(huckel_matrix, sum(electrons) - int(data["charge"]))

    assert_almost_equal(tre, float(data["tre"]), decimal=4)
    assert_almost_equal(p_tre, float(data["p_tre"]), decimal=3)


def test_tre_hetero(hetero_data):
    """Test TRE for heteroaromatics."""
    data = hetero_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    tre, p_tre = calculate_tre(huckel_matrix, sum(electrons) - int(data["charge"]))

    assert_almost_equal(tre, float(data["tre"]), decimal=4)
    assert_almost_equal(p_tre, float(data["p_tre"]), decimal=3)


def test_mre_pah(pah_data):
    """Test diamagnetic susceptibility for PAHs."""
    data = pah_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )

    huckel = HuckelCalculator(huckel_matrix, electrons)
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)

    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    circuits = calculate_circuits(
        cycles,
        huckel_matrix,
        shell_energies=huckel.unique_energies,
        shell_occupations=huckel.shell_occupations_avg,
        degeneracies=huckel.degeneracies,
        coordinates=coordinates,
    )
    mre, _ = global_analysis(circuits)

    assert_almost_equal(mre, float(data["mre"]), decimal=4)


def test_mre_pah_charged(pah_charged_data):
    """Test diamagnetic susceptibility for PAHs."""
    data = pah_charged_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )

    huckel = HuckelCalculator(huckel_matrix, electrons, charge=int(data["charge"]))
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)

    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    if data["name"] in ["Benzene", "Triphenylene", "Coronene"]:
        with pytest.raises(ValueError, match=r"^Number of unpaired"):
            circuits = calculate_circuits(
                cycles,
                huckel_matrix,
                shell_energies=huckel.unique_energies,
                shell_occupations=huckel.shell_occupations_avg,
                degeneracies=huckel.degeneracies,
                coordinates=coordinates,
                multiplicity=1,
            )
        return
    else:
        circuits = calculate_circuits(
            cycles,
            huckel_matrix,
            shell_energies=huckel.unique_energies,
            shell_occupations=huckel.shell_occupations_avg,
            degeneracies=huckel.degeneracies,
            coordinates=coordinates,
            multiplicity=1,
        )
    mre, _ = global_analysis(circuits)

    assert_almost_equal(mre, float(data["mre"]), decimal=3)


def test_mre_hetero(hetero_data):
    """Test diamagnetic susceptibility for PAHs."""
    data = hetero_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )

    huckel = HuckelCalculator(huckel_matrix, electrons, charge=int(data["charge"]))
    if data["name"] == "M(II) porphyrin":
        mol = Chem.MolFromMolFile(str(DATA_DIR / "porphine.mol"))
        scale_rdkit_mol(mol, bond_length=1.4)
        coordinates = mol.GetConformer().GetPositions()
    else:
        coordinates = coords_2D_from_rdkit(mol, coordgen=True)

    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    circuits = calculate_circuits(
        cycles,
        huckel_matrix,
        shell_energies=huckel.unique_energies,
        shell_occupations=huckel.shell_occupations_avg,
        degeneracies=huckel.degeneracies,
        coordinates=coordinates,
    )
    mre, _ = global_analysis(circuits)

    assert_almost_equal(mre, float(data["mre"]), decimal=4)


def test_susc_pah(pah_data):
    """Test diamagnetic susceptibility for PAHs."""
    data = pah_data
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )

    huckel = HuckelCalculator(huckel_matrix, electrons)
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)

    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    circuits = calculate_circuits(
        cycles,
        huckel_matrix,
        shell_energies=huckel.unique_energies,
        shell_occupations=huckel.shell_occupations_avg,
        degeneracies=huckel.degeneracies,
        coordinates=coordinates,
    )
    _, susc = global_analysis(circuits)

    assert_almost_equal(susc, float(data["susc"]), decimal=2)


def test_t_bre(bre_data):
    """Test t-BRE values."""
    data = bre_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    huckel = HuckelCalculator(huckel_matrix, electrons, charge=int(data["charge"]))
    for value in values:
        t_bre = calculate_bre(
            huckel_matrix,
            huckel.occupations,
            indices=[(int(value["idx_1"]) - 1, int(value["idx_2"]) - 1)],
        )
        assert_almost_equal(t_bre, float(value["t-bre"]), decimal=3)


def test_t_bre_charged(bre_charged_data):
    """Test t-BRE values for charged molecules."""
    data = bre_charged_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    huckel = HuckelCalculator(huckel_matrix, electrons, charge=int(data["charge"]))

    # Set occupations to closed-shell for open-shell species
    occupations: Array1DFloat
    if data["name"] in ["Benzene", "Triphenylene", "Coronene"]:
        occupations = np.array(
            [2] * int(len(huckel_matrix) / 2 - 1)
            + [0] * int(len(huckel_matrix) / 2 + 1),
            dtype=float,
        )
    else:
        occupations = huckel.occupations
    for value in values:
        t_bre = calculate_bre(
            huckel_matrix,
            occupations,
            indices=[(int(value["idx_1"]) - 1, int(value["idx_2"]) - 1)],
        )
        assert_almost_equal(t_bre, float(value["t-bre"]), decimal=3)


def test_m_bre(bre_data):
    """Test t-BRE values."""
    data = bre_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    huckel = HuckelCalculator(
        huckel_matrix, electrons, charge=int(data["charge"]), multiplicity=1
    )
    circuits = calculate_circuits(
        cycles,
        huckel_matrix,
        shell_energies=huckel.unique_energies,
        shell_occupations=huckel.shell_occupations_avg,
        degeneracies=huckel.degeneracies,
        coordinates=coordinates,
        multiplicity=1,
    )
    _, m_bres = bond_analysis(circuits)

    for value in values:
        m_bre = m_bres[frozenset([int(value["idx_1"]) - 1, int(value["idx_2"]) - 1])]
        assert_almost_equal(m_bre, float(value["m-bre"]), decimal=3)


def test_m_bre_charged(bre_charged_data):
    """Test t-BRE values."""
    data = bre_charged_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    huckel = HuckelCalculator(
        huckel_matrix, electrons, charge=int(data["charge"]), multiplicity=1
    )

    if data["name"] in ["Benzene", "Triphenylene", "Coronene"]:
        with pytest.raises(ValueError, match=r"^Number of unpaired"):
            circuits = calculate_circuits(
                cycles,
                huckel_matrix,
                shell_energies=huckel.unique_energies,
                shell_occupations=huckel.shell_occupations_avg,
                degeneracies=huckel.degeneracies,
                coordinates=coordinates,
                multiplicity=1,
            )
        return
    else:
        circuits = calculate_circuits(
            cycles,
            huckel_matrix,
            shell_energies=huckel.unique_energies,
            shell_occupations=huckel.shell_occupations_avg,
            degeneracies=huckel.degeneracies,
            coordinates=coordinates,
            multiplicity=1,
        )
    _, m_bres = bond_analysis(circuits)

    for value in values:
        m_bre = m_bres[frozenset([int(value["idx_1"]) - 1, int(value["idx_2"]) - 1])]
        assert_almost_equal(m_bre, float(value["m-bre"]), decimal=3)


def test_bond_currents(bc_data):
    """Test bond currents."""
    data = bc_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    huckel = HuckelCalculator(
        huckel_matrix, electrons, charge=int(data["charge"]), multiplicity=1
    )
    circuits = calculate_circuits(
        cycles,
        huckel_matrix,
        shell_energies=huckel.unique_energies,
        shell_occupations=huckel.shell_occupations_avg,
        degeneracies=huckel.degeneracies,
        coordinates=coordinates,
        multiplicity=1,
    )
    bcs, _ = bond_analysis(circuits)
    for value in values:
        idx_1, idx_2 = int(value["idx_1"]) - 1, int(value["idx_2"]) - 1
        bc = bcs.get((idx_1, idx_2))
        if bc is None:
            bc = -bcs[(idx_2, idx_1)]
        assert_almost_equal(bc, float(value["bond_current"]), decimal=3)


def test_cycle_currents(cc_data):
    """Test cycle currents."""
    data = cc_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    coordinates = coords_2D_from_rdkit(mol, coordgen=True)
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    cycles = cycles_from_connectivity(input_data.connectivity_matrix)
    huckel = HuckelCalculator(
        huckel_matrix, electrons, charge=int(data["charge"]), multiplicity=1
    )
    circuits = calculate_circuits(
        cycles,
        huckel_matrix,
        shell_energies=huckel.unique_energies,
        shell_occupations=huckel.shell_occupations_avg,
        degeneracies=huckel.degeneracies,
        coordinates=coordinates,
        multiplicity=1,
    )
    circuit_indices = [frozenset(circuit.indices) for circuit in circuits]
    for value in values:
        indices = [int(value) - 1 for value in value["indices"].split(",")]
        idx = circuit_indices.index(frozenset(indices))
        circuit = circuits[idx]
        I = sum(circuit.I)
        assert_almost_equal(I, float(value["current"]), decimal=3)


def test_sse(sse_data):
    """Test SSE."""
    data = sse_data
    values = data["values"]
    mol = Chem.MolFromSmiles(data["smiles"])
    input_data, _ = process_rdkit_mol(mol)
    huckel_matrix, electrons = prepare_huckel_matrix(
        input_data.atom_types, input_data.connectivity_matrix
    )
    huckel = HuckelCalculator(huckel_matrix, electrons, charge=int(data["charge"]))
    for value in values:
        i_indices = [int(i) - 1 for i in value["i_indices"].split(",")]
        pairs = list(zip(*[iter(i_indices)] * 2))
        sse = calculate_bre(huckel_matrix, huckel.occupations, indices=pairs)
        assert_almost_equal(sse, float(value["sse"]), decimal=4)
