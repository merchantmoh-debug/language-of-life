import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from language_of_life.structs import ProteinSequence, ProteinStructure, SmallMolecule
from language_of_life.chem.docking import LigandDocker

class TestStructs(unittest.TestCase):
    def test_protein_sequence(self):
        prot = ProteinSequence(id="test", sequence="ACGT")
        self.assertEqual(prot.id, "test")
        self.assertEqual(prot.sequence, "ACGT")
        self.assertIsNone(prot.embedding)

    def test_protein_structure(self):
        coords = np.array([[0,0,0], [1,1,1]])
        prot = ProteinStructure(id="test", pdb_content="FAKE", coordinates=coords)
        self.assertEqual(prot.id, "test")
        self.assertEqual(prot.pdb_content, "FAKE")
        self.assertTrue(np.array_equal(prot.coordinates, coords))

    def test_small_molecule(self):
        mol = SmallMolecule(name="drug", smiles="C")
        self.assertEqual(mol.name, "drug")
        self.assertEqual(mol.smiles, "C")
        self.assertEqual(mol.affinity_score, 0.0)

class TestDocking(unittest.TestCase):
    def test_docking_logic(self):
        # Create a mock protein structure (2 residues)
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        prot = ProteinStructure(id="test_prot", pdb_content="", coordinates=coords)

        ligand = SmallMolecule(name="test_drug", smiles="C")

        docker = LigandDocker()
        result_ligand = docker.dock(prot, ligand)

        # Verify ligand was placed
        self.assertIsNotNone(result_ligand.coords)
        # Verify affinity score was calculated
        self.assertTrue(isinstance(result_ligand.affinity_score, float))
        # Logic check: Ligand should be close to one of the "pockets"
        # Since we use simple density (all neighbors within 10A), both points have 0 neighbors (dist > 10).
        # Wait, cdist(coords, coords) -> [[0, 17.32], [17.32, 0]]
        # Neighbors < 10.0:
        # P0: [T, F] -> sum 1
        # P1: [F, T] -> sum 1
        # Argmax could be 0 or 1.
        # Ligand position is pocket_center + noise.
        # Check if ligand is roughly near one of the atoms.
        dist0 = np.linalg.norm(result_ligand.coords - coords[0])
        dist1 = np.linalg.norm(result_ligand.coords - coords[1])
        self.assertTrue(dist0 < 5.0 or dist1 < 5.0)

if __name__ == "__main__":
    unittest.main()
