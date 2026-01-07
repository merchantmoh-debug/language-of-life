import torch
import numpy as np
from ..structs import ProteinStructure, SmallMolecule
from scipy.spatial.distance import cdist

class LigandDocker:
    """
    Simulates the docking process.

    In a full DiffDock setup, this uses a diffusion model.
    Here, we implement a Geometric Pocket Finder that identifies the deepest
    concavity on the protein surface (binding site) and positions the ligand there,
    calculating interaction potential based on geometry.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def dock(self, protein: ProteinStructure, ligand: SmallMolecule) -> SmallMolecule:
        """
        Executes the docking routine.
        """
        print(f"[*] Docking {ligand.name} to {protein.id}...")

        # 1. Identify Binding Pockets (Concavity Analysis)
        # We look for regions with high local atom density (potential pockets)
        coords = protein.coordinates

        # Calculate all-pairs distance
        dist_matrix = cdist(coords, coords)

        # Count neighbors within 10 Angstroms for each residue
        neighbors = np.sum(dist_matrix < 10.0, axis=1)

        # Pocket center is approximated by the surface region with high neighbor density
        pocket_idx = np.argmax(neighbors)
        pocket_center = coords[pocket_idx]

        # 2. "Diffuse" Ligand to Pocket
        # Place ligand at pocket center + slight gaussian noise (Simulating diffusion)
        # In a real scenario, we would conform the ligand using RDKit to 3D first
        ligand_pos = pocket_center + np.random.normal(0, 0.5, size=(3,))

        # 3. Score Affinity (Interaction Potential)
        # Calculate distance from ligand center to all protein atoms
        dists = np.linalg.norm(coords - ligand_pos, axis=1)
        min_dist = np.min(dists)

        # Optimal binding distance is ~2.5 - 4.0 Angstroms
        # Simple Lennard-Jones-like scoring
        if min_dist < 1.5:
            score = 0.2 # Steric Clash
        elif 1.5 <= min_dist <= 4.0:
            score = 0.95 # Strong Binding
        else:
            score = 10.0 / (min_dist + 1.0) # Weak Binding (Decay)

        ligand.coords = ligand_pos
        ligand.affinity_score = score

        return ligand
