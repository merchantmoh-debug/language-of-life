import os
from .structs import ProteinSequence, SmallMolecule
from .bio.encoder import BioEncoder
from .bio.folder import StructurePredictor
from .chem.docking import LigandDocker

class DrugDiscoveryPipeline:
    def __init__(self):
        print(">>> INITIALIZING LANGUAGE OF LIFE PIPELINE...")
        self.encoder = BioEncoder()
        self.folder = StructurePredictor()
        self.docker = LigandDocker()

    def run(self, target_sequence: str, ligand_smiles: str, output_dir: str = "./results"):
        os.makedirs(output_dir, exist_ok=True)

        # 1. Sequence Initialization
        prot = ProteinSequence(id="Target_Alpha", sequence=target_sequence)
        ligand = SmallMolecule(name="Candidate_1", smiles=ligand_smiles)

        # 2. Language Encoding (ESM-2)
        print(f"\n[1/3] Encoding Biological Semantics...")
        prot = self.encoder( [prot] )[0]
        print(f"      Embedding Shape: {prot.embedding.shape}")

        # 3. Structure Prediction (Folding)
        print(f"\n[2/3] Predicting 3D Structure (Folding)...")
        structure = self.folder.fold(prot)

        # Save PDB
        pdb_path = os.path.join(output_dir, f"{prot.id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(structure.pdb_content)

        print(f"      Structure saved to {pdb_path}")
        print(f"      Mean pLDDT Confidence: {structure.confidence.mean().item():.2f}")

        # 4. Molecular Docking
        print(f"\n[3/3] Docking Ligand (Geometric Analysis)...")
        result_ligand = self.docker.dock(structure, ligand)

        print(f"\n>>> DISCOVERY COMPLETE")
        print(f"    Target: {prot.id}")
        print(f"    Ligand: {ligand.name}")
        print(f"    Binding Affinity Score: {result_ligand.affinity_score:.4f}")
        print(f"    Ligand Coordinates: {result_ligand.coords}")

        return structure, result_ligand
