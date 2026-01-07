import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmForProteinFolding
from ..structs import ProteinSequence, ProteinStructure
import numpy as np

class StructurePredictor(nn.Module):
    """
    Predicts 3D atomic coordinates from sequence data.
    Uses ESMFold as it is fully differentiable and python-native.
    """
    def __init__(self, model_name: str = "facebook/esmfold_v1"):
        super().__init__()
        print(f"[*] Loading Folding Model: {model_name}...")
        # low_cpu_mem_usage=True requires 'accelerate' library
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimization for Ampere+ GPUs
        if torch.cuda.is_available():
            self.model.half() # Use FP16 for inference speed

    def fold(self, protein: ProteinSequence) -> ProteinStructure:
        self.model.eval()

        inputs = self.tokenizer([protein.sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract PDB string using HuggingFace utility
        final_pdb = self.model.output_to_pdb(outputs)[0]

        # Extract Coordinates (Batch 0, Residues, Atoms (CA=1), XYZ)
        # ESMFold output positions are in Angstroms
        positions = outputs.positions.float().detach().cpu().numpy()
        # Index 1 is Alpha Carbon (CA)
        ca_coords = positions[0, :, 1, :]

        # Extract Confidence (pLDDT)
        plddt = outputs.plddt.float().detach().cpu()

        return ProteinStructure(
            id=protein.id,
            pdb_content=final_pdb,
            coordinates=ca_coords,
            confidence=plddt
        )
