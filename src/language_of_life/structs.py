from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import re

@dataclass
class ProteinSequence:
    id: str
    sequence: str
    embedding: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Allow standard 20 amino acids + B, Z, J, X (ambiguous)
        valid_chars = set("ACDEFGHIKLMNPQRSTVWYBZJX")
        self.sequence = self.sequence.upper()
        if not set(self.sequence).issubset(valid_chars):
            invalid = set(self.sequence) - valid_chars
            raise ValueError(f"Invalid characters in protein sequence: {invalid}")

        if not self.sequence:
            raise ValueError("Protein sequence cannot be empty")

@dataclass
class ProteinStructure:
    id: str
    pdb_content: str
    coordinates: np.ndarray # [N_residues, 3] (CA atoms)
    confidence: Optional[torch.Tensor] = None # pLDDT scores

@dataclass
class SmallMolecule:
    name: str
    smiles: str
    coords: Optional[np.ndarray] = None
    affinity_score: float = 0.0

    def __post_init__(self):
        if not self.smiles.isascii():
            raise ValueError(f"SMILES string must be ASCII: {self.smiles}")
        if not self.smiles:
            raise ValueError("SMILES string cannot be empty")
