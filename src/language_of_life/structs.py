from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np

@dataclass
class ProteinSequence:
    id: str
    sequence: str
    embedding: Optional[torch.Tensor] = None

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
