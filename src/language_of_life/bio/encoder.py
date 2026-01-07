import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
from typing import List
from ..structs import ProteinSequence

class BioEncoder(nn.Module):
    """
    Wraps ESM-2 (Evolutionary Scale Modeling) to convert
    amino acid sequences into high-dimensional semantic vectors.
    """
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        print(f"[*] Loading ESM-2 Model: {model_name}...")
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, proteins: List[ProteinSequence]) -> List[ProteinSequence]:
        self.model.eval()
        processed = []

        for prot in proteins:
            # Tokenize
            inputs = self.tokenizer(prot.sequence, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract Residue Embeddings [1, Seq_Len, Hidden_Dim]
            # We assume per-residue representations for folding/docking
            embeddings = outputs.last_hidden_state.squeeze(0)

            prot.embedding = embeddings.cpu()
            processed.append(prot)

        return processed
