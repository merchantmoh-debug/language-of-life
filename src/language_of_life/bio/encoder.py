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
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            print(f"[*] Loading ESM-2 Tokenizer: {self.model_name}...")
            self._tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            print(f"[*] Loading ESM-2 Model: {self.model_name}...")
            self._model = EsmModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def forward(self, proteins: List[ProteinSequence]) -> List[ProteinSequence]:
        processed = []

        # Ensure model is loaded
        _ = self.model

        for prot in proteins:
            # Tokenize
            inputs = self.tokenizer(prot.sequence, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.inference_mode():
                outputs = self.model(**inputs)

            # Extract Residue Embeddings [1, Seq_Len, Hidden_Dim]
            # We assume per-residue representations for folding/docking
            embeddings = outputs.last_hidden_state.squeeze(0)

            prot.embedding = embeddings.cpu()
            processed.append(prot)

        return processed
