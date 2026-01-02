"""Language of Life (LoL) Pipeline

An end-to-end framework for AI-powered drug discovery integrating:
- AlphaFold2 for protein structure prediction
- ESM-2 for protein embeddings
- DiffDock for molecular docking
- Network pharmacology for systems-level analysis

Author: Mohamad Al-Zawahreh
License: MIT
"""

__version__ = "0.1.0-alpha"
__author__ = "Mohamad Al-Zawahreh"
__license__ = "MIT"

from .structure import predict_structure
from .embeddings import extract_embeddings
from .docking import dock_compound
from .network import analyze_pathways

__all__ = [
    "predict_structure",
    "extract_embeddings",
    "dock_compound",
    "analyze_pathways",
]


class LoLPipeline:
    """Main pipeline class for end-to-end drug discovery workflow.
    
    Example:
        >>> pipeline = LoLPipeline()
        >>> results = pipeline.run(
        ...     sequence="MSEQENCE...",
        ...     compound="CC1=CC=C(C=C1)O",
        ...     analyze_network=True
        ... )
    """
    
    def __init__(self, config=None):
        """Initialize the pipeline with optional configuration.
        
        Args:
            config (dict, optional): Pipeline configuration parameters
        """
        self.config = config or {}
        
    def run(self, sequence, compound, analyze_network=False):
        """Run the complete pipeline.
        
        Args:
            sequence (str): Protein sequence
            compound (str): SMILES string of compound
            analyze_network (bool): Whether to perform network analysis
            
        Returns:
            dict: Results containing structure, embeddings, docking, and network data
        """
        # Structure prediction
        structure = predict_structure(sequence)
        
        # Extract embeddings
        embeddings = extract_embeddings(sequence)
        
        # Molecular docking
        docking_results = dock_compound(structure, compound)
        
        results = {
            "structure": structure,
            "embeddings": embeddings,
            "docking": docking_results,
        }
        
        # Optional network analysis
        if analyze_network:
            pathway_results = analyze_pathways(sequence, docking_results)
            results["pathways"] = pathway_results
            
        return results
