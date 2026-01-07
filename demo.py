from language_of_life.pipeline import DrugDiscoveryPipeline

def main():
    # Example: Insulin A chain (Short sequence for fast demo)
    # Sequence: G I V E Q C C T S I C S L Y Q L E N Y C N
    insulin_seq = "GIVEQCCTSICSLYQLENYCN"

    # Example: Aspirin SMILES
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    pipeline = DrugDiscoveryPipeline()

    try:
        pipeline.run(
            target_sequence=insulin_seq,
            ligand_smiles=aspirin_smiles
        )
    except Exception as e:
        print(f"\n[!] Pipeline interrupted: {e}")
        print("Note: Ensure you have installed all requirements and have ~4GB RAM available for ESMFold.")

if __name__ == "__main__":
    main()
