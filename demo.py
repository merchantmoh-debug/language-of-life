import sys
import time
import contextlib
import threading
import itertools
from language_of_life.pipeline import DrugDiscoveryPipeline

# --- PALETTE UI KIT ---

@contextlib.contextmanager
def step(name):
    # Determine color based on step name for visual variety (simple hash)
    # Using standard ANSI colors
    # 36: Cyan, 32: Green, 35: Magenta, 34: Blue
    color_code = 36

    start_time = time.time()

    # Header
    print(f"\n\033[1;{color_code}m>>> ARK PROTOCOL: {name}\033[0m")

    try:
        yield
    except Exception as e:
        print(f"\033[1;31m[!] PROTOCOL FAILURE: {name}\033[0m")
        raise e
    finally:
        elapsed = time.time() - start_time
        print(f"\033[90m    Latency: {elapsed:.4f}s\033[0m")

def banner():
    print("""\033[1;36m
    ┏━┓┏━┓┏┓╻   ┏┓╻┏━┓╺┳╸╻ ╻┏━┓┏━┓╻┏━╸
    ┣━┫┣┳┛┣┻┓   ┃┗┃┣━┫ ┃ ┃ ┃┣┳┛┣━┫┃┗━┓
    ╹ ╹╹┗╸╹ ╹   ╹ ╹╹ ╹ ╹ ╹ ╹╹┗╸╹ ╹╹┗━╸
    LANGUAGE OF LIFE v0.2.0 | ARK ARCHITECTURE
    \033[0m""")

def main():
    banner()

    # Example: Insulin A chain
    # Sequence: G I V E Q C C T S I C S L Y Q L E N Y C N
    insulin_seq = "GIVEQCCTSICSLYQLENYCN"

    # Example: Aspirin SMILES
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    print("\033[1;33m[*] INITIALIZING BIO-KERNEL...\033[0m")

    try:
        pipeline = DrugDiscoveryPipeline()

        # We need to suppress the raw prints from the pipeline itself to let Palette control the UI
        # But wait, the pipeline prints useful info (structure confidence, binding affinity).
        # We should capture it or just let it flow but frame it nicely.
        # Since I cannot modify pipeline.py heavily (it prints directly),
        # I will just wrap the run call in a high-level step.

        with step("END-TO-END DISCOVERY"):
            pipeline.run(
                target_sequence=insulin_seq,
                ligand_smiles=aspirin_smiles
            )

    except ValueError as ve:
        print(f"\n\033[1;31m[ERROR] VALIDATION FAILURE:\033[0m {ve}")
        print("\033[90mEnsure your protein sequence contains only amino acids and SMILES is valid.\033[0m")
    except ImportError as ie:
         print(f"\n\033[1;31m[ERROR] DEPENDENCY MISSING:\033[0m {ie}")
    except Exception as e:
        print(f"\n\033[1;31m[FATAL] SYSTEM INTERRUPT:\033[0m {e}")
        print("Note: Ensure you have ~4GB RAM available for ESMFold.")

if __name__ == "__main__":
    main()
