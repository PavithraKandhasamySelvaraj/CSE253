"""
Step 0 — Phase A setup. Copies the SAME 677-name list used for the Llama-2
Stage 1 run into this model's data directory.

Run this ONCE per new model, before 02_baseline_responses.py.
"""
import os
import shutil

import config


def main():
    if not os.path.exists(config.NAMES_SOURCE_PATH):
        print(f"ERROR: could not find the Stage 1 names file at {config.NAMES_SOURCE_PATH}")
        print("If this is a fresh session, re-upload names.json from your Stage 1 backup")
        print("to that path first, or point config.NAMES_SOURCE_PATH at wherever you put it.")
        return

    shutil.copy(config.NAMES_SOURCE_PATH, config.NAMES_PATH)

    import json
    with open(config.NAMES_PATH) as f:
        names = json.load(f)["public_figures"]

    print(f"Copied {len(names)} names from {config.NAMES_SOURCE_PATH}")
    print(f"             to {config.NAMES_PATH}")
    print(f"\nModel for this run: {config.TARGET_MODEL}")
    print(f"Data directory:     {config.DATA_DIR}")
    print("\nNext step: 02_baseline_responses.py (01_generate_names.py is skipped — reusing names)")


if __name__ == "__main__":
    main()
