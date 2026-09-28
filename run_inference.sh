#!/bin/bash
# Nutzung: ./run_inference.sh <input_dir> <output_dir>
# Beispiel: ./run_inference.sh dataset-verse19test out/results
#
# Anforderungen: Python >= 3.10, pip, kein Docker, GPU optional
# Limitation: Ohne Ground-Truth-Labels kein Crop auf Lendenregion

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Nutzung: ./run_inference.sh <input_dir> <output_dir>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -q -r "$SCRIPT_DIR/requirements.txt" --quiet
python "$SCRIPT_DIR/inference.py" --input "$1" --output "$2" --weights "$SCRIPT_DIR/best.pt"