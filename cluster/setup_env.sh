#!/usr/bin/env bash
# Shared environment setup for Snellius (SURF).
# Source this script from job scripts or run on the login node to create venvs.
#
# Usage:
#   source cluster/setup_env.sh
#
# Run on the login node first to create the venv (compute nodes may lack internet).

set -euo pipefail

# --- Configurable module names (verify with `module spider` on Snellius) ---
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"

# --- Project root (directory containing this script's parent) ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Load modules ---
module purge
module load 2023
module load "$PYTHON_MODULE"
# Note: Do NOT load a CUDA module. jax[cuda12] ships its own CUDA libraries
# and the system module's LD_LIBRARY_PATH causes version conflicts.

# --- Create / activate venv ---
VENV_NAME="venv-gpu"
VENV_DIR="$PROJECT_DIR/.venvs/$VENV_NAME"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -e "$PROJECT_DIR"
    pip install "jax[cuda12]"
    echo "Venv $VENV_NAME created and packages installed."
else
    source "$VENV_DIR/bin/activate"
    echo "Activated existing venv: $VENV_DIR"
fi
