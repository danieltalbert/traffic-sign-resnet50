#!/usr/bin/env bash

# Stop on errors, unset variables, or failed pipeline commands.
set -euo pipefail

# Resolve a Python interpreter without publishing a machine-specific home path.
# TF_PYTHON remains the explicit escape hatch for custom CUDA/Conda setups.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TF_PYTHON:-}" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    TF_PYTHON="$CONDA_PREFIX/bin/python"
fi

if [[ -z "${TF_PYTHON:-}" ]]; then
    for candidate in \
        "$PROJECT_ROOT/.venv/bin/python" \
        "$PROJECT_ROOT/venv/bin/python" \
        "$HOME/miniforge3/envs/tf/bin/python" \
        "$HOME/miniconda3/envs/tf/bin/python"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            TF_PYTHON="$candidate"
            break
        fi
    done
fi

# Fall back to PATH only after checking project-local and common Conda environments.
if [[ -z "${TF_PYTHON:-}" ]] && command -v python3 >/dev/null 2>&1; then
    TF_PYTHON="$(command -v python3)"
fi

if [[ -z "${TF_PYTHON:-}" || ! -x "$TF_PYTHON" ]]; then
    echo "No executable Python was found. Create .venv or set TF_PYTHON." >&2
    exit 1
fi

# NVIDIA pip wheels keep each CUDA library in a separate package-level lib folder.
NVIDIA_PACKAGE_ROOT="$($TF_PYTHON -c 'import site; print(site.getsitepackages()[0])')/nvidia"

# Build a colon-separated search path so TensorFlow can discover all packaged CUDA libraries.
NVIDIA_LIBRARY_PATHS=""
if [[ -d "$NVIDIA_PACKAGE_ROOT" ]]; then
    NVIDIA_LIBRARY_PATHS="$(find "$NVIDIA_PACKAGE_ROOT" -type d -name lib -print | paste -sd: -)"
fi

# WSL provides the Windows GPU driver bridge in /usr/lib/wsl/lib.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIBRARY_PATHS}:$(dirname "$TF_PYTHON")/../lib:${LD_LIBRARY_PATH:-}"

# Flush epoch and diagnostic messages immediately so long runs remain observable.
export PYTHONUNBUFFERED=1

# Execute the requested Python script with all original arguments unchanged.
exec "$TF_PYTHON" "$@"
