#!/bin/sh
# Build the native core on the Mac: CLI (core/build/stego_core_cli) + the pybind11 module (core/python/geostego_core*.so).
# Needs: cmake, ninja, and `uv pip install --python .venv/bin/python pybind11`.
set -e
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
cmake -S core -B core/build -G Ninja -DSTEGO_PYTHON=ON -Dpybind11_DIR="$($PY -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$PY" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/core/python" -DCMAKE_BUILD_TYPE=Release
cmake --build core/build -j4
