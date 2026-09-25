#!/bin/bash -l
#SBATCH --job-name=mc_env
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=mipt/logs/setup_env_%j.out
#SBATCH --error=mipt/logs/setup_env_%j.out

# Build the Python environment for this repo on mipt, from a compute node.
#
#   sbatch mipt/sbatch_setup_env.sh
#
# Deliberately NOT the *-common partition: AMD7V12-V100-CCP is our own and has
# no 1-day walltime cap, so production runs submitted later are not evicted.
# node211-27 carries 4x Tesla V100-SXM2-32GB; one is enough to validate CuPy.
#
# `#!/bin/bash -l` is required: uv lives in ~/miniconda3/bin, which only a
# login shell puts on PATH.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p mipt/logs

module load cuda/12.9
echo "node        : $(hostname)"
echo "CUDA_HOME   : ${CUDA_HOME:-unset}"
echo "nvcc        : $(command -v nvcc || echo MISSING)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
    echo "ERROR: uv not on PATH. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
echo "uv          : $UV ($("$UV" --version))"

# Rebuild from scratch when REBUILD=1, otherwise reuse an existing .venv.
if [ "${REBUILD:-0}" = "1" ] || [ ! -x .venv/bin/python ]; then
    "$UV" venv --python 3.13 --clear .venv
fi
"$UV" pip install --python .venv/bin/python -r requirements-cluster.txt

echo
echo "=== verifying the environment on the GPU ==="
.venv/bin/python - <<'PY'
import importlib, sys
print("python:", sys.version.split()[0])

# Everything src/ and mc_main.py import at runtime, plus matplotlib for the
# ewald_audit plotting scripts.
for name in ("numpy", "scipy", "pandas", "omegaconf", "pint",
             "psutil", "tqdm", "yaml", "matplotlib", "cupy"):
    print(f"  {name:12s} {importlib.import_module(name).__version__}")

import cupy as cp
dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("\ngpu:", props["name"].decode(), "cc",
      f"{props['major']}.{props['minor']}",
      "| CUDA runtime", cp.cuda.runtime.runtimeGetVersion())

# Real kernel launches, cross-checked against NumPy -- an import alone would
# not catch a broken libnvrtc or a compute-capability mismatch.  These are the
# double-precision operations the reciprocal-space Ewald sum is built from
# (the real-space part runs on the CPU through scipy).
import numpy as np
rng = np.random.default_rng(0)
pos = rng.random((4096, 3))
kv = rng.random((3, 512))

phase_c = np.exp(-0.25 * (pos @ kv) ** 2) * np.cos(pos @ kv)
sf_c = phase_c.sum(axis=0)

g_pos, g_kv = cp.asarray(pos), cp.asarray(kv)
g_kr = g_pos @ g_kv
sf_g = cp.asnumpy((cp.exp(-0.25 * g_kr ** 2) * cp.cos(g_kr)).sum(axis=0))

err = float(np.max(np.abs(sf_g - sf_c)) / np.max(np.abs(sf_c)))
print(f"\ngpu-vs-cpu structure factor, max relative error: {err:.3e}")
assert err < 1e-12, "CuPy disagrees with NumPy in double precision"
print("OK: cupy compiles and runs correctly on this device")
PY

echo
echo "=== environment ready: $(pwd)/.venv ==="
