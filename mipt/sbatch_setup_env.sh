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

# A real kernel launch plus the special function the Ewald real-space sum
# needs -- an import alone would not catch a broken libnvrtc.
a = cp.random.random((2048, 2048), dtype=cp.float64)
assert abs(float((a @ a.T).sum()) - float((a @ a.T).sum())) == 0.0
r = cp.linspace(0.1, 5.0, 1000, dtype=cp.float64)
e = cp.asnumpy(cp.special.erfc(r) / r)
print("erfc(r)/r at r=0.1:", float(e[0]))
print("OK: cupy compiles and runs on this device")
PY

echo
echo "=== environment ready: $(pwd)/.venv ==="
