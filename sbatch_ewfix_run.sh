#!/bin/bash
#SBATCH --job-name=ew_fixrun
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=ewald_audit/logs/fixrun_%A_%a.out
#SBATCH --error=ewald_audit/logs/fixrun_%A_%a.out

# Step 5 of CLAUDE_TASK_ewald_audit.md: one short production run of the base
# point with corrected Ewald parameters, using the patched *copy* of the code
# in ewald_audit/codefix (src/ and configs_prod/ are untouched).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

setup_cuda_libs() {
    export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-12.3}"
    if [ ! -e "$CUDA_HOME/lib64/libnvrtc.so.12" ]; then
        echo "ERROR: $CUDA_HOME/lib64/libnvrtc.so.12 not found; CuPy will fail." >&2
    fi
    export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
setup_cuda_libs

HS=(3 11)
ZS=(8.0 4.0)
i=${SLURM_ARRAY_TASK_ID:-0}
H=${HS[$i]}
Z=${ZS[$i]}

echo "corrected-Ewald run: H=$H z_scale=$Z"
# non_static_init.mode=random_hs: the production 'random' placement fails
# stochastically at this packing fraction (~0.42); random_hs is the documented
# fallback and only affects the initial configuration, not the physics.
.venv/bin/python ewald_audit/codefix/mc_main.py \
    ewald_audit/configs/ions_dipole_ewfix.yaml \
    pz=17 H=$H ewald.z_scale_factor=$Z \
    system.non_static_init.mode=random_hs \
    experiment.name=ions_dipole_ewfix_H_$H
