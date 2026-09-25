#!/bin/bash
#SBATCH --job-name=ew_bench
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:50:00
#SBATCH --output=ewald_audit/logs/bench_%j.out
#SBATCH --error=ewald_audit/logs/bench_%j.out

# Step 6 of CLAUDE_TASK_ewald_audit.md: cost of the corrected Ewald parameters.
#
# Measures MC throughput (it/s) for the ORIGINAL production setup and for the
# CORRECTED one, at several concurrency levels sharing a single GPU (production
# ran ~22 such processes across the available GPUs).  Nothing in src/ or
# configs_prod/ is modified; the original setup is simply *run* read-only.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CUDA_HOME="${CUDA_HOME:-$HOME/cuda-12.3}"
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

STEPS=${STEPS:-2000}
OUT=ewald_audit/results/bench_raw
mkdir -p "$OUT"

run_one() {   # variant  H  z_scale  slot
    local variant=$1 H=$2 Z=$3 slot=$4
    local log="$OUT/${variant}_H${H}_n${NPROC}_s${slot}.log"
    if [ "$variant" = old ]; then
        .venv/bin/python mc_main.py configs_prod/ions_dipole.yaml \
            pz=17 H="$H" simulation.n_steps="$STEPS" \
            system.non_static_init.mode=random_hs \
            experiment.results_dir=results_tmp/ewald_audit/bench \
            experiment.name="bench_old_H${H}_n${NPROC}_s${slot}" \
            > "$log" 2>&1
    else
        .venv/bin/python ewald_audit/codefix/mc_main.py \
            ewald_audit/configs/ions_dipole_ewfix.yaml \
            pz=17 H="$H" ewald.z_scale_factor="$Z" simulation.n_steps="$STEPS" \
            system.non_static_init.mode=random_hs \
            experiment.results_dir=results_tmp/ewald_audit/bench \
            experiment.name="bench_new_H${H}_n${NPROC}_s${slot}" \
            > "$log" 2>&1
    fi
}

for H_Z in "3 8.0" "11 4.0"; do
    set -- $H_Z; H=$1; Z=$2
    for variant in old new; do
        for NPROC in 1 3; do
            export NPROC
            echo "=== $variant H=$H z=$Z nproc=$NPROC ==="
            for s in $(seq 1 "$NPROC"); do run_one "$variant" "$H" "$Z" "$s" & done
            wait
        done
    done
done

echo "=== parsing ==="
.venv/bin/python ewald_audit/scripts/parse_bench.py
