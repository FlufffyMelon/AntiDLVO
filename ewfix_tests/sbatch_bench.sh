#!/bin/bash -l
#SBATCH --job-name=ewfix_bench
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=ewfix_tests/logs/bench_%j.out
#SBATCH --error=ewfix_tests/logs/bench_%j.out

# Stage 3.3: how many MC processes should share one V100?
#
#   sbatch ewfix_tests/sbatch_bench.sh
#
# Measures the per-process MC rate at 1, 2, 4, 8 and 16 processes on a single
# card, for the cheapest (H=3) and dearest (H=11) systems, then re-measures the
# chosen density across all four cards at once -- four cards contend for the
# same 128 CPU cores, so per-card numbers do not simply add up.
#
# The rate is read from simulation_data.csv (step and time_s), not from tqdm:
# it survives redirection and lets the warm-up be discarded exactly.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p ewfix_tests/logs ewfix_tests/bench results_tmp/ewfix_tests/bench

module load cuda/12.9
echo "node: $(hostname)  cores: $(nproc)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
PY=$PWD/.venv/bin/python

N_STEPS=${N_STEPS:-2500}

# One measurement point: $1 = H, $2 = processes per GPU, $3 = GPU list.
measure() {
    local H=$1 NPROC=$2 GPUS=$3
    local tag="H${H}_n${NPROC}_g$(echo "$GPUS" | tr -d ',')"
    echo
    echo "=== H=$H  ${NPROC} proc/GPU  on GPUs [$GPUS] ==="

    nvidia-smi dmon -s um -d 5 -o T > "ewfix_tests/bench/dmon_${tag}.txt" 2>&1 &
    local dmon=$!

    local pids=()
    for g in ${GPUS//,/ }; do
        for p in $(seq 1 "$NPROC"); do
            CUDA_VISIBLE_DEVICES=$g "$PY" mc_main.py \
                configs_prod/ions_dipole_ewfix.yaml \
                H=$H \
                simulation.n_steps=$N_STEPS \
                experiment.results_dir=results_tmp/ewfix_tests/bench \
                experiment.name="bench_${tag}_g${g}_p${p}" \
                > "ewfix_tests/logs/bench_${tag}_g${g}_p${p}.out" 2>&1 &
            pids+=($!)
        done
    done

    local t0=$SECONDS
    for pid in "${pids[@]}"; do wait "$pid" || echo "  a process FAILED" >&2; done
    echo "  wall: $((SECONDS - t0)) s for ${#pids[@]} processes"

    kill "$dmon" 2>/dev/null || true
    wait "$dmon" 2>/dev/null || true

    "$PY" ewfix_tests/parse_bench.py \
        --pattern "results_tmp/ewfix_tests/bench/bench_${tag}_*" \
        --label "$tag" --H "$H" --nproc "$NPROC" \
        --ngpu "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" \
        --dmon "ewfix_tests/bench/dmon_${tag}.txt" \
        --append ewfix_tests/bench/bench.csv
}

echo "################ single card, scaling with process count ################"
for H in 3 11; do
    for N in 1 2 4 8 16; do
        measure "$H" "$N" 0
    done
done

echo
echo "################ all four cards at the packing under consideration ################"
for N in ${FULL_NODE_N:-4 8}; do
    measure 3 "$N" 0,1,2,3
    measure 11 "$N" 0,1,2,3
done

echo
echo "################ summary ################"
"$PY" ewfix_tests/parse_bench.py --report ewfix_tests/bench/bench.csv
