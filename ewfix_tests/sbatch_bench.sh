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
# Thread count is a variable here, not a default. The node has 128 cores, so a
# 32-process production packing leaves 4 cores per process; if the BLAS inside
# each process still believes it owns all 128, the oversubscription is what we
# would be measuring. Points tagged t0 leave OMP_NUM_THREADS unset, the rest
# pin it, so the effect is visible rather than assumed.
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
CSV=ewfix_tests/bench/bench.csv

N_STEPS=${N_STEPS:-2500}
POINT=0

# measure <H> <proc per gpu> <gpu list> <threads, 0 = leave unset>
measure() {
    local H=$1 NPROC=$2 GPUS=$3 OMP=$4
    local ngpu; ngpu=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
    local tag="H${H}_n${NPROC}_g${ngpu}_t${OMP}"
    # The same point is measured more than once on purpose (the thread controls
    # repeat two of the scaling points), and run directories are kept. Without
    # a per-invocation serial number the glob below would collect the earlier
    # measurement's directories too and report twice the aggregate rate.
    POINT=$((POINT + 1))
    local run="${tag}_r${POINT}"
    echo
    echo "=== H=$H  ${NPROC} proc/GPU  ${ngpu} GPU  threads=${OMP} ==="

    if [ "$OMP" != "0" ]; then
        export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP \
               OPENBLAS_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP
    else
        unset OMP_NUM_THREADS MKL_NUM_THREADS \
              OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS || true
    fi

    nvidia-smi dmon -s um -d 5 -o T > "ewfix_tests/bench/dmon_${run}.txt" 2>&1 &
    local dmon=$!

    local pids=()
    for g in ${GPUS//,/ }; do
        for p in $(seq 1 "$NPROC"); do
            CUDA_VISIBLE_DEVICES=$g "$PY" mc_main.py \
                configs_prod/ions_dipole_ewfix.yaml \
                H=$H \
                simulation.n_steps=$N_STEPS \
                experiment.results_dir=results_tmp/ewfix_tests/bench \
                experiment.name="bench_${run}_g${g}_p${p}" \
                > "ewfix_tests/logs/bench_${run}_g${g}_p${p}.out" 2>&1 &
            pids+=($!)
        done
    done

    local t0=$SECONDS
    local bad=0
    for pid in "${pids[@]}"; do wait "$pid" || bad=$((bad + 1)); done
    echo "  wall $((SECONDS - t0)) s for ${#pids[@]} processes, ${bad} failed"

    kill "$dmon" 2>/dev/null || true
    wait "$dmon" 2>/dev/null || true

    uptime
    "$PY" ewfix_tests/parse_bench.py \
        --pattern "results_tmp/ewfix_tests/bench/bench_${run}_*" \
        --label "$tag" --H "$H" --nproc "$NPROC" --ngpu "$ngpu" \
        --threads "$OMP" --expect $((NPROC * ngpu)) \
        --dmon "ewfix_tests/bench/dmon_${run}.txt" \
        --append "$CSV" || true
}

echo "################ does the thread count matter? ################"
# One process owns the machine; many processes have to share it. If threading
# helps in the first case and hurts in the second, the packing plan has to say
# so explicitly.
measure 3 1 0 0
measure 3 1 0 1
measure 3 8 0 0
measure 3 8 0 1

echo
echo "################ single card, scaling with process count ################"
for H in 3 11; do
    for N in 1 2 4 8 16; do
        measure "$H" "$N" 0 1
    done
done

echo
echo "################ all four cards ################"
for N in ${FULL_NODE_N:-4 8}; do
    measure 3 "$N" 0,1,2,3 1
    measure 11 "$N" 0,1,2,3 1
done

echo
echo "################ summary ################"
"$PY" ewfix_tests/parse_bench.py --report "$CSV"
