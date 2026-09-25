#!/bin/bash -l
#SBATCH --job-name=ewfix_prod
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=7-00:00:00
#SBATCH --output=results_ewfix/logs/prod_%j.out
#SBATCH --error=results_ewfix/logs/prod_%j.out

# Stage 4 of CLAUDE_TASK_ewfix_production.md: the 32 production runs.
#
#   mkdir -p results_ewfix/logs && sbatch run_ewfix_production.sh
#   PER_GPU=4 sbatch run_ewfix_production.sh      # if the node is shared
#
# One slurm job holds the whole node, because the cluster caps a user at 20
# queued jobs and 32 separate submissions would not fit. Inside it, each run is
# an ordinary mc_main.py process pinned to one card with CUDA_VISIBLE_DEVICES.
#
# Which run goes on which card is not decided here: ewfix_tests/plan_packing.py
# deals them out from a measured cost model, balancing the four cards to within
# half a percent and spreading the expensive H=9, H=11 and C_z=0.7 points across
# different cards, as the task requires. This script only executes that plan.
#
# The benchmark (ewfix_tests/sbatch_bench.sh) found the per-process rate flat
# from one to eight processes on a card, so PER_GPU=8 puts all 32 runs in
# flight at once, each at very nearly its solo speed. Each process is pinned to
# a single BLAS thread: 32 processes on 128 cores have four cores each, and a
# BLAS that still thinks it owns the machine would spend them fighting.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

PY=$PWD/.venv/bin/python
CONFIG=configs_prod/ions_dipole_ewfix.yaml
OUT=results_ewfix
PER_GPU=${PER_GPU:-8}
N_GPU=${N_GPU:-4}
N_STEPS=${N_STEPS:-1000000}

mkdir -p "$OUT/logs" "$OUT/manifest.d"
module load cuda/12.9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

echo "node      : $(hostname)   cores: $(nproc)"
echo "job       : ${SLURM_JOB_ID:-manual}"
echo "started   : $(date -Is)"
echo "config    : $CONFIG"
echo "packing   : $PER_GPU processes per card on $N_GPU cards"
echo "n_steps   : $N_STEPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

PLAN="$OUT/plan_${SLURM_JOB_ID:-manual}.tsv"
"$PY" ewfix_tests/plan_packing.py --tsv > "$PLAN"
echo "plan      : $PLAN ($(wc -l < "$PLAN") runs)"
echo

# ---------------------------------------------------------------- one run ---
launch() {
    local gpu=$1 name=$2 results_dir=$3 overrides=$4
    local log="$OUT/logs/${name}.out"
    local row="$OUT/manifest.d/${name}.csv"
    local t0 t1 rc run_dir steps

    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] gpu$gpu start $name"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu "$PY" mc_main.py "$CONFIG" \
        $overrides \
        simulation.n_steps=$N_STEPS \
        experiment.name="$name" \
        experiment.results_dir="$results_dir" \
        > "$log" 2>&1
    rc=$?
    t1=$(date +%s)

    # The runner appends its own timestamp, so the directory is found, not named.
    run_dir=$(ls -dt "$results_dir/${name}_"*/ 2>/dev/null | head -1)
    run_dir=${run_dir%/}
    steps=0
    if [ -n "$run_dir" ] && [ -f "$run_dir/simulation_data.csv" ]; then
        steps=$(tail -1 "$run_dir/simulation_data.csv" | cut -d, -f1)
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$name" "$gpu" "$results_dir" "${run_dir:-MISSING}" "$rc" \
        "$steps" "$t0" "$t1" "$(( t1 - t0 ))" > "$row"

    echo "[$(date +%H:%M:%S)] gpu$gpu done  $name  rc=$rc  steps=$steps  $(( (t1 - t0) / 60 )) min"
    return 0
}

# ------------------------------------------------------------ worker slots ---
# Each (card, slot) pair reads the plan itself and takes every PER_GPU-th of its
# card's runs. That is a static split rather than a shared queue on purpose: a
# pipe shared by several `read`ers is not safe, because bash reads it a byte at
# a time and the bytes of two lines interleave between processes -- a dry run of
# the FIFO version turned 32 runs into 36 shredded names. Here each subshell
# opens $PLAN on its own descriptor and no two workers share any state.
#
# At the default PER_GPU=8 the split is exact anyway: plan_packing.py deals 8
# runs to each card, so every slot gets exactly one and nothing waits.
n_runs=$(grep -c . "$PLAN")
echo "queued $n_runs runs, starting $((N_GPU * PER_GPU)) workers"
echo

pids=()
for g in $(seq 0 $((N_GPU - 1))); do
    for s in $(seq 1 "$PER_GPU"); do
        (
            idx=0
            while IFS=$'\t' read -r gpu name results_dir overrides; do
                [ "${gpu:-}" = "$g" ] || continue
                if [ $(( idx % PER_GPU )) -eq $(( s - 1 )) ]; then
                    launch "$g" "$name" "$results_dir" "$overrides"
                fi
                idx=$((idx + 1))
            done < "$PLAN"
        ) &
        pids+=($!)
    done
done

for p in "${pids[@]}"; do wait "$p"; done

# --------------------------------------------------------------- manifest ---
MANIFEST="$OUT/manifest.csv"
{
    echo "name,gpu,results_dir,run_dir,exit_code,last_step,t_start,t_end,seconds"
    cat "$OUT"/manifest.d/*.csv 2>/dev/null | sort
} > "$MANIFEST"
echo
echo "manifest  : $MANIFEST"

# ---------------------------------------------- base point in every sweep ---
# The sweeps all contain the base point. Running it four more times per p_z
# would burn GPU-days and hand the analysis four differently seeded "base
# points", so each sweep gets a symlink to the one base run instead.
"$PY" ewfix_tests/run_list.py --links --tsv | while IFS=$'\t' read -r link_dir link_name target_dir; do
    target=$(ls -dt "$target_dir/ions_dipole_base_"*/ 2>/dev/null | head -1)
    if [ -z "$target" ]; then
        echo "  no base run under $target_dir, skipping $link_name"
        continue
    fi
    mkdir -p "$link_dir"
    ln -sfn "$(realpath --relative-to="$link_dir" "${target%/}")" \
        "$link_dir/$link_name"
    echo "  $link_dir/$link_name -> $(readlink "$link_dir/$link_name")"
done

echo
echo "finished  : $(date -Is)"
failed=$(awk -F, 'NR>1 && $5 != 0' "$MANIFEST" | wc -l)
short=$(awk -F, -v n="$N_STEPS" 'NR>1 && $6+0 < n' "$MANIFEST" | wc -l)
echo "runs      : $(( $(wc -l < "$MANIFEST") - 1 )), failed $failed, short of $N_STEPS steps $short"
[ "$failed" -eq 0 ] && [ "$short" -eq 0 ]
