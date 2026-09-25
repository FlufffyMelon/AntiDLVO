#!/bin/bash -l
#SBATCH --job-name=mc_smoke
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=mipt/logs/smoke_%j.out
#SBATCH --error=mipt/logs/smoke_%j.out

# Does the production code actually run on mipt?
#
#   sbatch mipt/sbatch_smoke.sh
#
# Takes the whole GPU complement of node211-27 and starts one short run per
# V100, each pinned to its own device.  That exercises the layout production
# will use (several independent processes sharing a node), not just a single
# import, and finishes in a few minutes.
#
# Writes to results_smoke_mipt/, which .gitignore's `results*/` keeps out of
# the repository.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p mipt/logs

module load cuda/12.9
echo "node: $(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

N_STEPS=${N_STEPS:-2000}
OUT=results_smoke_mipt

pids=()
for i in 0 1 2 3; do
    # non_static_init.mode=random_hs: the production 'random' placement fails
    # stochastically at this packing fraction; random_hs is the documented
    # fallback and only affects the starting configuration, not the physics.
    CUDA_VISIBLE_DEVICES=$i .venv/bin/python mc_main.py \
        configs_prod/ions_dipole.yaml \
        simulation.n_steps=$N_STEPS \
        system.non_static_init.mode=random_hs \
        experiment.results_dir=$OUT \
        experiment.name=smoke_gpu$i \
        > mipt/logs/smoke_gpu$i.out 2>&1 &
    pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "gpu $i: exit 0"
    else
        echo "gpu $i: FAILED (exit $?)" >&2
        rc=1
    fi
done

echo
echo "=== per-GPU tails ==="
for i in 0 1 2 3; do
    echo "--- gpu $i"
    tail -n 5 mipt/logs/smoke_gpu$i.out
done

echo
echo "=== produced trajectories ==="
for d in $OUT/smoke_gpu*/; do
    f="$d/simulation_data.csv"
    [ -f "$f" ] || { echo "$d: NO simulation_data.csv" >&2; rc=1; continue; }
    echo "$d: $(( $(wc -l < "$f") - 1 )) logged steps, last row:"
    tail -n 1 "$f" | cut -c1-160
done

[ $rc -eq 0 ] && echo "=== SMOKE TEST PASSED ===" || echo "=== SMOKE TEST FAILED ===" >&2
exit $rc
