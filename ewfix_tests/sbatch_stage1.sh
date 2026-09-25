#!/bin/bash -l
#SBATCH --job-name=ewfix_s1
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=ewfix_tests/logs/stage1_%j.out
#SBATCH --error=ewfix_tests/logs/stage1_%j.out

# Stage 1.4 of CLAUDE_TASK_ewfix_production.md: prove the PATCHED src/ (not
# ewald_audit/codefix/) reproduces the audit's accuracy on this cluster.
#
#   sbatch ewfix_tests/sbatch_stage1.sh
#
# The old production trajectories the audit used live on the previous cluster,
# so the frames come from short fresh runs with the corrected config instead --
# the comparison is against an independently converged reference either way, so
# the frames only need to be physically sensible configurations, not equilibrated
# ones.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# A fresh directory per job: check_regression globs the whole directory, so a
# rerun must not be judged on the previous run's frames.
FRAMES=${FRAMES:-ewfix_tests/frames_${SLURM_JOB_ID:-manual}}
mkdir -p ewfix_tests/logs "$FRAMES" results_tmp/ewfix_tests

module load cuda/12.9
echo "node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
PY=$PWD/.venv/bin/python

echo
echo "############ 1. unit tests on the patched src ############"
( cd tests && "$PY" run_all_tests.py ) 2>&1 | tail -40

echo
echo "############ 2. short runs at H = 3, 5, 11 ############"
N_STEPS=${N_STEPS:-5000}
HS=(3 5 11)
pids=()
for i in "${!HS[@]}"; do
    H=${HS[$i]}
    CUDA_VISIBLE_DEVICES=$i "$PY" mc_main.py configs_prod/ions_dipole_ewfix.yaml \
        H=$H \
        simulation.n_steps=$N_STEPS \
        experiment.results_dir=results_tmp/ewfix_tests \
        experiment.name=verify_H_$H \
        > ewfix_tests/logs/verify_run_H$H.out 2>&1 &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "H=${HS[$i]} run FAILED" >&2; rc=1; }
done
[ $rc -eq 0 ] || { echo "short runs failed, aborting"; exit 1; }

for H in "${HS[@]}"; do
    echo "--- H=$H k-grid and placement"
    grep -E "^\[EWALD\]|Insertion order|RANDOM_HS" ewfix_tests/logs/verify_run_H$H.out | head -4
done

echo
echo "############ 3. regression vs converged reference ############"
pids=()
i=0
for H in "${HS[@]}"; do
    RUN=$(ls -dt results_tmp/ewfix_tests/verify_H_${H}_*/ | head -1)
    echo "H=$H -> $RUN"
    CUDA_VISIBLE_DEVICES=$i "$PY" ewald_audit/scripts/frame_audit.py "$RUN" \
        --frames 40 70 --n-moves 250 --gpu 1 --out "$FRAMES" \
        --tag-prefix "ewfix_H${H}_" \
        > ewfix_tests/logs/verify_audit_H$H.out 2>&1 &
    pids+=($!)
    i=$((i + 1))
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "frame_audit ${HS[$i]} FAILED" >&2; rc=1; }
done

echo
echo "############ 4. verdict ############"
"$PY" ewfix_tests/check_regression.py "$FRAMES"
