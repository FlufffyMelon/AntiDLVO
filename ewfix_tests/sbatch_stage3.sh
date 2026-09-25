#!/bin/bash -l
#SBATCH --job-name=ewfix_s3
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=ewfix_tests/logs/stage3_%j.out
#SBATCH --error=ewfix_tests/logs/stage3_%j.out

# Stages 3.1 and 3.2 of CLAUDE_TASK_ewfix_production.md.
#
#   sbatch ewfix_tests/sbatch_stage3.sh
#
# 3.1 builds the starting configuration for the densest points in the set and
# the largest one, and checks counts, overlaps, slabs, charge and energy.
# 3.2 runs the base point with the sampler's own energy audit switched on.
#
# Deliberately NOT the packing benchmark: that one needs the node to itself.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p ewfix_tests/logs results_tmp/ewfix_tests/stage3

module load cuda/12.9
echo "node: $(hostname)"
PY=$PWD/.venv/bin/python
export OMP_NUM_THREADS=8

rc=0

echo
echo "############ 3.1 placement ############"
# The densest systems live at H=3 (the dipoles have the least room), the
# largest at H=11. C_z=0.7 is run at both p_z because the dipole length, and so
# the volume each one sweeps, is set by p_z.
"$PY" ewfix_tests/check_packing.py H=3 Cz=0.7 pz=17.0    || rc=1
"$PY" ewfix_tests/check_packing.py H=3 Cz=0.7 pz=27.6    || rc=1
"$PY" ewfix_tests/check_packing.py H=3 C=0.04            || rc=1
"$PY" ewfix_tests/check_packing.py H=3                   || rc=1
"$PY" ewfix_tests/check_packing.py H=11                  || rc=1
"$PY" ewfix_tests/check_packing.py H=11 pz=17.0          || rc=1

echo
echo "############ 3.2 cached energy and wall force ############"
N_STEPS=${N_STEPS:-2000}
LOG=ewfix_tests/logs/stage3_force_recompute.out
CUDA_VISIBLE_DEVICES=0 "$PY" mc_main.py configs_prod/ions_dipole_ewfix.yaml \
    simulation.n_steps=$N_STEPS \
    sampler.force_recompute=true \
    experiment.results_dir=results_tmp/ewfix_tests/stage3 \
    experiment.name=force_recompute \
    > "$LOG" 2>&1 || { echo "the force_recompute run FAILED"; tail -20 "$LOG"; exit 1; }

RUN=$(ls -dt results_tmp/ewfix_tests/stage3/force_recompute_*/ | head -1)
echo "run: $RUN"
grep -E "^\[EWALD\]" "$LOG" | head -1
"$PY" ewfix_tests/check_force_recompute.py "$LOG" "$RUN" || rc=1

echo
echo "############ verdict ############"
[ $rc -eq 0 ] && echo "STAGE 3.1/3.2 PASSED" || echo "STAGE 3.1/3.2 FAILED"
exit $rc
