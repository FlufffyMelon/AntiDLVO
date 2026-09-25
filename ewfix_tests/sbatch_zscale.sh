#!/bin/bash -l
#SBATCH --job-name=ewfix_zs
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodelist=node211-27
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=ewfix_tests/logs/zscale_%j.out
#SBATCH --error=ewfix_tests/logs/zscale_%j.out

# How much vacuum does the slab correction need?
#
#   sbatch ewfix_tests/sbatch_zscale.sh
#
# Stage 1 found the corrected config just misses the 1e-3 kT acceptance limit at
# H = 5 (1.13e-3), while H = 3 sits at 1.4e-4 and H = 11 at 4.9e-6. The audit's
# rule -- z_scale = 8 below H = 4, else 4 -- fixes the thin-slit case but leaves
# H = 5 with the *thinnest* vacuum gap of the whole production set: 15 nm,
# against 21 nm at H = 3 and 33 nm at H = 11. The residual error tracks that gap.
#
# So scan the gap directly at the H values where the rule bites, and pick the
# rule from the measurement rather than from a round number.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p ewfix_tests/logs ewfix_tests/frames_zs results_tmp/ewfix_tests/zs

module load cuda/12.9
echo "node: $(hostname)"
PY=$PWD/.venv/bin/python
N_STEPS=${N_STEPS:-5000}

# H, z_scale -- the resulting vacuum gap is H*(z_scale-1).
POINTS=(
    "5 6"    # gap 25 nm
    "5 8"    # gap 35 nm
    "7 4"    # gap 21 nm, the rule as it stands
    "7 6"    # gap 35 nm
    "9 4"    # gap 27 nm, the rule as it stands
    "3 6"    # gap 15 nm -- does a thin gap hurt at H=3 too?
)

echo
echo "############ short runs ############"
pids=()
i=0
for pt in "${POINTS[@]}"; do
    set -- $pt; H=$1; ZS=$2
    CUDA_VISIBLE_DEVICES=$((i % 4)) "$PY" mc_main.py \
        configs_prod/ions_dipole_ewfix.yaml \
        H=$H ewald.z_scale_factor=$ZS \
        simulation.n_steps=$N_STEPS \
        experiment.results_dir=results_tmp/ewfix_tests/zs \
        experiment.name="zs_H${H}_zs${ZS}" \
        > "ewfix_tests/logs/zs_H${H}_zs${ZS}.out" 2>&1 &
    pids+=($!)
    i=$((i + 1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] || { echo "a short run FAILED"; exit 1; }

for pt in "${POINTS[@]}"; do
    set -- $pt; H=$1; ZS=$2
    echo "--- H=$H z_scale=$ZS"
    grep -E "^\[EWALD\]" "ewfix_tests/logs/zs_H${H}_zs${ZS}.out" | head -1
done

echo
echo "############ audits ############"
pids=()
i=0
for pt in "${POINTS[@]}"; do
    set -- $pt; H=$1; ZS=$2
    RUN=$(ls -dt results_tmp/ewfix_tests/zs/zs_H${H}_zs${ZS}_*/ | head -1)
    CUDA_VISIBLE_DEVICES=$((i % 4)) "$PY" ewald_audit/scripts/frame_audit.py "$RUN" \
        --frames 40 70 --n-moves 250 --gpu 1 --out ewfix_tests/frames_zs \
        --tag-prefix "zs_H${H}_zs${ZS}_" \
        > "ewfix_tests/logs/zs_audit_H${H}_zs${ZS}.out" 2>&1 &
    pids+=($!)
    i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done

echo
echo "############ verdict ############"
"$PY" ewfix_tests/check_regression.py ewfix_tests/frames_zs || true
echo "(a non-zero verdict above is expected for the points kept as controls)"
