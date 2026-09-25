#!/bin/bash
#SBATCH --job-name=ew_obs
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=ewald_audit/logs/obs_%j.out
#SBATCH --error=ewald_audit/logs/obs_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
NEW=$(ls -dt results_tmp/ewald_audit/ions_dipole_ewfix_H_3_*/ | head -1)
.venv/bin/python ewald_audit/scripts/observables.py \
    results_prod_neutral/pz_17/sigma/ions_dipole_sigma_pz_17_H_3_sigma_-20_C_0.01_Cz_0.1_20260216_133324 \
    "$NEW" \
    --label old_H3 new_H3 --pz 17 --tag H3
