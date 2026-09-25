#!/bin/bash
#SBATCH --job-name=ew_vfix
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=ewald_audit/logs/vfix_%j.out
#SBATCH --error=ewald_audit/logs/vfix_%j.out

# Does the PROPOSED parameter set actually hit the accuracy target?
# Same frames/metrics as the step-4 audit, but with ewald_audit/codefix and the
# corrected Ewald parameters.  CPU only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export AUDIT_MEM_BUDGET=2.0e8

.venv/bin/python ewald_audit/scripts/verify_fix.py \
    results_prod_neutral/pz_17/sigma/ions_dipole_sigma_pz_17_H_3_sigma_-20_C_0.01_Cz_0.1_20260216_133324 \
    results_prod_neutral/pz_17/H/ions_dipole_H_pz_17_H_5_sigma_-20_C_0.01_Cz_0.1_20260216_125433 \
    results_prod_neutral/pz_17/H/ions_dipole_H_pz_17_H_11_sigma_-20_C_0.01_Cz_0.1_20260216_125433 \
    --frames 2000 --n-moves 120 --gpu 0
