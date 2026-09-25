#!/bin/bash
#SBATCH --job-name=zwitter
#SBATCH --cpus-per-task=16
#SBATCH --gpus 1
#SBATCH -w node3

# Set threading to match allocated CPUs to avoid oversubscription
# OpenBLAS will use all available threads by default (up to 64), which can cause
# performance degradation when more threads than CPUs are allocated
CORES=4
export OPENBLAS_NUM_THREADS=$CORES
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export NUMEXPR_NUM_THREADS=$CORES

# Base config to use
CONFIG_FILE="configs/ions_dipole_muVT.yaml"

N_STEPS=10000

PZ="17"
# PZ="27.6"

H=10
SIGMA="-20"
CP="0.1"
CM="0.05"
CZ="0.2"

MU_NA="-5.0"
MU_CL="-5.0"
MU_IP="-5.0"

name="ions_dipole_test_muVT"
results_dir="results_muVT"

source .venv/bin/activate

./run.sh "$CONFIG_FILE" \
        --simulation.n_steps "$N_STEPS" \
        --pz "$PZ" \
        --H "$H" \
        --sigma "$SIGMA" \
        --Cp "$CP" \
        --Cm "$CM" \
        --Cz "$CZ" \
        --simulation.mu.Na "$MU_NA" \
        --simulation.mu.Cl "$MU_CL" \
        --simulation.mu.Ip "$MU_IP" \
        --experiment.name "$name" \
        --experiment.results_dir "$results_dir"
