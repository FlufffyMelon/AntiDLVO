#!/bin/bash
# Direct parameter sweep without using Slurm
# Usage: ./run_sweep_direct.sh --rho=0.1,0.2,0.3 [--cores-per-job=4] [--start-core=0] [--max-cores=64]

# Default settings
CORES_PER_JOB=4
START_CORE=0
MAX_CORES=64
PARAM_NAME=""
PARAM_VALUES=()
PREFIX_BASE="lj_muVT_pore"

# Parse command line arguments
for arg in "$@"; do
  case "$arg" in
    --*=*)
      name="${arg%%=*}"; name="${name#--}"
      value="${arg#*=}"

      case "$name" in
        cores-per-job) CORES_PER_JOB="$value" ;;
        start-core) START_CORE="$value" ;;
        max-cores) MAX_CORES="$value" ;;
        *)
          # Assume it's a parameter sweep
          PARAM_NAME="$name"
          IFS=',' read -ra PARAM_VALUES <<< "$value"
          ;;
      esac
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: ./run_sweep_direct.sh --PARAM=VAL1,VAL2,... [--cores-per-job=N] [--start-core=N] [--max-cores=N]"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ -z "$PARAM_NAME" || ${#PARAM_VALUES[@]} -eq 0 ]]; then
  echo "Error: Must specify parameter to sweep with values"
  echo "Example: ./run_sweep_direct.sh --rho=0.1,0.2,0.3,0.4,0.5"
  exit 1
fi

echo "Parameter sweep: $PARAM_NAME = ${PARAM_VALUES[*]}"
echo "Using $CORES_PER_JOB cores per job (starting at core $START_CORE, max $MAX_CORES cores)"

# Launch jobs with automatic core allocation
for i in "${!PARAM_VALUES[@]}"; do
  param_value="${PARAM_VALUES[$i]}"

  # Calculate core range for this job
  base_core=$((START_CORE + (i * CORES_PER_JOB) % (MAX_CORES - START_CORE)))

  # Create CPU list for mpirun binding
  cpu_list=""
  for ((c=0; c<CORES_PER_JOB; c++)); do
    if [[ -n "$cpu_list" ]]; then cpu_list+=","; fi
    cpu_list+="$((base_core + c))"
  done

  # Set job name and prefix
  job_name="${PREFIX_BASE}_${PARAM_NAME}_${param_value}"

  echo "Launching job for ${PARAM_NAME}=${param_value} on CPUs: ${cpu_list}"

  # Launch in background
  ./run_mc.sh --cores="$cpu_list" ${PARAM_NAME}=${param_value} PREFIX=${job_name} &

  # Wait briefly between launches
  sleep 2
done

echo "All jobs launched"
wait
