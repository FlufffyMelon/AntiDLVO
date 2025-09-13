#!/bin/bash
# Flexible parameter sweep with automatic core allocation
# Usage: ./run_sweep.sh [options]
#   --var=VALUE_LIST   : Comma-separated list of values for variable 'var'
#   --cores-per-job=N  : Number of cores to allocate per job (default: 4)
#   --node=NODENAME    : Node to run on (default: node2)
#   --max-cores=N      : Maximum cores to use (default: 64)

# Default settings
NODE="node2"
CORES_PER_JOB=4
MAX_CORES=64
PARAM_NAME=""
PARAM_VALUES=()
PREFIX_BASE="lj_nvt"

# Parse command line arguments
for arg in "$@"; do
  case "$arg" in
    --*=*)
      name="${arg%%=*}"; name="${name#--}"
      value="${arg#*=}"

      case "$name" in
        cores-per-job) CORES_PER_JOB="$value" ;;
        node) NODE="$value" ;;
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
      echo "Usage: ./run_sweep.sh --PARAM=VAL1,VAL2,... [--cores-per-job=N] [--node=NAME] [--max-cores=N]"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ -z "$PARAM_NAME" || ${#PARAM_VALUES[@]} -eq 0 ]]; then
  echo "Error: Must specify parameter to sweep with values"
  echo "Example: ./run_sweep.sh --rho=0.1,0.2,0.3,0.4,0.5"
  exit 1
fi

# Calculate how many jobs can run in parallel
MAX_JOBS=$((MAX_CORES / CORES_PER_JOB))
if [[ ${#PARAM_VALUES[@]} -gt $MAX_JOBS ]]; then
  echo "Warning: More parameter values (${#PARAM_VALUES[@]}) than available job slots ($MAX_JOBS)"
  echo "Some jobs will be queued"
fi

echo "Parameter sweep: $PARAM_NAME = ${PARAM_VALUES[*]}"
echo "Using $CORES_PER_JOB cores per job on $NODE (max $MAX_CORES cores)"

# Submit jobs with automatic core allocation
for i in "${!PARAM_VALUES[@]}"; do
  param_value="${PARAM_VALUES[$i]}"

  # Calculate core range for this job
  start_core=$((i * CORES_PER_JOB % MAX_CORES))

  # Create CPU list for mpirun binding
  cpu_list=""
  for ((c=0; c<CORES_PER_JOB; c++)); do
    if [[ -n "$cpu_list" ]]; then cpu_list+=","; fi
    cpu_list+="$((start_core + c))"
  done

  # Set job name and prefix
  job_name="${PREFIX_BASE}_${PARAM_NAME}_${param_value}"

  echo "Submitting job for ${PARAM_NAME}=${param_value} on CPUs: ${cpu_list}"

  # Create a custom run script for this specific job with CPU binding
  tmp_script="tmp_run_${PARAM_NAME}_${param_value}.sh"
  cat > "$tmp_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodelist=${NODE}
#SBATCH --nodes=1
#SBATCH --ntasks=${CORES_PER_JOB}
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

export OMP_NUM_THREADS=1
mpirun --cpu-list ${cpu_list} --bind-to core --report-bindings \\
  lmp_mc -in in.lj_mc -var outdir "results/${job_name}_\$(date +%Y%m%d_%H%M%S)" \\
  -var ${PARAM_NAME} ${param_value} -var PREFIX ${job_name}

# Move Slurm outputs to results directory
RUNDIR=\$(ls -td results/${job_name}_* | head -1)
mv "slurm_\${SLURM_JOB_ID}.out" "\$RUNDIR/" 2>/dev/null || true
mv "slurm_\${SLURM_JOB_ID}.err" "\$RUNDIR/" 2>/dev/null || true
rm -f "$tmp_script"
EOF

  chmod +x "$tmp_script"

  # Submit the job
  sbatch "$tmp_script"

  # Wait briefly between submissions
  sleep 1
done

echo "All jobs submitted"
