#!/bin/bash
# Run multiple density simulations on different cores

# Base directory
BASE_DIR="/home/fluffymelon/AntiDLVO"

# Define density values to simulate
RHO_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Define core ranges for each job (4 cores per job, spread across node)
# Format: start_core-end_core
CORE_RANGES=(
  "0-3"    # Job 1: cores 0-3
  "4-7"    # Job 2: cores 4-7
  "8-11"   # Job 3: cores 8-11
  "12-15"  # Job 4: cores 12-15
  "16-19"  # Job 5: cores 16-19
  "20-23"  # Job 6: cores 20-23
  "24-27"  # Job 7: cores 24-27
  "28-31"  # Job 8: cores 28-31
  "32-35"  # Job 9: cores 32-35
)

# Check if we have enough core ranges defined
if [ ${#RHO_VALUES[@]} -gt ${#CORE_RANGES[@]} ]; then
  echo "Error: Not enough core ranges defined for all density values"
  exit 1
fi

# Submit jobs with explicit core assignments
for i in "${!RHO_VALUES[@]}"; do
  if [ $i -lt ${#CORE_RANGES[@]} ]; then
    rho="${RHO_VALUES[$i]}"
    cores="${CORE_RANGES[$i]}"

    echo "Submitting job for rho=$rho on cores $cores"

    # Submit with explicit core binding
    sbatch \
      --job-name="lj_rho_${rho}" \
      --nodelist=node2 \
      --nodes=1 \
      --ntasks=4 \
      --cpus-per-task=1 \
      --cpu_bind=map_cpu:${cores//[-,]/:} \
      --output="slurm_%j.out" \
      --error="slurm_%j.err" \
      --wrap="./run_mc.sh rho=$rho PREFIX=lj_nvt_rho_${rho}"

    # Wait briefly between submissions
    sleep 1
  fi
done

echo "All jobs submitted"
