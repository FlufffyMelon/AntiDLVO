#!/bin/bash
#SBATCH --job-name=python_mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=node2

# Reserve entire node
#SBATCH --exclusive

echo "Starting parameter sweep on node2 at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Change to the working directory
cd /home/fluffymelon/AntiDLVO

# Configuration: Number of parallel jobs to run simultaneously
MAX_PARALLEL_JOBS=49

# Define parameter ranges
max_displacement_values=(0.05 0.1 0.2 0.4 0.6 0.8 1.0)
max_rotation_values=(10 30 60 90 120 150 180)

# Create results directory for this sweep
sweep_dir="results/parameter_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$sweep_dir"

echo "Results will be saved to: $sweep_dir"
echo "=========================================="

# Counter for tracking progress
total_runs=$((${#max_displacement_values[@]} * ${#max_rotation_values[@]}))
current_run=0
completed_runs=0

echo "Running $total_runs simulations with up to $MAX_PARALLEL_JOBS parallel jobs"
echo "=========================================="

# Function to run a single simulation
run_simulation() {
    local max_disp="$1"
    local max_rot="$2"
    local run_num="$3"

    echo "Starting run $run_num/$total_runs: max_displacement=$max_disp, max_rotation=$max_rot at $(date)"

    # Create a unique name for this run
    local run_name="disp_${max_disp}_rot_${max_rot}"

    # Run the simulation with timing and capture output
    local start_time=$(date +%s)
    local output_file="$sweep_dir/run_${run_num}_${run_name}.log"

    time ./run.sh configs/ions_dipole.yaml \
        --sampler.max_displacement "$max_disp" \
        --sampler.max_rotation "$max_rot" \
        --experiment.name "ions_dipole_${run_name}" \
        --experiment.results_dir "$sweep_dir" > "$output_file" 2>&1

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "Completed run $run_num/$total_runs: max_displacement=$max_disp, max_rotation=$max_rot"
    echo "  Duration: ${duration}s, Exit code: $exit_code"

    # Log the run details
    echo "$(date): run=$run_num, max_displacement=$max_disp, max_rotation=$max_rot, duration=${duration}s, exit_code=$exit_code" >> "$sweep_dir/run_log.txt"

    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Run $run_num successful"
    else
        echo "  ✗ Run $run_num failed with exit code $exit_code"
    fi

    return $exit_code
}

# Function to wait for jobs to complete and manage parallel execution
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        # Wait for any job to complete
        wait -n
        completed_runs=$((completed_runs + 1))
        echo "Progress: $completed_runs/$total_runs runs completed"
    done
}

# Generate all parameter combinations
declare -a param_combinations
for max_disp in "${max_displacement_values[@]}"; do
    for max_rot in "${max_rotation_values[@]}"; do
        param_combinations+=("$max_disp $max_rot")
    done
done

# Loop over all parameter combinations with parallel execution
for param_combo in "${param_combinations[@]}"; do
    current_run=$((current_run + 1))

    # Wait for a slot to become available
    wait_for_slot

    # Start the simulation in background
    {
        read -r max_disp max_rot <<< "$param_combo"
        run_simulation "$max_disp" "$max_rot" "$current_run"
    } &

    echo "Launched run $current_run/$total_runs in background"
done

# Wait for all remaining jobs to complete
echo "Waiting for all remaining jobs to complete..."
while [ $(jobs -r | wc -l) -gt 0 ]; do
    wait -n
    completed_runs=$((completed_runs + 1))
    echo "Progress: $completed_runs/$total_runs runs completed"
done

echo ""
echo "=========================================="
echo "Parameter sweep completed at $(date)"
echo "Total runs: $total_runs"
echo "Results saved in: $sweep_dir"
echo "Run log: $sweep_dir/run_log.txt"
echo "Individual run logs: $sweep_dir/run_*_*.log"
echo "=========================================="

# Generate summary statistics
echo "Generating summary statistics..."
successful_runs=$(grep "exit_code=0" "$sweep_dir/run_log.txt" | wc -l)
failed_runs=$((total_runs - successful_runs))

echo "Summary:"
echo "  Successful runs: $successful_runs"
echo "  Failed runs: $failed_runs"
echo "  Success rate: $(( successful_runs * 100 / total_runs ))%"

if [ $failed_runs -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    grep "exit_code=[^0]" "$sweep_dir/run_log.txt" | while read line; do
        echo "  $line"
    done
fi
