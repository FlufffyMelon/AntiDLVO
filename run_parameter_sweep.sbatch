#!/bin/bash
#SBATCH --job-name=ions_dipole_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodelist=node2
#SBATCH --output=parameter_sweep_%j.out
#SBATCH --error=parameter_sweep_%j.err

# Reserve entire node
#SBATCH --exclusive

echo "Starting parameter sweep on node2 at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Change to the working directory
cd /home/fluffymelon/AntiDLVO

# Define parameter ranges
max_displacement_values=(0.1 0.15 0.2 0.25 0.3 0.35 0.4)
max_rotation_values=(15 20 25 30 35 40 45)

# Create results directory for this sweep
sweep_dir="results/parameter_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$sweep_dir"

echo "Results will be saved to: $sweep_dir"
echo "=========================================="

# Counter for tracking progress
total_runs=$((${#max_displacement_values[@]} * ${#max_rotation_values[@]}))
current_run=0

# Loop over all parameter combinations
for max_disp in "${max_displacement_values[@]}"; do
    for max_rot in "${max_rotation_values[@]}"; do
        current_run=$((current_run + 1))

        echo ""
        echo "Run $current_run/$total_runs: max_displacement=$max_disp, max_rotation=$max_rot"
        echo "Started at: $(date)"
        echo "----------------------------------------"

        # Create a unique name for this run
        run_name="disp_${max_disp}_rot_${max_rot}"

        # Run the simulation with timing
        time ./run.sh configs/ions_dipole.yaml \
            --sampler.max_displacement "$max_disp" \
            --sampler.max_rotation "$max_rot" \
            --experiment.name "ions_dipole_${run_name}"

        exit_code=$?

        echo "----------------------------------------"
        echo "Run $current_run/$total_runs completed at: $(date)"
        echo "Exit code: $exit_code"

        if [ $exit_code -eq 0 ]; then
            echo "✓ Run $current_run/$total_runs successful"
        else
            echo "✗ Run $current_run/$total_runs failed with exit code $exit_code"
        fi

        # Log the run details
        echo "$(date): max_displacement=$max_disp, max_rotation=$max_rot, exit_code=$exit_code" >> "$sweep_dir/run_log.txt"

        # Brief pause between runs
        sleep 2
    done
done

echo ""
echo "=========================================="
echo "Parameter sweep completed at $(date)"
echo "Total runs: $total_runs"
echo "Results saved in: $sweep_dir"
echo "Run log: $sweep_dir/run_log.txt"
echo "=========================================="
