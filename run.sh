#!/bin/bash

# Monte Carlo Simulation Runner Script

# Default configuration file
DEFAULT_CONFIG="configs/main.yaml"

# Function to display help
show_help() {
    echo "Usage: $0 [CONFIG_FILE] [OVERRIDES...]"
    echo ""
    echo "Run Monte Carlo simulations with the specified configuration."
    echo ""
    echo "Arguments:"
    echo "  CONFIG_FILE    Path to YAML configuration file (default: $DEFAULT_CONFIG)"
    echo "  OVERRIDES      Optional overrides as dotlist or --key value (e.g., system.size=20 or --system.size 20)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run with default config"
    echo "  $0 configs/main.yaml         # Run with specific config"
    echo "  $0 configs/lj_nvt.yaml --system.size 20 --simulation.n_steps 5000"
}

# Function to run simulation
run_simulation() {
    local config_file="$1"
    shift
    local overrides=("$@")

    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo "Error: Configuration file '$config_file' not found."
        echo "Available configs in configs/:"
        find configs -name "*.yaml" 2>/dev/null || echo "  No config files found"
        exit 1
    fi

    echo "Running Monte Carlo simulation with config: $config_file"
    echo "Starting at $(date)"
    echo "=========================================="

    # Run the simulation with overrides passed through
    python3 mc_main.py "$config_file" ${overrides[@]}

    local exit_code=$?

    echo "=========================================="
    echo "Finished at $(date)"

    if [ $exit_code -eq 0 ]; then
        echo "Simulation completed successfully!"
    else
        echo "Simulation failed with exit code $exit_code"
        exit $exit_code
    fi
}

# Main script logic
main() {
    # Check for help flag
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_help
        exit 0
    fi

    # Determine config file
    local config_file="${1:-$DEFAULT_CONFIG}"
    shift || true

    # Remaining args are overrides
    run_simulation "$config_file" "$@"
}

# Check if running directly (not sourced)
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
