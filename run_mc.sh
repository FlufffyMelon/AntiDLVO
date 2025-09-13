#!/bin/bash
# Direct LAMMPS launcher with explicit core binding
# Usage: ./run_mc.sh --cores=0,1,2,3 [param=value...]

# Default settings
CORES=""
VAR_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cores=*)
      CORES="${1#*=}"
      shift;;
    -cores|--cores)
      [[ $# -ge 2 ]] || { echo "Error: $1 requires a value"; exit 1; }
      CORES="$2"
      shift 2;;
    -var|--var)
      [[ $# -ge 3 ]] || { echo "Error: $1 requires <name> <value>"; exit 1; }
      if [[ "$2" == "PREFIX" ]]; then PREFIX="$3"; fi
      VAR_ARGS+=( -var "$2" "$3" )
      shift 3;;
    --*=*)
      name="${1%%=*}"; name="${name#--}"; val="${1#*=}"
      if [[ "$name" == "PREFIX" ]]; then PREFIX="$val"; fi
      VAR_ARGS+=( -var "$name" "$val" )
      shift;;
    -*)
      name="${1#-}"; [[ $# -ge 2 ]] || { echo "Error: $1 requires a value"; exit 1; }
      if [[ "$name" == "PREFIX" ]]; then PREFIX="$2"; fi
      VAR_ARGS+=( -var "$name" "$2" )
      shift 2;;
    *=*)
      name="${1%%=*}"; val="${1#*=}"
      if [[ "$name" == "PREFIX" ]]; then PREFIX="$val"; fi
      VAR_ARGS+=( -var "$name" "$val" )
      shift;;
    *)
      echo "Ignoring arg: $1"
      shift;;
  esac
done

# Validate core list
if [[ -z "$CORES" ]]; then
  echo "Error: Must specify cores with --cores=0,1,2,3"
  echo "Usage: ./run_mc.sh --cores=0,1,2,3 [param=value...]"
  exit 1
fi

# Count number of cores
CORE_COUNT=$(echo "$CORES" | tr ',' '\n' | wc -l)
echo "Using $CORE_COUNT cores: $CORES"

# Set up results directory
PREFIX=${PREFIX:-lj_nvt}
TS=$(date +%Y%m%d_%H%M%S)
RUNDIR="results/${PREFIX}_${TS}"
mkdir -p "$RUNDIR"

echo "Running LAMMPS with parameters: ${VAR_ARGS[@]}"
echo "Results directory: $RUNDIR"

# Run LAMMPS with explicit core binding - redirect all output to log file
export OMP_NUM_THREADS=1
mpirun -np $CORE_COUNT --cpu-list $CORES --bind-to core \
  lmp-mc -in in.ionic_dipolar_pore_mc -var outdir "$RUNDIR" -screen none -log "$RUNDIR/output.log" "${VAR_ARGS[@]}" \
  > /dev/null 2>&1

echo "Run completed. Results in: $RUNDIR"
