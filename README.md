# Monte Carlo Simulation Framework

A comprehensive Python framework for performing Monte Carlo simulations with support for multiple ensembles (NVT, NPT, muVT), force fields (Lennard-Jones, Coulomb), and unit systems (standard Gromacs units and Lennard-Jones reduced units).

## Features

- **Multiple Ensembles**: NVT, NPT, and Grand Canonical (muVT) Monte Carlo
- **Force Fields**: Lennard-Jones 12-6 potential and Coulomb electrostatics
- **Unit Systems**: Standard units (Gromacs-compatible) and Lennard-Jones reduced units
- **Flexible Configuration**: YAML-based configuration system
- **Comprehensive Logging**: Detailed simulation logs and XYZ trajectory output
- **Modular Design**: Clean, extensible architecture with strong typing

## Architecture

The framework consists of several key components:

### Core Classes

1. **System**: Manages atoms, geometry, and ensemble parameters
   - Dynamic array resizing for efficient memory usage
   - Periodic boundary conditions support
   - Ensemble validation and parameter checking

2. **Topology**: Handles force field interactions and energy calculations
   - Flexible interaction mapping between atom types
   - Efficient energy calculation methods for MC moves
   - Support for multiple force components

3. **Sampler**: Performs Monte Carlo moves based on ensemble type
   - Translation, insertion, deletion, and volume change moves
   - Metropolis acceptance criterion
   - Configurable action probabilities

4. **Forces**: Base force class with LJ and Coulomb implementations
   - Abstract base class for extensibility
   - Proper unit handling and cutoff support
   - Mixing rules for cross-interactions

5. **Units**: Unit management with pint integration
   - Support for both standard and LJ unit systems
   - Automatic unit conversion and validation

6. **Logger**: Comprehensive logging and trajectory output
   - Structured log files with simulation statistics
   - XYZ format trajectory files for visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AntiDLVO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or use the automated setup:
```bash
./run.sh
```

## Usage

### Quick Start

Run a simulation with the default configuration:
```bash
./run.sh
```

### Custom Configuration

Run with a specific configuration file:
```bash
./run.sh configs/my_simulation.yaml
```

Or directly with Python:
```bash
python3 mc_main.py configs/main.yaml
```

### Configuration Files

The framework uses YAML configuration files. Here's the structure:

```yaml
# Main simulation parameters
simulation:
  ensemble: "NVT"          # NVT, NPT, or muVT
  temperature: 300.0       # K
  pressure: 1.0           # bar (NPT only)
  mu: -10.0               # chemical potential (muVT only)
  n_steps: 10000

# System setup
system:
  n_atoms: 100
  initial_capacity: 128
  box: [10.0, 10.0, 10.0] # nm
  pbc: [true, true, true]

# Force field
topology:
  lj_params_file: "configs/lj_params/argon.yaml"
  coulomb_enabled: false
  cutoff: 2.5             # nm

# Monte Carlo sampling
sampler:
  actions:
    - action: "translate"
      probability: 0.8
    - action: "insert"     # muVT only
      probability: 0.1
    - action: "delete"     # muVT only
      probability: 0.1
  max_displacement: 0.1   # nm

# Output
logger:
  log_file: "simulation.log"
  xyz_file: "trajectory.xyz"
  log_interval: 100
  xyz_interval: 1000
```

### Force Field Parameters

LJ parameters are defined in separate YAML files:

```yaml
# configs/lj_params/argon.yaml
lj_parameters:
  Ar:
    sigma: 0.3405     # nm
    epsilon: 0.9959   # kJ/mol
    mass: 39.948      # amu
    charge: 0.0
```

## Examples

### NVT Simulation of Argon
```bash
# Edit configs/main.yaml to set ensemble: "NVT"
./run.sh configs/main.yaml
```

### Grand Canonical (muVT) Simulation
```yaml
simulation:
  ensemble: "muVT"
  temperature: 120.0
  mu: -10.0

sampler:
  actions:
    - action: "translate"
      probability: 0.7
    - action: "insert"
      probability: 0.15
    - action: "delete"
      probability: 0.15
```

### NPT Simulation
```yaml
simulation:
  ensemble: "NPT"
  temperature: 300.0
  pressure: 1.0

sampler:
  actions:
    - action: "translate"
      probability: 0.9
    - action: "volume"
      probability: 0.1
```

## Output Files

- **simulation.log**: Detailed simulation log with energies, acceptance ratios, and system properties
- **trajectory.xyz**: Atomic positions in XYZ format for visualization

## Unit Systems

### Standard Units (Gromacs-compatible)
- Length: nm
- Energy: kJ/mol
- Mass: amu
- Time: ps
- Temperature: K
- Pressure: bar

### Lennard-Jones Reduced Units
- All quantities expressed in terms of LJ sigma, epsilon, and mass
- Convenient for comparing with literature results

## Extending the Framework

### Adding New Force Fields

1. Inherit from the `Force` base class
2. Implement the `__call__` method
3. Add to topology configuration

```python
class MyForce(Force):
    def __call__(self, pos1, pos2, type1, type2, **kwargs):
        # Implement your force calculation
        return energy
```

### Adding New MC Moves

1. Add move to `Sampler` class
2. Update action validation
3. Implement acceptance criterion

## Dependencies

- numpy: Numerical computations
- pint: Unit handling
- pyyaml: Configuration file parsing
- typing_extensions: Enhanced type hints

## License

This project is open source. Please see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this framework in your research, please cite:

```
Monte Carlo Simulation Framework
https://github.com/your-repo/AntiDLVO
```
