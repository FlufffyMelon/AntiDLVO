#!/usr/bin/env python3
"""
Main test runner for all Ewald method validation tests.
Runs all individual test files and generates a comprehensive summary report.
"""

import argparse
import sys
import traceback
from pathlib import Path

# No typing imports needed for this file
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test classes
from test_1_two_ions import TestTwoIons
from test_2_parallel_dipoles import TestParallelDipoles
from test_3_antiparallel_dipoles import TestAntiparallelDipoles
from test_4_coaxial_parallel_dipoles import TestCoaxialParallelDipoles
from test_5_coaxial_antiparallel_dipoles import TestCoaxialAntiparallelDipoles
from test_6_nacl_crystal import TestNaClCrystal
from test_7_periodic_system import TestPeriodicSystem
from test_8_2d_periodic_system import Test2DPeriodicSystem
from test_9_two_ions_columb import TestTwoIonsColumb


class EwaldTestSuite:
    """Main test suite runner for all Ewald validation tests."""

    def __init__(
        self,
        config_dir: str = "configs",
        results_dir: str = "results",
        verbose: bool = False,
    ):
        """Initialize the test suite.

        Args:
            config_dir: Directory containing test configurations
            results_dir: Directory to save results
            verbose: Whether to show detailed output
        """
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.verbose = verbose

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)

        # Define all tests with their configurations
        self.tests = [
            {
                "name": "test_1",
                "class": TestTwoIons,
                "config": "test_1_two_ions.yaml",
                "description": "Two Ions",
            },
            {
                "name": "test_2",
                "class": TestParallelDipoles,
                "config": "test_2_parallel_dipoles.yaml",
                "description": "Parallel Dipoles",
            },
            {
                "name": "test_3",
                "class": TestAntiparallelDipoles,
                "config": "test_3_antiparallel_dipoles.yaml",
                "description": "Antiparallel Dipoles",
            },
            {
                "name": "test_4",
                "class": TestCoaxialParallelDipoles,
                "config": "test_4_coaxial_parallel_dipoles.yaml",
                "description": "Coaxial Parallel Dipoles",
            },
            {
                "name": "test_5",
                "class": TestCoaxialAntiparallelDipoles,
                "config": "test_5_coaxial_antiparallel_dipoles.yaml",
                "description": "Coaxial Antiparallel Dipoles",
            },
            {
                "name": "test_6",
                "class": TestNaClCrystal,
                "config": "test_6_nacl_crystal.yaml",
                "description": "NaCl Crystal",
            },
            {
                "name": "test_7",
                "class": TestPeriodicSystem,
                "config": "test_7_periodic_system.yaml",
                "description": "3D Periodic System",
            },
            {
                "name": "test_8",
                "class": Test2DPeriodicSystem,
                "config": "test_8_2d_periodic_system.yaml",
                "description": "2D Periodic System",
            },
            {
                "name": "test_9",
                "class": TestTwoIonsColumb,
                "config": "test_9_two_ions_columb.yaml",
                "description": "Two Ions Columb",
            },
        ]

        self.results = {}

    def run_all_tests(self):
        """Run all Ewald validation tests."""
        print("=" * 80)
        print("EWALD METHOD VALIDATION TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {self.results_dir}")
        print()

        for i, test_info in enumerate(self.tests, 1):
            config_path = self.config_dir / test_info["config"]

            if not config_path.exists():
                print(
                    f"Warning: Configuration file {config_path} not found, skipping..."
                )
                continue

            print(f"\n{'=' * 60}")
            print(f"RUNNING TEST {i}: {test_info['description']}")
            print(f"{'=' * 60}")

            try:
                # Create test instance
                test_instance = test_info["class"](
                    config_file=str(config_path),
                    results_dir=str(self.results_dir),
                    verbose=self.verbose,
                )

                # Run the test
                result = test_instance.run_test()
                self.results[test_info["name"]] = result

                print(f"✓ Test {i} completed successfully")

            except Exception as e:
                print(f"✗ Test {i} failed: {e}")
                if self.verbose:
                    traceback.print_exc()
                self.results[test_info["name"]] = {"error": str(e)}

        # Generate summary report
        self._generate_summary_report()

        print(f"\n{'=' * 80}")
        print("ALL TESTS COMPLETED")
        print(f"{'=' * 80}")
        print(f"Summary report saved to: {self.results_dir / 'ewald_test_summary.txt'}")

    def run_single_test(self, test_name: str):
        """Run a single test by name.

        Args:
            test_name: Name of the test to run (e.g., 'test_1', 'test_2', etc.)
        """
        test_info = None
        for test in self.tests:
            if test["name"] == test_name:
                test_info = test
                break

        if test_info is None:
            print(f"Error: Test '{test_name}' not found")
            print(f"Available tests: {[t['name'] for t in self.tests]}")
            return

        config_path = self.config_dir / test_info["config"]

        if not config_path.exists():
            print(f"Error: Configuration file {config_path} not found")
            return

        print(f"\n{'=' * 60}")
        print(f"RUNNING TEST: {test_info['description']}")
        print(f"{'=' * 60}")

        try:
            # Create test instance
            test_instance = test_info["class"](
                config_file=str(config_path),
                results_dir=str(self.results_dir),
                verbose=self.verbose,
            )

            # Run the test
            result = test_instance.run_test()

            print(f"\nTest completed successfully!")
            print(f"L2 error: {result['l2_error']:.6e}")
            print(f"Max error: {result['max_error']:.6e}")
            print(f"Mean error: {result['mean_error']:.6e}")

            # Show additional parameters if available
            if "dipole_length" in result:
                print(f"Dipole length: {result['dipole_length']:.3f} nm")
            if "lattice_constant" in result:
                print(f"Lattice constant: {result['lattice_constant']:.3f} nm")
            if "madelung_constant" in result:
                print(f"Madelung constant: {result['madelung_constant']:.6f}")
            if "ewald_energy" in result:
                print(f"Ewald energy: {result['ewald_energy']:.6f}")

        except Exception as e:
            print(f"Test failed: {e}")
            if self.verbose:
                traceback.print_exc()

    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = self.results_dir / "ewald_test_summary.txt"

        with open(report_path, "w") as f:
            f.write("EWALD METHOD VALIDATION TEST SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for test_name, result in self.results.items():
                if "error" in result:
                    f.write(f"{test_name.upper()}: FAILED\n")
                    f.write(f"Error: {result['error']}\n\n")
                    continue

                f.write(f"{test_name.upper()}: {result['test_name']}\n")
                f.write("-" * 30 + "\n")

                # Ewald parameters
                ewald_params = result.get("ewald_params", {})
                f.write("Ewald Parameters:\n")
                for param, value in ewald_params.items():
                    if value is not None:
                        f.write(f"  {param}: {value}\n")
                f.write("\n")

                # Test results
                f.write("Test Results:\n")
                f.write(f"  L2 norm error: {result['l2_error']:.6e}\n")
                f.write(f"  Max absolute error: {result['max_error']:.6e}\n")
                f.write(f"  Mean absolute error: {result['mean_error']:.6e}\n")

                # Additional parameters
                if "dipole_length" in result:
                    f.write(f"  Dipole length: {result['dipole_length']:.3f} nm\n")
                if "lattice_constant" in result:
                    f.write(
                        f"  Lattice constant: {result['lattice_constant']:.3f} nm\n"
                    )
                if "madelung_constant" in result:
                    f.write(f"  Madelung constant: {result['madelung_constant']:.6f}\n")
                if "ewald_energy" in result:
                    f.write(f"  Ewald energy: {result['ewald_energy']:.6f}\n")

                f.write("\n")

        print(f"Summary report saved to: {report_path}")

    def list_tests(self):
        """List all available tests."""
        print("Available Ewald validation tests:")
        print("-" * 40)
        for i, test in enumerate(self.tests, 1):
            print(f"{i}. {test['name']}: {test['description']}")
            print(f"   Config: {test['config']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ewald method validation test suite", add_help=True
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing test configurations",
    )
    parser.add_argument(
        "--results-dir", default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--test", help="Run a single test by name (e.g., test_1, test_2, etc.)"
    )
    parser.add_argument("--list", action="store_true", help="List all available tests")

    args = parser.parse_args()

    try:
        suite = EwaldTestSuite(
            config_dir=args.config_dir,
            results_dir=args.results_dir,
            verbose=args.verbose,
        )

        if args.list:
            suite.list_tests()
        elif args.test:
            suite.run_single_test(args.test)
        else:
            suite.run_all_tests()

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            print("Full traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
