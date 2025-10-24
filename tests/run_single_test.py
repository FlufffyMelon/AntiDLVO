#!/usr/bin/env python3
"""
Run a single Ewald test for debugging.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_ewald_tests import EwaldTestRunner


def main():
    """Run a single test."""
    runner = EwaldTestRunner(verbose=True)

    # Run just the first test
    config_path = Path("configs/test_1_two_ions.yaml")
    if config_path.exists():
        result = runner._run_single_test(config_path, 1)
        print("\nTest completed successfully!")
        print(f"L2 error: {result['l2_error']:.6e}")
        print(f"Max error: {result['max_error']:.6e}")
        print(f"Mean error: {result['mean_error']:.6e}")
    else:
        print(f"Configuration file {config_path} not found")


if __name__ == "__main__":
    main()
