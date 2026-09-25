#!/usr/bin/env python3
"""Verdict on the stage-1 regression: is the patched src accurate enough?

Reads the JSON files frame_audit.py wrote and applies the acceptance
thresholds from CLAUDE_TASK_ewfix_production.md step 1.4:

  * RMS of the MC energy difference against the converged reference
    <= 1e-3 kT, over every move pool and every frame;
  * the wall force agrees with the reference.

    python ewfix_tests/check_regression.py ewfix_tests/frames

Exit status is non-zero if anything fails, so an sbatch script can gate on it.
"""
from __future__ import annotations

import glob
import json
import os
import sys

RMS_LIMIT = 1.0e-3        # kT, from the task
WALL_REL_LIMIT = 0.02     # prod vs ref wall force


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else "ewfix_tests/frames"
    paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not paths:
        print(f"no audit output in {out_dir}", file=sys.stderr)
        return 2

    print(f"{'run':<26} {'frame':>6} {'worst pool':<18} {'rms/kT':>10} "
          f"{'max/kT':>10} {'f_wall prod':>12} {'ref':>10} {'prod/ref':>9}")
    print("-" * 108)

    failures = []
    for path in paths:
        with open(path) as fh:
            res = json.load(fh)
        tag = res["tag"][:26]

        for fr in res["frames"]:
            moves = fr.get("moves", {})
            # The worst pool is the one that decides the verdict; reporting the
            # mean would hide a single badly-behaved species.
            worst, rms, mx = "-", 0.0, 0.0
            for key, m in moves.items():
                if not m.get("n"):
                    continue
                if m["rms_err_kT"] > rms:
                    worst, rms, mx = key, m["rms_err_kT"], m["max_err_kT"]

            w = fr["wall"]
            ratio = w["prod"] / w["ref"] if w["ref"] else float("nan")
            print(f"{tag:<26} {fr['frame']:>6} {worst:<18} {rms:>10.2e} "
                  f"{mx:>10.2e} {w['prod']:>12.4f} {w['ref']:>10.4f} "
                  f"{ratio:>9.4f}")

            if rms > RMS_LIMIT:
                failures.append(f"{tag} frame {fr['frame']}: RMS {rms:.3e} kT "
                                f"> {RMS_LIMIT:.0e} in pool {worst}")
            if w["ref"] and abs(ratio - 1.0) > WALL_REL_LIMIT:
                failures.append(f"{tag} frame {fr['frame']}: wall force "
                                f"prod/ref = {ratio:.4f}")

    print()
    if failures:
        print("REGRESSION FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"REGRESSION PASSED: RMS <= {RMS_LIMIT:.0e} kT and wall force within "
          f"{WALL_REL_LIMIT:.0%} of the reference in every frame and pool")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
