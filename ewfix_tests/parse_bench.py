#!/usr/bin/env python3
"""Turn a benchmark point into one row of ewfix_tests/bench/bench.csv.

The rate comes from simulation_data.csv rather than from tqdm: tqdm's figure
includes the placement and the first, unrepresentative steps, and it is lost
once stdout is redirected to a file.

    python ewfix_tests/parse_bench.py --pattern 'results_tmp/.../bench_X_*' \
        --label X --H 3 --nproc 4 --ngpu 1 --append ewfix_tests/bench/bench.csv
    python ewfix_tests/parse_bench.py --report ewfix_tests/bench/bench.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

HOURS_1E6 = 1_000_000 / 3600.0  # steps -> (steps/s) -> hours, divided by rate

FIELDS = ("label", "H", "nproc", "ngpu", "threads", "n_proc_total", "rate_mean",
          "rate_min", "rate_max", "rate_total", "hours_1e6_slowest",
          "gpu_util_pct", "gpu_mem_mb", "n_parsed")


def rate_of(run_dir: str, warmup_frac: float = 0.4):
    """Steps per second over the tail of the run, skipping the warm-up."""
    path = os.path.join(run_dir, "simulation_data.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh)]
    pts = []
    for r in rows:
        try:
            pts.append((float(r["step"]), float(r["time_s"])))
        except (KeyError, ValueError):
            continue
    if len(pts) < 3:
        return None
    pts.sort()
    start = pts[int(len(pts) * warmup_frac)]
    end = pts[-1]
    dt = end[1] - start[1]
    if dt <= 0:
        return None
    return (end[0] - start[0]) / dt


def dmon_stats(path: str):
    """Mean utilisation and peak memory from `nvidia-smi dmon -s um`.

    dmon prints a header line every so often and one line per GPU per sample;
    only the cards that actually ran anything are of interest, so idle-zero
    rows would drag the mean down and are dropped.
    """
    if not path or not os.path.exists(path):
        return float("nan"), float("nan")
    utils, mems = [], []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            f = line.split()
            # time gpu fb bar1 ... sm mem enc dec -- layout varies by driver,
            # so locate the columns by taking the numeric fields after the gpu
            # index: fb (MiB) first, then sm/mem utilisation percentages.
            nums = []
            for tok in f:
                try:
                    nums.append(float(tok))
                except ValueError:
                    nums.append(None)
            nums = [n for n in nums if n is not None]
            if len(nums) < 5:
                continue
            fb, sm = nums[1], nums[-4] if len(nums) >= 6 else nums[-1]
            if fb and fb > 0:
                mems.append(fb)
                utils.append(sm)
    if not mems:
        return float("nan"), float("nan")
    return sum(utils) / len(utils), max(mems)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern")
    ap.add_argument("--label")
    ap.add_argument("--H", type=float)
    ap.add_argument("--nproc", type=int)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP_NUM_THREADS used; 0 means it was left unset")
    ap.add_argument("--dmon")
    ap.add_argument("--append")
    ap.add_argument("--report")
    args = ap.parse_args(argv)

    if args.report:
        with open(args.report) as fh:
            rows = list(csv.DictReader(fh))
        print(f"{'label':<18} {'H':>4} {'proc/gpu':>8} {'gpu':>4} {'thr':>4} "
              f"{'procs':>6} {'it/s each':>10} {'min':>8} {'it/s node':>10} "
              f"{'h per 1e6':>10} {'sm%':>6} {'fb MB':>8}")
        print("-" * 110)
        for r in rows:
            print(f"{r['label']:<18} {float(r['H']):>4.0f} {r['nproc']:>8} "
                  f"{r['ngpu']:>4} {r.get('threads', '?'):>4} "
                  f"{r['n_proc_total']:>6} "
                  f"{float(r['rate_mean']):>10.2f} {float(r['rate_min']):>8.2f} "
                  f"{float(r['rate_total']):>10.1f} "
                  f"{float(r['hours_1e6_slowest']):>10.2f} "
                  f"{float(r['gpu_util_pct']):>6.0f} "
                  f"{float(r['gpu_mem_mb']):>8.0f}")
        return 0

    dirs = [d for d in sorted(glob.glob(args.pattern)) if os.path.isdir(d)]
    rates = [r for r in (rate_of(d) for d in dirs) if r]
    if not rates:
        print(f"  no usable timings under {args.pattern}", file=sys.stderr)
        return 1

    util, mem = dmon_stats(args.dmon)
    slowest = min(rates)
    row = dict(
        label=args.label, H=args.H, nproc=args.nproc, ngpu=args.ngpu,
        threads=args.threads, n_proc_total=len(rates),
        rate_mean=round(sum(rates) / len(rates), 3),
        rate_min=round(slowest, 3), rate_max=round(max(rates), 3),
        rate_total=round(sum(rates), 3),
        hours_1e6_slowest=round(HOURS_1E6 / slowest, 3),
        gpu_util_pct=round(util, 1), gpu_mem_mb=round(mem, 1),
        n_parsed=len(rates),
    )
    print(f"  {args.label}: {row['rate_mean']:.2f} it/s each (min "
          f"{row['rate_min']:.2f}), {row['rate_total']:.1f} it/s aggregate, "
          f"1e6 steps in {row['hours_1e6_slowest']:.2f} h")

    if args.append:
        new = not os.path.exists(args.append)
        with open(args.append, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
