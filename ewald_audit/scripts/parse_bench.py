#!/usr/bin/env python3
"""Parse the tqdm rate out of the benchmark logs written by sbatch_bench.sh."""
import csv
import glob
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
RAW = os.path.join(RES, "bench_raw")

# "MC: 100%|####| 2000/2000 [00:24<00:00, 81.23it/s]"  (or "1.23s/it")
PAT = re.compile(r"(\d+)/(\d+)\s*\[[\d:]+<[\d:?]+,\s*([\d.]+)(it/s|s/it)")


def rate(path):
    best = None
    with open(path, errors="replace") as f:
        for m in PAT.finditer(f.read()):
            done, total, val, unit = m.groups()
            if done != total:
                continue
            best = float(val) if unit == "it/s" else 1.0 / float(val)
    return best


def main():
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RAW, "*.log"))):
        m = re.match(r"(old|new)_H(\d+)_n(\d+)_s(\d+)\.log", os.path.basename(p))
        if not m:
            continue
        r = rate(p)
        if r:
            groups[(m.group(1), int(m.group(2)), int(m.group(3)))].append(r)

    rows = []
    for (variant, H, nproc), rs in sorted(groups.items()):
        rows.append(dict(variant=variant, H=H, nproc=nproc, n_ok=len(rs),
                         it_per_s_per_proc=sum(rs) / len(rs),
                         it_per_s_total=sum(rs)))
    out = os.path.join(RES, "bench.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                           ["variant", "H", "nproc", "n_ok",
                            "it_per_s_per_proc", "it_per_s_total"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'variant':>8} {'H':>3} {'nproc':>6} {'it/s/proc':>10} {'it/s tot':>9} "
          f"{'h per 1e6 steps':>16}")
    for r in rows:
        print(f"{r['variant']:>8} {r['H']:>3} {r['nproc']:>6} "
              f"{r['it_per_s_per_proc']:>10.2f} {r['it_per_s_total']:>9.2f} "
              f"{1e6 / r['it_per_s_per_proc'] / 3600:>16.2f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
