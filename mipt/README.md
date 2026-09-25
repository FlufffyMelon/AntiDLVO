# Running this repo on mipt (`calc.cod.phystech.edu`)

Migration target after softcluster filled up. Everything here is additive:
`src/` and `configs_prod/` are untouched.

## Target hardware

`node211-27` — 4× Tesla V100-SXM2-32GB, 128 CPU, 1 TB RAM.

| partition | walltime | use it? |
|---|---|---|
| `AMD7V12-V100-CCP` | infinite | **yes** — ours, no eviction |
| `AMD7V12-V100-CCP-common` | 1 day | no — shared, low priority |

Both sbatch scripts here pin `--partition=AMD7V12-V100-CCP --nodelist=node211-27`.

## One-time setup

```bash
git clone git@github.com:FlufffyMelon/AntiDLVO.git ~/AntiDLVO
cd ~/AntiDLVO && git checkout Dev
sbatch mipt/sbatch_setup_env.sh        # builds .venv via uv, on the compute node
```

The build runs as a batch job rather than on the login node, and takes a GPU so
it can prove CuPy compiles a kernel before the job exits. It needs ~2 min.
Pass `REBUILD=1` to force a clean rebuild of an existing `.venv`:

```bash
REBUILD=1 sbatch mipt/sbatch_setup_env.sh
```

## Verify

```bash
sbatch mipt/sbatch_smoke.sh            # 4 short runs, one per V100
```

Look for `=== SMOKE TEST PASSED ===` in `mipt/logs/smoke_<jobid>.out`.

## Notes

- **uv** is at `~/miniconda3/bin/uv`, which only a *login* shell puts on PATH —
  hence `#!/bin/bash -l` at the top of both scripts. A plain `#!/bin/bash`
  sbatch script will not find it.
- **CUDA** comes from `module load cuda/12.9`
  (`/opt/modules/nvidia/cuda/12.9`), which supplies the `libnvrtc.so.12` that
  `cupy-cuda12x` needs at import time. `cuda/13.1` also exists but is not
  matched by the pinned `cupy-cuda12x` wheel.
- **Versions** are pinned in `requirements-cluster.txt` to the working
  softcluster venv (Python 3.13.2), so the two machines stay comparable.
- Compute nodes have outbound network (PyPI and GitHub both reachable), so
  `uv` can resolve and download from inside the job.
- Smoke-test output goes to `results_smoke_mipt/`, ignored by `results*/`.
