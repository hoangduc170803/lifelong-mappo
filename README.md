# A Learned MAPPO Router Extends the Lifelong Operating Window in Multi-Agent Pickup-and-Delivery

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies (env + all **classical** baselines): `numpy`, `networkx`, `scipy`,
`matplotlib`, `pandas`. The classical stack runs with nothing else.

**For the MAPPO router (Layer 2) only**, also install **`torch`** (and **`torch-geometric`** for
the graph-attention encoder):

```bash
pip install torch torch-geometric
```

The MARL backbone — **on-policy** (Yu et al., MAPPO) — is **vendored under `on-policy/`**, pinned to
upstream commit `de66d7a` with three small local patches already applied: stripped the eager env
imports that would otherwise pull in `absl`/`smac`/`gym`, made `wandb` optional, and added the
GAT-encoder hook to `R_Actor`. There is nothing to clone or patch — importing
`src.rl.mappo_onpolicy` auto-inserts `on-policy/` onto `sys.path`. Only `torch`, `numpy`, and
`tensorboardX` are used from it.

The env and all classical baselines need only `numpy`/`networkx`; `torch` and the vendored
`on-policy/` are used solely by the MAPPO training/eval entry points.

---

## Quick start

### 1. Benchmark a classical baseline (works out of the box)

```bash
# PIBT, hotspot regime, 20 agents, 30 seeds, throughput over 1024 steps
python -m src.baselines.pibt_lifelong \
  --seeds $(seq 0 29) --num-agents 20 --episode-length 1024 \
  --task-rate 0.1 --task-distribution hotspot --parallel 10 \
  --results-dir results/demo/pibt_hotspot_n20

# windowed-LaCAM + reactive fallback (the strongest classical baseline)
python -m src.baselines.lacam_lifelong \
  --num-agents 20 --episode-length 1024 --task-rate 0.1 \
  --task-distribution hotspot --reactive-fallback \
  --results-dir results/demo/lacam_hotspot_n20
```

Outputs: `*_runs.csv` (per seed) + `*_summary.json` (mean ± std, and success rate + Wilson-95
CI when `--task-budget` is set). Distributions: `hotspot`, `burst_wave`,
`betweenness_bottleneck`, `bipartite_pickup_dropoff`, `uniform`, `corner_exchange`.

### 2. Train a router (requires `on-policy`)

```bash
python -m src.rl.mappo_onpolicy.train --help        # curriculum / hyperparameters
python -m src.rl.mappo_onpolicy.gate_eval --help     # evaluate a frozen actor
```

### 3. Reproduce the headline results

Run the baselines and the router across the four distributions
(`hotspot`, `burst_wave`, `betweenness_bottleneck`, `bipartite_pickup_dropoff`) and fleet sizes
N ∈ {10, 15, 20}, in both regimes: throughput (`--episode-length 1024`, no budget) and success
(`--episode-length 4096 --task-budget <B>`). Aggregate the per-seed `*_summary.json` files into
the per-cell throughput / success matrices.

Full per-cell commands, seeds, pinned budgets, and the metric definitions (the 1024 vs 4096
horizons, the PIBT-pinned budgets, and the Wilson-95 intervals) are described in the accompanying
paper/thesis — see [Citation](#citation).

---

## The map data

`warehouse_map.xml` is an [OpenTCS](https://www.opentcs.org/) plant model: a directed graph of
named points and one-way paths. Only its strongly connected component is used (so every task is
reachable). To swap maps, replace the file and update `DEFAULT_MAP_FILE` in
`src/rl/mappo_onpolicy/env_adapter.py`.

---

## Citation

If you use this code, please cite the accompanying thesis/paper (see `CITATION.cff`):

```bibtex
@misc{learnedrouter_mapd_2026,
  title  = {A Learned MAPPO Router Extends the Lifelong Operating Window in
            Multi-Agent Pickup-and-Delivery Where Classical Planners Freeze:
            A Two-Layer Study with a Pre-Registered Dispatcher Gate},
  author = {Hoang Duc Nham},
  year   = {2026},
  note   = {Code: https://github.com/hoangduc170803/lifelong-mappo.git}
}
```

---

## License

Released under the **MIT License** — see `LICENSE`.

---

## Acknowledgements

This work builds on excellent open-source research code:

- **on-policy / MAPPO** — Yu et al., <https://github.com/marlbenchmark/on-policy>
- **LaCAM** — Okumura, reference implementation <https://github.com/Kei18/pylacam>
- **RHCR** (lifelong MAPF / windowed planning) — Li et al., <https://github.com/Jiaoyang-Li/RHCR>
- **PIBT** — Okumura et al.
- **OpenTCS** — open-source transport control system, <https://www.opentcs.org/>

---
