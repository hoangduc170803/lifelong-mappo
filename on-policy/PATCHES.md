# Vendored on-policy patches

Upstream: <https://github.com/marlbenchmark/on-policy>
Pinned commit: `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`

This vendor copy is consumed by `src/rl/mappo_onpolicy/` for MAPPO training
of the warehouse PettingZoo env. We do **not** use the bundled SMAC, MPE,
Hanabi, or Football env wrappers from `onpolicy/envs/`, so we strip the
eager imports that would force those dependencies.

## Patches applied

1. **`onpolicy/runner/shared/base_runner.py`** — wrapped the
   top-level `import wandb` in a `try/except ImportError` so the runner
   can be used without wandb installed. wandb-dependent code paths are
   only reached when ``--use_wandb`` is set (we default to ``False``).

2. **`onpolicy/__init__.py`** — replaced the eager
   `from onpolicy import algorithms, envs, runner, scripts, utils, config`
   with an empty module docstring. The original triggers
   `onpolicy/envs/__init__.py` which calls `from absl import flags` and
   forces absl + smac + gym imports even when only the algorithms
   subpackage is wanted.

   Submodules are now imported explicitly, e.g.
   ```python
   from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
   from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
   from onpolicy.utils.shared_buffer import SharedReplayBuffer
   from onpolicy.utils.valuenorm import ValueNorm
   ```

3. **`onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`** — `R_Actor`
   may select a GNN observation encoder over the per-agent kNN agent graph
   instead of `MLPBase`, chosen by `args.encoder == "gat"` (Sprint 4b). The
   base is built INSIDE `R_Actor.__init__` (before `R_MAPPOPolicy` constructs
   the optimizer over `actor.parameters()`), so a post-hoc swap is not an
   option — hence the in-package patch. `GNNBase` lives in
   `src/rl/mappo_onpolicy/gnn_encoder.py` and is imported lazily so the
   default `mlp` path needs no torch_geometric. `R_Critic` is unchanged
   (its cent_obs is a flat concat, not one agent's graph).

## Dependencies actually pulled in

Only these from the upstream `requirements.txt` are needed at runtime:
- `torch` (already pinned in project `requirements.txt`)
- `numpy` (already pinned)
- `tensorboardX` (added in project `requirements.txt`)

We do **not** install: absl-py, gym (legacy), smac, sacred, wandb (optional),
imageio, mpi4py, setproctitle. Action mask is wired through the existing
`available_actions` plumbing in `R_Actor`/`ACTLayer`/`SharedReplayBuffer`,
so no algorithm changes are required.

## Numpy 2.x compatibility

The algorithm and utils modules used by `src/rl/mappo_onpolicy/` do not call
the deprecated `np.bool` / `np.int` / `np.float` aliases. Such calls only
exist under `onpolicy/envs/starcraft2/` and `onpolicy/envs/hanabi/`, which
this project never imports.

## How to refresh

```bash
cd on-policy
git fetch origin
git checkout <new_commit>
# Re-apply the patches above to onpolicy/__init__.py.
```
