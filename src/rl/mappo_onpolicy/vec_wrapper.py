"""Vectorized env wrapper compatible with the on-policy `SharedReplayBuffer`.

on-policy runners read ``self.envs.observation_space[0]`` directly and pass
arrays shaped ``(n_rollout_threads, num_agents, ...)`` into the buffer. The
simplest backing for Sprint 3 is ``DummyVecEnv``. Add a process-backed wrapper
in Sprint 4 only when env throughput becomes the bottleneck.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from src.rl.mappo_onpolicy.env_adapter import WarehouseOnPolicyEnv


EnvFactory = Callable[[int], WarehouseOnPolicyEnv]


class DummyVecEnv:
    """Run multiple `WarehouseOnPolicyEnv` instances sequentially in-process.

    Mirrors the surface the on-policy runners expect: ``observation_space``,
    ``share_observation_space``, ``action_space`` are *lists* over agents
    (taken from the first sub-env), ``reset`` / ``step`` return numpy arrays
    leading with the rollout dimension, and ``close`` releases all sub-envs.
    """

    def __init__(self, env_fns: Sequence[EnvFactory]):
        if not env_fns:
            raise ValueError("DummyVecEnv requires at least one env factory")
        self.envs: List[WarehouseOnPolicyEnv] = [
            fn(i) for i, fn in enumerate(env_fns)
        ]
        self.num_envs = len(self.envs)
        first = self.envs[0]
        self.num_agents = first.num_agents
        self.observation_space = first.observation_space
        self.share_observation_space = first.share_observation_space
        self.action_space = first.action_space

    # ------------------------------------------------------------ lifecycle
    def reset(
        self, seeds: Optional[Sequence[int]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seeds is None:
            results = [env.reset() for env in self.envs]
        else:
            if len(seeds) != self.num_envs:
                raise ValueError("seeds length must equal num_envs")
            results = [env.reset(seed=int(seeds[i])) for i, env in enumerate(self.envs)]
        obs = np.stack([r[0] for r in results], axis=0)
        share_obs = np.stack([r[1] for r in results], axis=0)
        avail = np.stack([r[2] for r in results], axis=0)
        return obs, share_obs, avail

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray
    ]:
        """Step every sub-env with its slice of ``actions``.

        Parameters
        ----------
        actions:
            Shape ``(num_envs, num_agents, 1)`` or ``(num_envs, num_agents)``.
        """
        actions = np.asarray(actions)
        if actions.ndim == 2:
            actions = actions[:, :, None]
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"expected actions for {self.num_envs} envs, got {actions.shape[0]}"
            )

        obs_list, share_list, rew_list, done_list, info_list, avail_list = (
            [], [], [], [], [], []
        )
        for i, env in enumerate(self.envs):
            obs, share, rew, done, info, avail = env.step(actions[i])
            obs_list.append(obs)
            share_list.append(share)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
            avail_list.append(avail)

        return (
            np.stack(obs_list, axis=0),
            np.stack(share_list, axis=0),
            np.stack(rew_list, axis=0),
            np.stack(done_list, axis=0),
            info_list,
            np.stack(avail_list, axis=0),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    # ----------------------------------------------------------------- misc
    def seed(self, seeds: Sequence[int]) -> None:
        for env, s in zip(self.envs, seeds):
            env.seed(int(s))
