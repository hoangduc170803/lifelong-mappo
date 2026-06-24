"""Patched onpolicy package.

Upstream commit: de66d7a4b23fac2513f56f96f73b3f5cb96695ac

The original ``__init__.py`` eagerly imported ``algorithms``, ``envs``,
``runner``, ``scripts``, ``utils``, ``config``. The ``envs`` package in turn
calls ``from absl import flags`` at import time which forces every consumer
to install absl, gym, smac, etc. even when only ``algorithms`` is needed.

We do not use the vendored SMAC/MPE/Hanabi/Football env wrappers from this
repository — only the algorithm, buffer, and policy modules — so we keep
this file empty. Submodules are imported explicitly via fully-qualified
paths from ``src/rl/mappo_onpolicy``.

See ``on-policy/PATCHES.md`` for the change log.
"""

__version__ = "0.1.0+ippo_mapf"
