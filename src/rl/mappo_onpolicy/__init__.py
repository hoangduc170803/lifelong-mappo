"""MAPPO training stack backed by the vendored marlbenchmark/on-policy code.

Exposes the env adapter, vec wrapper, and warehouse runner used by
`src/rl/mappo_onpolicy/train.py`. The on-policy package itself is vendored
under `on-policy/` at a pinned commit (see `on-policy/PATCHES.md`).

Importing this package automatically inserts the vendored ``on-policy/``
directory onto ``sys.path`` so that ``from onpolicy.algorithms... import ...``
works without requiring a ``PYTHONPATH`` setting.
"""

import sys as _sys
from pathlib import Path as _Path

_ON_POLICY_ROOT = _Path(__file__).resolve().parents[3] / "on-policy"
if _ON_POLICY_ROOT.is_dir() and str(_ON_POLICY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ON_POLICY_ROOT))
