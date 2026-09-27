"""
Centralized, hierarchical random-number management for BASS experiments.

Usage
-----
    exp = ExperimentSeeder(entropy=20260722)   # or None for a fresh run
    print(exp.entropy)                          # log this -- it is everything

    rng = exp.trial_rng(family="1D Brickwork", k=2000, trial=7)
    circuit = make_brickwork(N, depth=6, rng=rng)

Every (family, k, trial) triple gets its own independent, reproducible
`np.random.Generator`, addressed by name rather than by manually computed
integer offsets. Reusing the same `entropy` on a different machine, in a
different process, calling paths in a different order, or skipping some
paths entirely, all reproduce bit-identical streams for any path you do
call -- unlike a running spawn() counter, which is order- and
history-dependent.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


def _name_to_index(name: str) -> int:
    """Deterministically map an arbitrary string label to a bounded, stable
    integer for folding into a SeedSequence spawn_key.

    Uses SHA-256 truncated to 32 bits rather than Python's built-in hash():
    str hash() is randomized per-process via PYTHONHASHSEED unless
    explicitly disabled, so it is *not* reproducible across sessions --
    using it to derive seeds (as an earlier draft of this codebase did in
    one place) silently breaks reproducibility. SHA-256 is stable across
    processes, machines, and Python versions.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _path_to_spawn_key(path_parts) -> tuple:
    key = []
    for p in path_parts:
        if isinstance(p, str):
            key.append(_name_to_index(p))
        elif isinstance(p, bool):
            key.append(int(p))
        elif isinstance(p, (int, np.integer)):
            key.append(int(p))
        elif isinstance(p, float):
            # Only used for things like `k` when it is passed as a float by
            # accident; keep deterministic by hashing the repr.
            key.append(_name_to_index(repr(p)))
        else:
            key.append(_name_to_index(repr(p)))
    return tuple(key)


class ExperimentSeeder:
    """
    Hierarchical, reproducible seed manager built on numpy.random.SeedSequence.

    Parameters
    ----------
    entropy : int or None
        The single root entropy value. If None, a fresh, unpredictable value
        is drawn and exposed via `.entropy` -- log this immediately if the
        run needs to be reproducible later. This mirrors NumPy's own
        recommended SeedSequence pattern: default to None, then persist
        `.entropy` after the fact, rather than hand-picking a seed up front.
    """

    def __init__(self, entropy: int | None = None):
        self._root = np.random.SeedSequence(entropy)
        self.entropy: int = self._root.entropy

    def _child(self, *path_parts) -> np.random.SeedSequence:
        """
        Derive one deterministic child SeedSequence for an arbitrary,
        human-readable path, e.g. ("family", "1D Brickwork", "k", 2000,
        "trial", 7).

        Built directly via the `spawn_key` constructor argument (a
        documented, public part of the SeedSequence API) rather than via
        repeated `.spawn(n)` calls, so a given path always yields the same
        child regardless of what other paths have been requested before it
        in the same process -- `.spawn(n)` alone is a sequential counter and
        is therefore order-dependent, which we explicitly do not want here.
        """
        return np.random.SeedSequence(
            entropy=self.entropy, spawn_key=_path_to_spawn_key(path_parts)
        )

    def rng(self, *path_parts) -> np.random.Generator:
        """Return an independent, reproducible Generator for this path."""
        return np.random.default_rng(self._child(*path_parts))

    def seq(self, *path_parts) -> np.random.SeedSequence:
        """
        Return the raw SeedSequence for this path (rather than a Generator).
        """
        return self._child(*path_parts)

    def trial_rng(self, family: str, k, trial: int) -> np.random.Generator:
        """Convenience wrapper for the common (family, k, trial) path."""
        return self.rng("family", family, "k", k, "trial", trial)

    def config_rng(self, family: str, config: str, trial: int) -> np.random.Generator:
        """Convenience wrapper for (family, config-label, trial) paths, used
        e.g. by circuit families keyed by a config label rather than a
        sparse budget (Schmidt-truncation depth/family sweeps, PRZ scaling
        sweeps over N)."""
        return self.rng("family", family, "config", config, "trial", trial)

    def to_dict(self) -> Dict[str, int]:
        """Serializable record of the root entropy, for logging into any
        per-run metadata file (JSON/NPZ). This one number is sufficient to
        regenerate every RNG stream this seeder can produce."""
        return {"entropy": int(self.entropy)}

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "ExperimentSeeder":
        return cls(entropy=d["entropy"])

    def __repr__(self):
        return f"ExperimentSeeder(entropy={self.entropy})"
