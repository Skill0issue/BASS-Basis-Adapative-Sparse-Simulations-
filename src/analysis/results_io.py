"""
Structured, per-trial result recording.

Every individual (family, method, k/chi, trial) simulation run is written
as one row -- fidelity, participation ratio, runtime, and the exact
information needed to regenerate that specific trial's random circuit
(the `seed_entropy` root value plus the `family`/`k`/`trial` path, which
together fully determine the RNG stream via
`src.core.seeding.ExperimentSeeder`).

Usage
-----
    ledger = TrialLedger()
    ledger.add(TrialRecord(
        family="1D Brickwork", method="BASS", N=20, k=2000, trial=7,
        seed_entropy=exp.entropy, fidelity=0.83, participation_ratio=12.4,
        runtime_s=0.021,
    ))
    ...
    ledger.to_csv(DATA_DIR / "fidelity_pr_trials.csv")
    ledger.to_npz(DATA_DIR / "fidelity_pr_trials.npz")

    # later, to re-derive an array for statistics:
    f_bass = ledger.get_array("fidelity", family="1D Brickwork", method="BASS", k=2000)
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class TrialRecord:
    """One fully-traceable simulation trial.

    Every field needed to (a) understand what was measured and (b) exactly
    reproduce the trial from scratch (via `seed_entropy` + `family` + `k`
    + `trial`, replayed through `ExperimentSeeder`) is included. This is
    the unit written to disk, one row per trial -- aggregation/statistics
    happen downstream, from these rows, never before writing.
    """

    family: str
    method: str  # e.g. "BASS", "Fixed-basis", "MPS", "Schmidt-1cut", "random-k"
    N: int
    k: Optional[int] = None
    chi: Optional[int] = None  # bond dimension, for MPS trials
    trial: int = 0
    seed_entropy: int = 0  # ExperimentSeeder root entropy that produced this trial
    fidelity: float = float("nan")
    participation_ratio: float = float("nan")
    runtime_s: float = float("nan")
    gamma_squared: float = float("nan")
    extra: Dict[str, Any] = field(default_factory=dict)  # method-specific fields


class TrialLedger:
    """
    Accumulates `TrialRecord`s and writes them to CSV (human-inspectable,
    diffable, spreadsheet- and git-diff friendly) and/or NPZ (fast to
    reload for downstream analysis).
    """

    def __init__(self):
        self._rows: List[TrialRecord] = []

    def add(self, record: TrialRecord) -> None:
        self._rows.append(record)

    def extend(self, records) -> None:
        self._rows.extend(records)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    # ── Persistence ────────────────────────────────────────────────────────

    def to_csv(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            path.write_text("")
            return
        extra_keys = sorted({k for r in self._rows for k in r.extra.keys()})
        fieldnames = [
            f for f in TrialRecord.__dataclass_fields__ if f != "extra"
        ] + extra_keys
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._rows:
                row = {k: v for k, v in asdict(r).items() if k != "extra"}
                row.update(r.extra)
                writer.writerow(row)

    def to_npz(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            np.savez(path)
            return
        keys = [f for f in TrialRecord.__dataclass_fields__ if f != "extra"]

        def _col(k):
            vals = [getattr(r, k) for r in self._rows]
            if k in ("family", "method"):
                return np.array(vals, dtype=object)
            return np.array([np.nan if v is None else v for v in vals])

        arrays = {k: _col(k) for k in keys}
        extra_keys = sorted({k for r in self._rows for k in r.extra.keys()})
        for ek in extra_keys:
            arrays[f"extra__{ek}"] = np.array(
                [r.extra.get(ek, np.nan) for r in self._rows]
            )
        np.savez(path, **arrays)

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "TrialLedger":
        path = Path(path)
        ledger = cls()
        base_fields = set(TrialRecord.__dataclass_fields__) - {"extra"}
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kwargs, extra = {}, {}
                for k, v in row.items():
                    if k in base_fields:
                        kwargs[k] = v
                    else:
                        extra[k] = v
                # best-effort type coercion for the known numeric fields
                for numf in ("N", "k", "chi", "trial", "seed_entropy"):
                    if kwargs.get(numf) not in (None, ""):
                        kwargs[numf] = int(float(kwargs[numf]))
                    else:
                        kwargs[numf] = None if numf in ("k", "chi") else 0
                for floatf in (
                    "fidelity",
                    "participation_ratio",
                    "runtime_s",
                    "gamma_squared",
                ):
                    if kwargs.get(floatf) not in (None, ""):
                        kwargs[floatf] = float(kwargs[floatf])
                ledger.add(TrialRecord(extra=extra, **kwargs))
        return ledger

    # ── Querying ───────────────────────────────────────────────────────────

    def filter(self, **kwargs) -> List[TrialRecord]:
        """Return the subset of records matching every given field=value
        pair -- e.g. `ledger.filter(family="1D Brickwork", method="BASS", k=2000)`
        -- the natural way to re-derive a fidelity array for one
        (family, method, k) cell before calling `paired_comparison_stats`."""

        def match(r):
            for key, val in kwargs.items():
                actual = getattr(r, key, None)
                if actual is None:
                    actual = r.extra.get(key)
                if actual != val:
                    return False
            return True

        return [r for r in self._rows if match(r)]

    def get_array(self, field_name: str, **kwargs) -> np.ndarray:
        """Convenience: `filter(**kwargs)` then pull one field into an
        ndarray, ordered by `trial` for reproducible downstream pairing."""
        rows = sorted(self.filter(**kwargs), key=lambda r: r.trial)
        out = []
        for r in rows:
            v = getattr(r, field_name, None)
            if v is None:
                v = r.extra.get(field_name, np.nan)
            out.append(v)
        return np.array(out, dtype=float)

    def families(self) -> List[str]:
        return sorted({r.family for r in self._rows})

    def methods(self) -> List[str]:
        return sorted({r.method for r in self._rows})
