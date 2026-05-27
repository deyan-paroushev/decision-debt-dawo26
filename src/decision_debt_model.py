"""Core Decision Debt computation utilities.

This module implements the model used in the DAWO26 paper:

    Decision debt: a framework for reasoning about avoidable information
    loss in collective decision procedures for DAO governance

The code deliberately keeps the model small and inspectable. It validates
inputs, handles ties as frontier sets, and exposes adequacy/scalarisation
choices explicitly.

Important epistemic boundary
----------------------------
This module verifies internal model behaviour under a stated capacity matrix,
requirement profile, and salience weights. It does not validate the capacity
matrix, the elicitation wizard, the independence of dimensions, or the claim
that lower debt corresponds to better governance outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

DIMS: tuple[str, ...] = ("A", "O", "I", "F", "M", "P", "D", "T", "J")
AdequacyMode = Literal["ratio", "strict", "sqrt_ratio", "presence_only"]
Norm = Literal["L1", "L2", "max"]


@dataclass(frozen=True)
class MatrixSpec:
    """A validated method-capacity matrix."""

    methods: tuple[str, ...]
    dimensions: tuple[str, ...]
    capacities: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.capacities, dtype=float)
        if c.ndim != 2:
            raise ValueError("capacity matrix must be two-dimensional")
        if c.shape != (len(self.methods), len(self.dimensions)):
            raise ValueError(
                "capacity matrix shape does not match methods/dimensions: "
                f"{c.shape} vs {(len(self.methods), len(self.dimensions))}"
            )
        if not np.isfinite(c).all():
            raise ValueError("capacity matrix contains non-finite values")
        if (c < 0).any():
            raise ValueError("capacity matrix contains negative values")


def load_capacity_matrix(path: str | Path) -> MatrixSpec:
    """Load a CSV capacity matrix with columns: method,A,O,I,F,M,P,D,T,J."""
    df = pd.read_csv(path)
    required = ["method", *DIMS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"capacity matrix missing columns: {missing}")
    if df["method"].duplicated().any():
        duplicated = df.loc[df["method"].duplicated(), "method"].tolist()
        raise ValueError(f"duplicate method names: {duplicated}")
    capacities = df.loc[:, list(DIMS)].to_numpy(dtype=float)
    # The paper uses a 0/1/2 author-specified capacity scale. Keep this as a
    # hard check for reproducibility of the paper's calculations.
    allowed = {0.0, 1.0, 2.0}
    observed = set(np.unique(capacities).tolist())
    if not observed.issubset(allowed):
        raise ValueError(f"capacity values must be in {{0,1,2}}, got {sorted(observed)}")
    return MatrixSpec(tuple(df["method"].astype(str)), DIMS, capacities)


def validate_profile(r: np.ndarray, w: np.ndarray, n_dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return requirement vector and salience weights as floats."""
    r = np.asarray(r, dtype=float)
    w = np.asarray(w, dtype=float)
    if r.shape != (n_dims,):
        raise ValueError(f"r must have shape ({n_dims},), got {r.shape}")
    if w.shape != (n_dims,):
        raise ValueError(f"w must have shape ({n_dims},), got {w.shape}")
    if not np.isfinite(r).all() or not np.isfinite(w).all():
        raise ValueError("r and w must contain finite values")
    if (r < 0).any():
        raise ValueError("r must be non-negative")
    if (w < 0).any():
        raise ValueError("w must be non-negative")
    active = r > 0
    if not active.any():
        raise ValueError("at least one dimension must be active")
    if not np.isclose(w[active].sum(), 1.0, atol=1e-9):
        raise ValueError(f"weights over active dimensions must sum to 1, got {w[active].sum()}")
    if not np.isclose(w[~active].sum(), 0.0, atol=1e-9):
        raise ValueError(f"inactive dimensions must have zero weight, got {w[~active].sum()}")
    return r, w


def normalise_active_weights(raw_weights: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Normalise non-negative raw weights over active dimensions."""
    raw_weights = np.asarray(raw_weights, dtype=float)
    active = np.asarray(active, dtype=bool)
    if raw_weights.shape != active.shape:
        raise ValueError("raw_weights and active must have the same shape")
    if (raw_weights < 0).any():
        raise ValueError("raw_weights must be non-negative")
    if not active.any():
        raise ValueError("at least one dimension must be active")
    total = raw_weights[active].sum()
    if total <= 0:
        raise ValueError("active raw weights must sum to a positive value")
    w = np.zeros_like(raw_weights, dtype=float)
    w[active] = raw_weights[active] / total
    return w


def debt_matrix(
    capacities: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    mode: AdequacyMode = "ratio",
) -> np.ndarray:
    """Compute b_kd(m) for all methods and dimensions.

    Parameters
    ----------
    capacities:
        Method-by-dimension capacity matrix.
    r:
        Requirement vector. r[d] = 0 marks inactive dimensions.
    w:
        Salience weights, summing to 1 over active dimensions.
    mode:
        Adequacy function. The paper's default is "ratio".
    """
    capacities = np.asarray(capacities, dtype=float)
    r, w = validate_profile(r, w, capacities.shape[1])
    active = r > 0
    adequacy = np.ones_like(capacities, dtype=float)
    if mode == "ratio":
        adequacy[:, active] = np.minimum(capacities[:, active], r[active]) / r[active]
    elif mode == "strict":
        adequacy[:, active] = (capacities[:, active] >= r[active]).astype(float)
    elif mode == "sqrt_ratio":
        adequacy[:, active] = np.sqrt(np.minimum(capacities[:, active], r[active]) / r[active])
    elif mode == "presence_only":
        adequacy[:, active] = (capacities[:, active] > 0).astype(float)
    else:
        raise ValueError(f"unknown adequacy mode: {mode}")
    return (1.0 - adequacy) * w


def issuance(
    capacities: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    mode: AdequacyMode = "ratio",
    norm: Norm = "L1",
) -> np.ndarray:
    """Compute scalar issuance I_k(m) for all methods."""
    b = debt_matrix(capacities, r, w, mode)
    if norm == "L1":
        return b.sum(axis=1)
    if norm == "L2":
        return np.sqrt((b * b).sum(axis=1))
    if norm == "max":
        return b.max(axis=1)
    raise ValueError(f"unknown norm: {norm}")


def frontier(
    capacities: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    mode: AdequacyMode = "ratio",
    norm: Norm = "L1",
    atol: float = 1e-12,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return frontier indices, repertoire floor U, and all issuance values."""
    i = issuance(capacities, r, w, mode, norm)
    u = float(i.min())
    return np.where(np.isclose(i, u, atol=atol))[0], u, i


def reform_delta(
    capacities: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    chosen_idx: int,
    frontier_idx: int,
    mode: AdequacyMode = "ratio",
) -> np.ndarray:
    """Compute Delta_kd = b_kd(m_chosen) - b_kd(m_frontier)."""
    b = debt_matrix(capacities, r, w, mode)
    return b[chosen_idx] - b[frontier_idx]


def batch_frontier_masks(
    capacities: np.ndarray,
    rs: np.ndarray,
    ws: np.ndarray,
    mode: AdequacyMode = "ratio",
    norm: Norm = "L1",
    atol: float = 1e-12,
    validate_inputs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised frontier masks for many profiles."""
    capacities = np.asarray(capacities, dtype=float)
    rs = np.asarray(rs, dtype=float)
    ws = np.asarray(ws, dtype=float)
    if rs.shape != ws.shape or rs.ndim != 2 or rs.shape[1] != capacities.shape[1]:
        raise ValueError("rs and ws must be n_profiles by n_dimensions")
    if validate_inputs:
        # Optional explicit validation catches silent reproducibility drift.
        # It is disabled for large repeated sensitivity loops for speed.
        for r, w in zip(rs, ws):
            validate_profile(r, w, capacities.shape[1])
    c3 = capacities[None, :, :]
    r3 = rs[:, None, :]
    active = r3 > 0
    if mode == "ratio":
        adequacy = np.where(active, np.minimum(c3, r3) / np.where(active, r3, 1.0), 1.0)
    elif mode == "strict":
        adequacy = np.where(active, (c3 >= r3).astype(float), 1.0)
    elif mode == "sqrt_ratio":
        adequacy = np.where(active, np.sqrt(np.minimum(c3, r3) / np.where(active, r3, 1.0)), 1.0)
    elif mode == "presence_only":
        adequacy = np.where(active, (c3 > 0).astype(float), 1.0)
    else:
        raise ValueError(f"unknown adequacy mode: {mode}")
    b = (1.0 - adequacy) * ws[:, None, :]
    if norm == "L1":
        i = b.sum(axis=2)
    elif norm == "L2":
        i = np.sqrt((b * b).sum(axis=2))
    elif norm == "max":
        i = b.max(axis=2)
    else:
        raise ValueError(f"unknown norm: {norm}")
    u = i.min(axis=1)
    mask = np.isclose(i, u[:, None], atol=atol)
    return mask, u, i
