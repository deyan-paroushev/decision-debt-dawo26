#!/usr/bin/env python3
"""Reproduce DAWO26 Decision Debt Section 5.2 and Table 2.

This script is the main reproducibility entry point for the repository. It
creates CSV outputs under ./outputs and prints a concise audit log. Random
operations use fixed seed 20260526.

Important: this is an internal robustness/sensitivity analysis of the model.
It is not empirical validation of the capacity matrix, the elicitation wizard,
the dimension basis, or the relationship between lower modelled debt and better
DAO governance outcomes.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decision_debt_model import (  # noqa: E402
    DIMS,
    batch_frontier_masks,
    frontier,
    issuance,
    load_capacity_matrix,
    normalise_active_weights,
    reform_delta,
)

SEED = 20260526


def make_rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def worked_example_profile() -> tuple[np.ndarray, np.ndarray]:
    r = np.array([1, 1, 2, 0, 1, 1, 0, 0, 0.0])
    w = np.array([0.250, 0.167, 0.333, 0, 0.083, 0.167, 0, 0, 0])
    return r, w


def assert_worked_example(spec) -> None:
    r, w = worked_example_profile()
    i = issuance(spec.capacities, r, w)
    expected = {
        "approval_voting": 0.750,
        "score_voting": 0.500,
        "quadratic_voting": 0.333,
        "condorcet_schulze": 0.750,
    }
    for method, expected_value in expected.items():
        idx = spec.methods.index(method)
        observed = round(float(i[idx]), 3)
        assert observed == expected_value, f"{method}: expected {expected_value}, got {observed}"
    four = [spec.methods.index(m) for m in expected]
    idx, u, _ = frontier(spec.capacities[four], r, w)
    assert list(idx) == [2], f"four-method frontier should be quadratic, got {idx}"
    assert round(u, 3) == 0.333, f"four-method U should be 0.333, got {u}"


def random_profiles(n: int, offset: int = 0, p_active: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    rng = make_rng(offset)
    rs = np.zeros((n, len(DIMS)))
    ws = np.zeros_like(rs)
    for row in range(n):
        active = rng.random(len(DIMS)) < p_active
        if not active.any():
            active[rng.integers(0, len(DIMS))] = True
        rs[row, active] = rng.integers(1, 3, size=active.sum())
        raw = np.zeros(len(DIMS))
        raw[active] = rng.dirichlet(np.ones(active.sum()))
        ws[row] = normalise_active_weights(raw, active)
    return rs, ws


def frontier_counts(spec, rs, ws, mode="ratio", norm="L1") -> tuple[pd.DataFrame, float]:
    mask, _, _ = batch_frontier_masks(spec.capacities, rs, ws, mode=mode, norm=norm)
    n = len(rs)
    unique = mask.sum(axis=1) == 1
    rows = []
    for i, method in enumerate(spec.methods):
        rows.append(
            {
                "method": method,
                "on_frontier_pct": 100 * mask[:, i].sum() / n,
                "unique_frontier_pct": 100 * np.logical_and(unique, mask[:, i]).sum() / n,
            }
        )
    return pd.DataFrame(rows).sort_values(["on_frontier_pct", "unique_frontier_pct"], ascending=False), 100 * (~unique).sum() / n


def exhaustive_equal_weight_profiles() -> tuple[np.ndarray, np.ndarray]:
    rs = []
    ws = []
    for vals in product([0, 1, 2], repeat=len(DIMS)):
        if not any(vals):
            continue
        r = np.array(vals, dtype=float)
        active = r > 0
        raw = np.zeros(len(DIMS))
        raw[active] = 1.0
        rs.append(r)
        ws.append(normalise_active_weights(raw, active))
    return np.vstack(rs), np.vstack(ws)


def dominance_relations(spec) -> pd.DataFrame:
    rows = []
    c = spec.capacities
    for i, mi in enumerate(spec.methods):
        for j, mj in enumerate(spec.methods):
            if i != j and (c[i] >= c[j]).all() and (c[i] > c[j]).any():
                rows.append({"dominant": mi, "dominated": mj})
    return pd.DataFrame(rows)


def identical_capacity_groups(spec) -> pd.DataFrame:
    groups: dict[tuple[int, ...], list[str]] = {}
    for method, row in zip(spec.methods, spec.capacities.astype(int)):
        groups.setdefault(tuple(row.tolist()), []).append(method)
    rows = [
        {"capacity_vector": str(vector), "methods": ";".join(methods)}
        for vector, methods in groups.items()
        if len(methods) > 1
    ]
    return pd.DataFrame(rows)


def load_battery(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    profiles = []
    for _, row in df.iterrows():
        r = row.loc[list(DIMS)].to_numpy(dtype=float)
        wcols = [f"w_{d}" for d in DIMS]
        w = row.loc[wcols].to_numpy(dtype=float)
        active = r > 0
        w = normalise_active_weights(w, active)
        profiles.append({"decision_type": row["decision_type"], "r": r, "w": w, "default_method": row["default_method"]})
    return profiles


def battery_table(spec, profiles, mode="ratio", norm="L1") -> pd.DataFrame:
    rows = []
    for profile in profiles:
        idx, u, i = frontier(spec.capacities, profile["r"], profile["w"], mode=mode, norm=norm)
        ex = i - u
        default_idx = spec.methods.index(profile["default_method"])
        rows.append(
            {
                "decision_type": profile["decision_type"],
                "active_dims": ",".join(d for d, val in zip(DIMS, profile["r"]) if val > 0),
                "frontier": ";".join(spec.methods[j] for j in idx),
                "U": u,
                "default_method": profile["default_method"],
                "default_excess": ex[default_idx],
                "mean_excess": ex.mean(),
                "max_excess": ex.max(),
            }
        )
    return pd.DataFrame(rows)


def scalarisation_table(spec, profiles) -> pd.DataFrame:
    r, w = worked_example_profile()
    all_profiles = [{"decision_type": "worked_example", "r": r, "w": w}, *profiles]
    rows = []
    for profile in all_profiles:
        for norm in ["L1", "L2", "max"]:
            idx, u, _ = frontier(spec.capacities, profile["r"], profile["w"], norm=norm)
            rows.append({"profile": profile["decision_type"], "norm": norm, "frontier": ";".join(spec.methods[j] for j in idx), "U": u})
    return pd.DataFrame(rows)


def adequacy_sensitivity_random(spec, rs, ws) -> pd.DataFrame:
    rows = []
    for mode in ["ratio", "strict", "sqrt_ratio", "presence_only"]:
        fc, tie = frontier_counts(spec, rs, ws, mode=mode)
        top = fc.iloc[0]
        never = ";".join(fc.loc[np.isclose(fc["on_frontier_pct"], 0), "method"].tolist())
        rows.append({"mode": mode, "top_method": top["method"], "top_on_frontier_pct": top["on_frontier_pct"], "tie_pct": tie, "never_frontier": never})
    return pd.DataFrame(rows)


def adequacy_sensitivity_battery(spec, profiles) -> pd.DataFrame:
    rows = []
    for mode in ["ratio", "strict", "sqrt_ratio", "presence_only"]:
        table = battery_table(spec, profiles, mode=mode)
        for _, row in table.iterrows():
            rows.append({"mode": mode, "decision_type": row["decision_type"], "frontier": row["frontier"], "U": row["U"]})
    return pd.DataFrame(rows)


def per_entry_influence(spec, profiles) -> pd.DataFrame:
    rs, ws = random_profiles(5000, offset=202)
    rs = np.vstack([rs] + [p["r"][None, :] for p in profiles])
    ws = np.vstack([ws] + [p["w"][None, :] for p in profiles])
    base_mask, _, _ = batch_frontier_masks(spec.capacities, rs, ws)
    rows = []
    for mi, method in enumerate(spec.methods):
        for di, dim in enumerate(spec.dimensions):
            candidates = []
            for delta in (-1, +1):
                new_value = spec.capacities[mi, di] + delta
                if 0 <= new_value <= 2:
                    c2 = spec.capacities.copy()
                    c2[mi, di] = new_value
                    mask, _, _ = batch_frontier_masks(c2, rs, ws)
                    flip_pct = 100 * (mask != base_mask).any(axis=1).mean()
                    candidates.append((delta, flip_pct))
            if candidates:
                best = max(candidates, key=lambda pair: pair[1])
                rows.append(
                    {
                        "entry": f"{method}.{dim}",
                        "method": method,
                        "dimension": dim,
                        "base_capacity": spec.capacities[mi, di],
                        "most_disruptive_delta": best[0],
                        "frontier_set_flip_pct": best[1],
                        "valid_deltas": ";".join(f"{d:+d}:{p:.3f}" for d, p in candidates),
                    }
                )
    return pd.DataFrame(rows).sort_values("frontier_set_flip_pct", ascending=False)


def dimension_ablation(spec) -> pd.DataFrame:
    rs, ws = random_profiles(10000, offset=505)
    base_mask, _, _ = batch_frontier_masks(spec.capacities, rs, ws)
    rows = []
    for di, dim in enumerate(spec.dimensions):
        active_rows = rs[:, di] > 0
        changed = 0
        applicable = 0
        for r, w, base in zip(rs[active_rows], ws[active_rows], base_mask[active_rows]):
            r2 = r.copy()
            w2 = w.copy()
            r2[di] = 0
            w2[di] = 0
            if w2.sum() == 0:
                continue
            w2 = w2 / w2.sum()
            applicable += 1
            idx, _, _ = frontier(spec.capacities, r2, w2)
            vector = np.zeros(len(spec.methods), dtype=bool)
            vector[idx] = True
            if not np.array_equal(vector, base):
                changed += 1
        rows.append({"removed_dimension": dim, "profiles_where_active": applicable, "frontier_change_pct_when_removed": 100 * changed / applicable})
    return pd.DataFrame(rows).sort_values("frontier_change_pct_when_removed", ascending=False)


def weight_space_stability(spec, profiles) -> pd.DataFrame:
    rng = make_rng(303)
    rows = []
    for profile in profiles:
        active = profile["r"] > 0
        n = 20000
        rs = np.tile(profile["r"], (n, 1))
        ws = np.zeros((n, len(DIMS)))
        ws[:, active] = rng.dirichlet(np.ones(active.sum()), size=n)
        mask, _, _ = batch_frontier_masks(spec.capacities, rs, ws)
        unique = mask.sum(axis=1) == 1
        for i, method in enumerate(spec.methods):
            pct = 100 * mask[:, i].sum() / n
            if pct > 0:
                rows.append({"decision_type": profile["decision_type"], "method": method, "on_frontier_pct": pct})
        rows.append({"decision_type": profile["decision_type"], "method": "<tie_rate>", "on_frontier_pct": 100 * (~unique).sum() / n})
    return pd.DataFrame(rows).sort_values(["decision_type", "on_frontier_pct"], ascending=[True, False])


def worked_example_tables(spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    r, w = worked_example_profile()
    i = issuance(spec.capacities, r, w)
    rows = []
    for method in ["approval_voting", "score_voting", "quadratic_voting", "condorcet_schulze"]:
        idx = spec.methods.index(method)
        rows.append({"method": method, "L1_issuance": i[idx]})
    chosen = spec.methods.index("approval_voting")
    front = spec.methods.index("quadratic_voting")
    delta = reform_delta(spec.capacities, r, w, chosen, front)
    delta_rows = [{"dimension": dim, "delta": val} for dim, val in zip(DIMS, delta) if r[list(DIMS).index(dim)] > 0]
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    spec = load_capacity_matrix(args.data_dir / "capacity_matrix.csv")
    profiles = load_battery(args.data_dir / "decision_type_battery.csv")
    assert_worked_example(spec)

    print(f"SEED {SEED}")
    print(f"numpy {np.__version__} pandas {pd.__version__}")
    print("Worked-example verification passed.")
    print("Matrix:")
    print(pd.DataFrame(spec.capacities.astype(int), index=spec.methods, columns=spec.dimensions).to_string())

    random_rs, random_ws = random_profiles(20000, offset=101)
    random_counts, random_tie = frontier_counts(spec, random_rs, random_ws)
    exhaustive_rs, exhaustive_ws = exhaustive_equal_weight_profiles()
    exhaustive_counts, exhaustive_tie = frontier_counts(spec, exhaustive_rs, exhaustive_ws)

    random_counts.to_csv(args.outputs_dir / "table2_random_frontier_counts.csv", index=False)
    exhaustive_counts.to_csv(args.outputs_dir / "table2_exhaustive_frontier_counts.csv", index=False)

    # Compact Table 2 exactly as used by the paper, rounded for readability.
    table2 = random_counts[["method", "on_frontier_pct"]].rename(columns={"on_frontier_pct": "on_frontier_random_profiles_pct"}).merge(
        exhaustive_counts[["method", "on_frontier_pct"]].rename(columns={"on_frontier_pct": "on_frontier_exhaustive_profiles_pct"}),
        on="method",
        how="left",
        sort=False,
    ).sort_values("on_frontier_random_profiles_pct", ascending=False)
    table2.to_csv(args.outputs_dir / "table2_frontier_shares.csv", index=False)

    dominance_relations(spec).to_csv(args.outputs_dir / "dominance_relations.csv", index=False)
    identical_capacity_groups(spec).to_csv(args.outputs_dir / "identical_capacity_groups.csv", index=False)
    battery_table(spec, profiles).to_csv(args.outputs_dir / "decision_type_battery.csv", index=False)
    scalarisation_table(spec, profiles).to_csv(args.outputs_dir / "scalarisation_table.csv", index=False)
    adequacy_sensitivity_random(spec, random_rs, random_ws).to_csv(args.outputs_dir / "adequacy_sensitivity_random.csv", index=False)
    adequacy_sensitivity_battery(spec, profiles).to_csv(args.outputs_dir / "adequacy_sensitivity_battery.csv", index=False)
    per_entry_influence(spec, profiles).to_csv(args.outputs_dir / "per_entry_influence.csv", index=False)
    dimension_ablation(spec).to_csv(args.outputs_dir / "dimension_ablation.csv", index=False)
    weight_space_stability(spec, profiles).to_csv(args.outputs_dir / "weight_space_stability_battery.csv", index=False)
    we, delta = worked_example_tables(spec)
    we.to_csv(args.outputs_dir / "worked_example_issuance.csv", index=False)
    delta.to_csv(args.outputs_dir / "worked_example_reform_delta.csv", index=False)

    summary = {
        "seed": SEED,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_random_profiles": int(len(random_rs)),
        "n_exhaustive_profiles": int(len(exhaustive_rs)),
        "random_tie_pct": float(random_tie),
        "exhaustive_tie_pct": float(exhaustive_tie),
        "top_random_frontier_method": str(random_counts.iloc[0]["method"]),
        "top_random_frontier_pct": float(random_counts.iloc[0]["on_frontier_pct"]),
        "top_exhaustive_frontier_method": str(exhaustive_counts.iloc[0]["method"]),
        "top_exhaustive_frontier_pct": float(exhaustive_counts.iloc[0]["on_frontier_pct"]),
    }
    (args.outputs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTABLE 2 FRONTIER SHARES")
    print(table2.to_string(index=False))
    print(f"random_tie_pct {random_tie:.3f}")
    print(f"exhaustive_tie_pct {exhaustive_tie:.3f}")
    print(f"\nWrote outputs to {args.outputs_dir}")


if __name__ == "__main__":
    main()
