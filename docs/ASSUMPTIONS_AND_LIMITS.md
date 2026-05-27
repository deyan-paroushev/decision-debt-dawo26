# Assumptions and limits

This document states the assumptions that the reproducibility package intentionally does not hide.

## 1. Capacity matrix

The twelve-method capacity matrix is author-specified. It encodes a qualitative interpretation of what each method's ballot/information format allows participants to express.

The matrix has not been calibrated by an expert panel. The code therefore treats it as an inspectable modelling artifact, not as a validated measurement instrument.

Attack surface: change individual entries in `data/capacity_matrix.csv` and rerun `scripts/reproduce_section_5_2.py`. The `per_entry_influence.csv` output ranks the entries whose one-step changes are most disruptive to frontier sets.

## 2. 0/1/2 scale

The default adequacy function treats the scale as a ratio-like interpolation:

```text
a = min(c, r) / r
```

This does not prove that `2` is empirically twice `1`. The script therefore includes `strict`, `sqrt_ratio`, and `presence_only` adequacy modes as sensitivity checks.

## 3. Dimension independence

The L1 issuance sum assumes separable dimensions. L2 and max-norm checks test scalarisation sensitivity, but they do not prove dimension independence. The repository does not settle whether intensity, ordinal ordering, proportional-distributive structure, and other dimensions overlap conceptually.

## 4. Frontier set is not a governance recommendation

A method on the frontier minimises modelled informational shortfall under the stated assumptions. It is not necessarily the best governance choice. Strategic behaviour, legitimacy, implementation cost, legal constraints, participation burden, and community norms are outside the model.

## 5. Synthetic profile-generation designs

The 20,000 random profiles and 19,682 exhaustive equal-weight profiles are synthetic designs. Their frontier percentages are not empirical frequencies for real DAOs.

## 6. Liquid Democracy overlay and delegable agency

The twelve base methods do not exercise the `D` dimension. The `D` dimension is exercised by the Liquid Democracy overlay, represented separately in `data/liquid_democracy_overlay.csv`. Analyses using only the twelve base methods should not interpret `D` as active.

## 7. Validation remains future work

The package does not provide:

- expert validation of the capacity matrix;
- inter-rater reliability for the elicitation wizard;
- evidence that lower modelled debt predicts better governance outcomes;
- calibrated temporal debt accumulation.

Those are empirical and conceptual research tasks beyond this reproducibility package.
