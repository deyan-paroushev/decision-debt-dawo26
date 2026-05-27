# Zenodo release checklist

Use this checklist before minting the DOI.

## Before GitHub release

- [ ] Run `python scripts/reproduce_section_5_2.py`.
- [ ] Run `python -m unittest discover -s tests`.
- [ ] Confirm `outputs/run_summary.json` exists.
- [ ] Confirm `outputs/table2_frontier_shares.csv` matches the paper.
- [ ] Confirm `README.md` says the package is robustness analysis, not validation.
- [ ] Confirm `CITATION.cff` has the correct repository URL.
- [ ] Confirm license files are present.
- [ ] Commit all files, including generated outputs.

## GitHub release

```bash
git tag -a v1.0.0 -m "DAWO26 reproducibility package v1.0.0"
git push origin v1.0.0
```

Create a GitHub release from tag `v1.0.0`.

## Zenodo

- [ ] Link Zenodo to the GitHub repository.
- [ ] Archive release `v1.0.0`.
- [ ] Copy the minted DOI.
- [ ] Update `CITATION.cff` DOI field.
- [ ] Update the paper's Data and Code Availability statement if still possible.
- [ ] Create a final patch release only if absolutely necessary.

## Suggested Zenodo metadata

Title: `Decision Debt DAWO26 Reproducibility Package: capacity matrix and robustness analysis`

Creators: `Deyan Paroushev (ORCID: 0009-0003-8231-8265)`

Description: `Focused software and data artifact for reproducing the Section 5.2 robustness analysis and Table 2 of the DAWO26 paper "Decision debt: a framework for reasoning about avoidable information loss in collective decision procedures for DAO governance".`

Keywords: `DAO governance`, `decision support`, `social choice`, `voting methods`, `reproducibility`, `capacity matrix`, `Decision Debt`

License: `MIT for code; CC BY 4.0 for data/documentation`
