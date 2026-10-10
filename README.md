# Empirical Evaluation of Geo-Indistinguishability Under Bayesian Linkage Attack

Empirical evaluation of the planar Laplace mechanism (Andrés et al., 2013) on NYC restaurant inspection data, with single-query and multi-record Bayesian linkage attacks using PLUTO as auxiliary information.

## Setup

```bash
conda env create -f environment.yml
conda activate geo-privacy
```

## Data

Downloaded automatically via NYC Open Data API:
- [DOHMH Restaurant Inspections](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)
- [PLUTO](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks) — download CSV manually and set `PLUTO_PATH` in the notebook

## Key Findings

All numbers come from one notebook run, saved in `results/*.csv`, except the Spearman ρ range, which comes from an earlier session (ε ≤ 0.01) and will be replaced on the next full run.

- The strongest prior (cuisine → land use) raises re-identification by 48% relative to nearest-building matching at ε = 0.01 (13.4% vs 9.0%), and by up to 89% at ε = 0.002
- The posterior-to-prior bound (Theorem 3.1) holds for 100% of records and sits more than two orders of magnitude above median empirical ratios (1.1 to 3.8)
- Within a setting, record-level vulnerability tracks noise displacement (Spearman ρ −0.89 to −0.96), but rates differ by borough under the same ε
- Multi-record attacks improve re-identification with diminishing returns (9.0% → 14.2% from 1 to 4 linked inspections at ε = 0.01); the composition bound n·ε overstates empirical degradation

## Reproducing the paper's numbers and figures

1. Run `geo-privacy-restaurant.ipynb` top to bottom in one session. Its last cell writes `results/*.csv`.
2. `python scripts/make_figures.py results paper` rebuilds every figure from those CSVs.
3. Rebuild `paper/main.tex` and update any number that moved.

`scripts/extract_notebook_results.py` recovered the CSVs from the saved outputs of the existing run. Only use it again if the notebook outputs are rerun without the export cell.

## Reference

Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013). Geo-indistinguishability: Differential privacy for location-based systems. *CCS '13*.
