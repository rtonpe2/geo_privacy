# Realized Risk and Uneven Protection Under Geo-Indistinguishability

Code, results and paper for an evaluation of the planar Laplace mechanism (geo-indistinguishability; Andrés et al., 2013) on New York City restaurant inspection records. A simulated noisy release is attacked with a Bayesian linkage attack that uses the PLUTO tax lot database as its reference dataset and learns its priors from earlier, unprotected releases.

## Setup

```bash
conda env create -f environment.yml
conda activate geo-privacy
```

Package versions are pinned to the ones that produced the reported results.

## Data

- **DOHMH Restaurant Inspection Results** ([NYC Open Data](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)). The snapshot used in the paper (downloaded October 2026) is included in `data/`, and the notebook loads it instead of calling the API. The public dataset only keeps recent inspections, so a fresh download would return different records; delete the files in `data/` only if you want to rerun on current data.
- **PLUTO, version 26v2** ([NYC Open Data](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks)). Download the CSV and set `PLUTO_PATH` in the notebook's config cell.
- **2020 Census Tracts** ([NYC Open Data](https://data.cityofnewyork.us/City-Government/2020-Census-Tracts/63ge-mke6)). Download the GeoJSON and set `TRACTS_PATH` in the config cell.

Training data covers inspections from 2020 through 2024, and test data covers January 1, 2025 to October 9, 2026.

## Key findings

All numbers come from `results/*.csv`, at ε = 0.01 unless noted.

- Building attributes add little to an attacker who knows only locations (21.7% re-identification against 15.9% for location alone), while knowing which lots previously hosted a restaurant of the same cuisine raises re-identification to 66.7%. The two negative-control priors stay within about one percentage point of the location-only baseline.
- The gain from restaurant history comes almost entirely from restaurants that appeared in earlier releases (83.7% of continuing restaurants re-identified, against 12.6% of new ones).
- Protection is uneven. A record is exposed when no similar lot lies within reach of the noise, so re-identification rises with the distance to the nearest similar lot. Against a location-only attacker, Staten Island and the Bronx are the least protected boroughs and Brooklyn the most protected. Under the history priors, Manhattan becomes the most protected.
- The posterior-to-prior bound (Andrés et al., Theorem 3.1) holds for every record and sits two to three orders of magnitude above the empirical ratios.
- Linking repeated inspections degrades privacy far less than the nε composition bound implies (29.0% for four linked records, against 54.2% for a single release at ε = 0.04).
- Mean tract-level cuisine distortion (total variation distance) is 0.15, falling from 0.45 at ε = 0.001 to 0.02 at ε = 0.05.

## Reproducing the results

1. Run `geo-privacy-restaurant.ipynb` top to bottom in one session. It writes every results table to `results/`, and the land-use heatmap to `paper/`.
2. Run `python scripts/make_figures.py results paper` to rebuild the remaining figures from those tables.
3. Compile `paper/main.tex`.

The noise draws use fixed seeds, so a rerun on the included data snapshot, PLUTO 26v2 and the pinned environment reproduces the reported numbers.

## Repository layout

- `geo-privacy-restaurant.ipynb`: data loading, priors, single-query attack, look-alike spacing, multi-record attack, posterior analysis, utility, borough analysis, and the export of all results.
- `data/`: the inspection snapshot used in the paper.
- `results/`: the tables behind every number in the paper.
- `scripts/make_figures.py`: builds the paper's figures from `results/`.
- `paper/`: the paper source, compiled PDF and figures.

## Reference

Andrés, M. E., Bordenabe, N. E., Chatzikokolakis, K., & Palamidessi, C. (2013). Geo-indistinguishability: Differential privacy for location-based systems. *Proceedings of the 2013 ACM SIGSAC Conference on Computer and Communications Security (CCS '13)*, 901–914.
