import json
import re
import sys
from pathlib import Path

import pandas as pd

CONFIGS = ["spatial_only", "borough_only", "cuisine_landuse", "cuisine_units", "cuisine_year"]
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
EPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def cell_text(nb, idx):
    return "".join("".join(o.get("text", "")) for o in nb["cells"][idx].get("outputs", []))


def rows_after(text, start_marker, n_rows, skip=0):
    pos = text.index(start_marker)
    out = []
    for line in text[pos:].splitlines()[1:]:
        tok = line.split()
        if tok and re.match(r"^-?\d", tok[0]):
            out.append(tok)
            if len(out) == n_rows + skip:
                break
    return out[skip:]


def main(nb_path, out_dir):
    nb = json.load(open(nb_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = cell_text(nb, 13)
    t = t[t.index("=== SINGLE-QUERY RESULTS ==="):]
    first = rows_after(t, "=== SINGLE-QUERY", 6)
    second = rows_after(t, "spatial_only", 6)
    rows = []
    for a, b in zip(first, second):
        eps = float(a[0])
        vals = dict(zip(["borough_only", "cuisine_landuse", "cuisine_units", "cuisine_year"], map(float, a[1:5])))
        vals["spatial_only"] = float(b[1])
        for cfg in CONFIGS:
            rows.append({"epsilon": eps, "config": cfg, "reid_rate": vals[cfg]})
    single = pd.DataFrame(rows)
    single.to_csv(out / "single_query.csv", index=False)

    t = cell_text(nb, 16)
    rows = []
    for cfg in CONFIGS:
        for tok in rows_after(t, f"=== {cfg} ===", 7):
            n = int(tok[0])
            for eps, v in zip(EPS, tok[1:7]):
                rows.append({"epsilon": eps, "n_obs": n, "config": cfg, "reid_rate": float(v)})
    multi = pd.DataFrame(rows)
    t15 = cell_text(nb, 15)
    counts = {int(a): int(b) for a, b in rows_after(t15, "n_inspections", 9)}
    multi["n_restaurants"] = multi["n_obs"].map(counts)
    multi.to_csv(out / "multi_record.csv", index=False)

    score = [re.findall(r"eps=([\d.]+): score RMSE=([\d.]+), MAE=([\d.]+)", cell_text(nb, 19))][0]
    pd.DataFrame([(float(e), float(r), float(m)) for e, r, m in score],
                 columns=["epsilon", "rmse", "mae"]).to_csv(out / "utility_score.csv", index=False)
    cuis = re.findall(r"eps=([\d.]+): cuisine TVD mean=([\d.]+), median=([\d.]+)", cell_text(nb, 20))
    pd.DataFrame([(float(e), float(a), float(b)) for e, a, b in cuis],
                 columns=["epsilon", "tvd_mean", "tvd_median"]).to_csv(out / "utility_cuisine.csv", index=False)

    t = cell_text(nb, 25)
    t = t[t.index("=== POSTERIOR SUMMARY ==="):]
    rows = []
    for line in t.splitlines():
        tok = line.split()
        if tok and tok[0] in CONFIGS:
            rows.append({"config": tok[0], "epsilon": float(tok[1]), "median_posterior_true": float(tok[2]),
                         "median_ratio": float(tok[3]), "max_ratio": float(tok[4]),
                         "share_posterior_gt_0_5": float(tok[5]),
                         "bound_satisfied_pct": float(tok[6].rstrip("%"))})
    pd.DataFrame(rows).to_csv(out / "posterior_summary.csv", index=False)

    t = cell_text(nb, 22)
    rows = []
    for cfg in ["spatial_only", "cuisine_landuse"]:
        for tok in rows_after(t, f"=== {cfg} ===", 6):
            for b, v in zip(BOROUGHS, tok[1:6]):
                rows.append({"config": cfg, "epsilon": float(tok[0]), "borough": b, "reid_rate": float(v)})
    pd.DataFrame(rows).to_csv(out / "borough.csv", index=False)

    dens = re.findall(r"(\w[\w ]*?)\s*: avg candidates = (\d+), median = (\d+)", cell_text(nb, 23))
    pd.DataFrame([(b.strip(), int(a), int(m)) for b, a, m in dens],
                 columns=["borough", "mean_candidates", "median_candidates"]).to_csv(out / "candidate_density.csv", index=False)

    lift = re.findall(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", cell_text(nb, 33), re.M)
    pd.DataFrame([tuple(map(float, r)) for r in lift],
                 columns=["epsilon", "spatial_only", "cuisine_landuse", "abs_lift", "rel_lift_pct"]
                 ).to_csv(out / "prior_lift.csv", index=False)

    t6 = cell_text(nb, 6)
    tr = list(re.finditer(r"\nTraining: (\d+) visits, (\d+) restaurants", t6))[-1]
    te = list(re.finditer(r"\nTest: (\d+) visits, (\d+) restaurants", t6))[-1]
    summary = {
        "pluto_lots": int(re.search(r"PLUTO rows: (\d+)", cell_text(nb, 5)).group(1)),
        "train_visits": int(tr.group(1)), "train_restaurants": int(tr.group(2)),
        "train_joined_to_pluto": int(re.search(r"Joined: (\d+)", cell_text(nb, 9)).group(1)),
        "test_visits": int(te.group(1)), "test_restaurants": int(te.group(2)),
        "test_restaurants_2plus": int(re.search(r"2\+ inspections: (\d+)", t6).group(1)),
        "test_restaurants_3plus": int(re.search(r"3\+ inspections: (\d+)", t6).group(1)),
    }
    pd.Series(summary).rename_axis("quantity").rename("value").to_csv(out / "data_summary.csv")

    print(f"wrote {len(list(out.glob('*.csv')))} CSVs to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
