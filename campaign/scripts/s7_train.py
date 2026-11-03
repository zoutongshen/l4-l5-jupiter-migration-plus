"""Stage 7: learn a snapshot rule for "Trojan" and score it, and the catalogue boxes, on the same labels.

Trains on the grid (test particles), so the model sees both sides of the edge, and
scores on held-out grid particles and on the real catalogue. The label is
``label_k<k>``; real objects are scored all together and with U <= 4.

Feature sets (trojans.definition):
  raw       the six osculating elements as catalogues publish them
  physical  a - a_J, e, I and the angles relative to Jupiter, as sin/cos
  minimal   a - a_J, e, I, sin/cos sigma: does anything beyond sigma matter?

Writes campaign/reports/<out-id>/: scores.csv, importance.csv, tree_<set>_d<depth>.txt,
map_slice.png, summary.json; and data/runs/<out-id>/models_k<k>.pkl.

    python campaign/scripts/s7_train.py --out-id definition_v1 --k 5
"""

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from trojans import definition, paths
from trojans.outputs import provenance, write_json

FEATURE_SETS = {
    "raw": definition.RAW,
    "physical": definition.PHYSICAL,
    "minimal": ["da", "e", "inc", "sin_sigma", "cos_sigma"],
}
DEPTHS = (2, 3, 4, 6, 8)
TROJAN = {"L4", "L5"}


def errors(true, pred) -> dict:
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    t_troj = np.isin(true, list(TROJAN))
    p_troj = np.isin(pred, list(TROJAN))
    return {
        "n": len(true),
        "n_trojan": int(t_troj.sum()),
        "accuracy": float(np.mean(true == pred)) if len(true) else np.nan,
        "false_trojan": int((p_troj & ~t_troj).sum()),
        "missed_trojan": int((t_troj & ~p_troj).sum()),
        "wrong_swarm": int((t_troj & p_troj & (true != pred)).sum()),
        # Trojan vs not, swarm ignored: the map is mostly "none", so accuracy alone hides the edge.
        "precision": float((p_troj & t_troj).sum() / p_troj.sum()) if p_troj.any() else np.nan,
        "recall": float((p_troj & t_troj).sum() / t_troj.sum()) if t_troj.any() else np.nan,
    }


def score_sets(data: pd.DataFrame, target: str, test_ids) -> dict:
    """The rows each rule is scored on."""
    grid = data[data["source"] == "grid"]
    out = {"grid (held out)": grid[grid["spkid"].isin(test_ids)]}
    real = data[(data["source"] == "real") & data[target].notna()]
    if len(real):
        good = pd.to_numeric(real["condition_code"], errors="coerce") <= 4
        out.update({"real, all": real, "real, U <= 4": real[good]})
    return out


def map_slice(grid: pd.DataFrame, target: str, path) -> None:
    """Where the label flips in (sigma, a - a_J) for near-circular, near-planar particles."""
    cold = grid[(grid["e"] < 0.1) & (grid["inc"] < 15)]
    colors = {"L4": "#2a78d6", "L5": "#eb6834", "none": "#b9b8b0"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for lab in ("none", "L4", "L5"):
        s = cold[cold[target] == lab]
        ax.scatter(s["sigma"], s["da"], s=5, c=colors[lab], label=f"{lab} ({len(s)})", lw=0)
    ax.set_xlabel("σ = λ − λ_J at t0 [deg]")
    ax.set_ylabel("a − a_J at t0 [au]")
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_title(f"Map particles with e < 0.1 and I < 15°, labelled by {target}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-id", required=True, help="reads data/runs/<out-id>/dataset.parquet")
    parser.add_argument("--k", type=int, default=5, help="label = first k windows of 10 kyr")
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()

    target = f"label_k{args.k}"
    data = pd.read_parquet(paths.run_dir(args.out_id) / "dataset.parquet")
    if target not in data:
        raise SystemExit(f"{target} not in the dataset; rebuild it with --k {args.k}")
    grid = data[(data["source"] == "grid") & data[target].notna()]
    if grid.empty:
        raise SystemExit("no grid rows: the model needs test particles to see both sides of the edge")
    train, test = train_test_split(grid, test_size=0.25, random_state=args.seed, stratify=grid[target])
    sets = score_sets(data, target, set(test["spkid"]))

    out = paths.report_dir(args.out_id)
    out.mkdir(parents=True, exist_ok=True)
    rows, importance, fitted = [], [], {}
    for name, cols in FEATURE_SETS.items():
        models = {f"tree d{d}": DecisionTreeClassifier(max_depth=d, random_state=args.seed) for d in DEPTHS}
        models["boosted"] = HistGradientBoostingClassifier(random_state=args.seed)
        for label, model in models.items():
            model.fit(train[cols], train[target])
            fitted[f"{label} · {name}"] = {"model": model, "features": cols}
            for set_name, rows_ in sets.items():
                if len(rows_):
                    rows.append({"rule": f"{label} · {name}", "features": name, "scored_on": set_name,
                                 **errors(rows_[target], model.predict(rows_[cols]))})
            if label == "tree d4":
                (out / f"tree_{name}_d4.txt").write_text(export_text(model, feature_names=cols, decimals=3))
            if label == "boosted":
                imp = permutation_importance(model, test[cols], test[target], n_repeats=5, random_state=args.seed)
                importance += [{"features": name, "feature": c, "importance": m, "std": s}
                               for c, m, s in zip(cols, imp.importances_mean, imp.importances_std)]
    for box_name, box in definition.BOXES.items():
        for set_name, rows_ in sets.items():
            if len(rows_):
                rows.append({"rule": box_name, "features": "box", "scored_on": set_name,
                             **errors(rows_[target], definition.box_label(rows_, box))})

    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False, float_format="%.5f")
    pd.DataFrame(importance).sort_values(["features", "importance"], ascending=[True, False]).to_csv(
        out / "importance.csv", index=False, float_format="%.5f")
    map_slice(grid, target, out / "map_slice.png")
    # The fitted rules, for s7_screen.py. Heavy-ish, so in the data root, not the repo.
    with open(paths.run_dir(args.out_id) / f"models_k{args.k}.pkl", "wb") as f:
        pickle.dump({"target": target, "models": fitted}, f)
    write_json(out / "summary.json", {"target": target, "n_train": len(train), "n_test": len(test),
                                      "label_counts_grid": grid[target].value_counts().to_dict(),
                                      **provenance()})
    with pd.option_context("display.width", 160):
        print(scores.pivot_table(index="rule", columns="scored_on", values=["precision", "recall"]).round(3).to_string())


if __name__ == "__main__":
    main()
