# make_appendix_metric_distributions.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def boolify(series):
    if series.dtype == bool:
        return series.astype(int)
    if str(series.dtype).startswith("int") or str(series.dtype).startswith("float"):
        return series.fillna(0).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map({
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0
    }).fillna(0).astype(int)

def infer_outcome(row):
    cv = int(row["clean_valid"])
    pv = int(row["photo_valid"])
    if cv == 1 and pv == 1:
        return "Stable"
    if cv == 1 and pv == 0:
        return "Degraded"
    if cv == 0 and pv == 0:
        return "Baseline failure"
    if cv == 0 and pv == 1:
        return "Photo recovery"
    return "Unknown"

def save_hist(series, title, xlabel, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_boxplot(df, col, out_path):
    groups = ["Stable", "Degraded", "Baseline failure", "Photo recovery"]
    data = [df.loc[df["outcome"] == g, col].dropna().values for g in groups if (df["outcome"] == g).any()]
    labels = [g for g in groups if (df["outcome"] == g).any()]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels)
    plt.title(f"{col} by outcome group")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["clean_valid"] = boolify(df["clean_valid"])
    df["photo_valid"] = boolify(df["photo_valid"])
    df["outcome"] = df.apply(infer_outcome, axis=1)

    if "clean_wall_cc" in df.columns and "photo_wall_cc" in df.columns:
        df["delta_wall_cc"] = df["photo_wall_cc"] - df["clean_wall_cc"]

    metric_cols = [c for c in [
        "clean_wall_cc", "photo_wall_cc", "delta_wall_cc",
        "clean_enclosure_count", "photo_enclosure_count",
        "clean_enclosed_free_ratio", "photo_enclosed_free_ratio",
        "clean_outside_free_ratio", "photo_outside_free_ratio",
        "clean_wall_area_ratio", "photo_wall_area_ratio"
    ] if c in df.columns]

    summary = df.groupby("outcome")[metric_cols].agg(["mean", "median", "std", "min", "max"])
    summary.to_csv(out_dir / "metric_summary_by_outcome.csv")

    df["is_degraded"] = ((df["clean_valid"] == 1) & (df["photo_valid"] == 0)).astype(int)
    dg_summary = df.groupby("is_degraded")[metric_cols].agg(["mean", "median", "std", "min", "max"])
    dg_summary.to_csv(out_dir / "metric_summary_degraded_vs_other.csv")

    if "clean_wall_cc" in df.columns:
        save_hist(df["clean_wall_cc"], "Clean wall_cc distribution", "clean_wall_cc", out_dir / "hist_clean_wall_cc.png")
        save_boxplot(df, "clean_wall_cc", out_dir / "box_clean_wall_cc_by_outcome.png")

    if "photo_wall_cc" in df.columns:
        save_hist(df["photo_wall_cc"], "Photo wall_cc distribution", "photo_wall_cc", out_dir / "hist_photo_wall_cc.png")
        save_boxplot(df, "photo_wall_cc", out_dir / "box_photo_wall_cc_by_outcome.png")

    if "delta_wall_cc" in df.columns:
        save_hist(df["delta_wall_cc"], "Delta wall_cc distribution", "photo_wall_cc - clean_wall_cc", out_dir / "hist_delta_wall_cc.png")
        save_boxplot(df, "delta_wall_cc", out_dir / "box_delta_wall_cc_by_outcome.png")

    print(f"Wrote outputs to {out_dir}")

if __name__ == "__main__":
    main()