import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon, shapiro, ttest_rel, binomtest
import numpy as np
import math

def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half

def mcnemar_exact(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    x01 = int(((a == 0) & (b == 1)).sum())
    x10 = int(((a == 1) & (b == 0)).sum())
    n = x01 + x10
    if n == 0:
        return x01, x10, 1.0
    k = min(x01, x10)
    p = min(1.0, 2 * binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return x01, x10, p

def cohens_d_paired(x, y):
    diff = np.asarray(y) - np.asarray(x)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return diff.mean() / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/app/base_comparison/comparison.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    req = ["base_valid", "ft_valid", "base_wall_cc", "ft_wall_cc"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    base_valid = df["base_valid"].astype(int).to_numpy()
    ft_valid = df["ft_valid"].astype(int).to_numpy()

    n = len(df)
    x01, x10, p = mcnemar_exact(base_valid, ft_valid)

    print("=== Base vs Fine-tuned validity ===")
    print(f"n                : {n}")
    print(f"base valid       : {base_valid.sum()}/{n}")
    print(f"fine-tuned valid : {ft_valid.sum()}/{n}")
    print(f"McNemar 0->1     : {x01}")
    print(f"McNemar 1->0     : {x10}")
    print(f"McNemar p        : {p:.6g}")

    for label, arr in [("base", base_valid), ("fine_tuned", ft_valid)]:
        lo, hi = wilson_ci(int(arr.sum()), n)
        print(f"{label} 95% CI    : [{lo:.6f}, {hi:.6f}]")

    x = df["base_wall_cc"].to_numpy()
    y = df["ft_wall_cc"].to_numpy()

    print("\n=== Base vs Fine-tuned wall_cc ===")
    print(f"base mean        : {x.mean():.4f}")
    print(f"fine-tuned mean  : {y.mean():.4f}")
    print(f"mean diff        : {(y - x).mean():.4f}")
    print(f"paired t-test p  : {ttest_rel(y, x, nan_policy='omit').pvalue:.6g}")
    try:
        w = wilcoxon(y, x, zero_method='wilcox', alternative='two-sided')
        print(f"Wilcoxon p       : {w.pvalue:.6g}")
    except Exception as e:
        print(f"Wilcoxon p       : failed ({e})")
    try:
        s = shapiro(y - x)
        print(f"Shapiro p        : {s.pvalue:.6g}")
    except Exception as e:
        print(f"Shapiro p        : failed ({e})")
    print(f"Cohen's d paired : {cohens_d_paired(x, y):.4f}")

if __name__ == "__main__":
    main()