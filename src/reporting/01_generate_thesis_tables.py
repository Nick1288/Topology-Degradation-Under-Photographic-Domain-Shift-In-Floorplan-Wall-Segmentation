import argparse
import math
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    ttest_rel,
    wilcoxon,
    shapiro,
    friedmanchisquare,
    chi2,
    binomtest,
)


# ----------------------------
# Basic helpers
# ----------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def mcnemar_exact_from_vectors(a, b):
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    x01 = int(((a == 0) & (b == 1)).sum())
    x10 = int(((a == 1) & (b == 0)).sum())
    n = x01 + x10
    if n == 0:
        return {"x01": x01, "x10": x10, "p": 1.0}
    k = min(x01, x10)
    p = min(1.0, 2 * binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return {"x01": x01, "x10": x10, "p": p}


def cohens_d_paired(x, y):
    diff = np.asarray(y) - np.asarray(x)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return diff.mean() / sd


def cochrans_q_test(binary_matrix: np.ndarray):
    """
    binary_matrix shape: (n_subjects, k_conditions), values 0/1
    """
    X = np.asarray(binary_matrix).astype(int)
    n, k = X.shape
    col_sums = X.sum(axis=0)
    row_sums = X.sum(axis=1)
    T = col_sums.sum()

    denom = k * T - np.sum(row_sums ** 2)
    if denom == 0:
        return {"Q": np.nan, "df": k - 1, "p": np.nan}

    Q = (k - 1) * (k * np.sum(col_sums ** 2) - T ** 2) / denom
    p = chi2.sf(Q, df=k - 1)
    return {"Q": float(Q), "df": int(k - 1), "p": float(p)}


def bootstrap_metrics(y_true, y_pred, silent_mask=None, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if silent_mask is not None:
        silent_mask = np.asarray(silent_mask).astype(bool)

    n = len(y_true)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        sm = silent_mask[idx] if silent_mask is not None else None

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if sm is None:
            silent_recall = np.nan
        else:
            yt_s = yt[sm]
            yp_s = yp[sm]
            denom = int((yt_s == 1).sum())
            silent_recall = int(((yt_s == 1) & (yp_s == 1)).sum()) / denom if denom else np.nan

        vals.append([precision, recall, f1, silent_recall])

    arr = np.array(vals, dtype=float)
    mean = np.nanmean(arr, axis=0)
    lo = np.nanpercentile(arr, 2.5, axis=0)
    hi = np.nanpercentile(arr, 97.5, axis=0)
    return mean, lo, hi


def add_metric_row(rows, name, y_true, y_pred, silent_mask=None):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if silent_mask is None:
        silent_recall = np.nan
    else:
        yt_s = y_true[silent_mask]
        yp_s = y_pred[silent_mask]
        denom = int((yt_s == 1).sum())
        silent_recall = int(((yt_s == 1) & (yp_s == 1)).sum()) / denom if denom else np.nan

    mean, lo, hi = bootstrap_metrics(y_true, y_pred, silent_mask=silent_mask)

    rows.append({
        "method": name,
        "precision": precision, "precision_lo": lo[0], "precision_hi": hi[0],
        "recall": recall, "recall_lo": lo[1], "recall_hi": hi[1],
        "f1": f1, "f1_lo": lo[2], "f1_hi": hi[2],
        "silent_recall": silent_recall, "silent_recall_lo": lo[3], "silent_recall_hi": hi[3],
        "tp": tp, "fp": fp, "fn": fn,
    })


# ----------------------------
# Loaders
# ----------------------------
def load_paired_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing paired_results.csv: {path}")
    df = pd.read_csv(path)

    required = [
        "clean_valid", "photo_valid",
        "clean_wall_cc", "photo_wall_cc",
        "photo_enclosed_free_ratio", "photo_outside_free_ratio",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"paired_results.csv missing required columns: {missing}")
    return df


def load_mitigation_results(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Long format expected: plan_id, method, valid
    if {"plan_id", "method", "valid"}.issubset(df.columns):
        out = df[["plan_id", "method", "valid"]].copy()
        out["method"] = out["method"].astype(str)
        out["valid"] = out["valid"].astype(int)
        return out

    # Wide format fallback
    if "plan_id" in df.columns:
        possible = [c for c in df.columns if c != "plan_id"]
        binary_like = []
        for c in possible:
            vals = set(pd.Series(df[c]).dropna().astype(int).unique().tolist()) if len(df[c].dropna()) else set()
            if vals.issubset({0, 1}):
                binary_like.append(c)

        if binary_like:
            out = df.melt(id_vars="plan_id", value_vars=binary_like, var_name="method", value_name="valid")
            out["valid"] = out["valid"].astype(int)
            return out

    return None


def load_base_ft_results(base_dir: Path) -> pd.DataFrame | None:
    if not base_dir.exists():
        return None

    csvs = list(base_dir.glob("*.csv"))
    if not csvs:
        return None

    for p in csvs:
        df = pd.read_csv(p)

        # wide format
        if {"plan_id", "base_valid", "finetuned_valid"}.issubset(df.columns):
            out = df[["plan_id", "base_valid", "finetuned_valid"]].copy()
            out["base_valid"] = out["base_valid"].astype(int)
            out["finetuned_valid"] = out["finetuned_valid"].astype(int)
            return out

        # long format
        if {"plan_id", "model", "valid"}.issubset(df.columns):
            piv = df.pivot(index="plan_id", columns="model", values="valid").reset_index()
            cols_lower = {c.lower(): c for c in piv.columns}
            base_col = None
            ft_col = None
            for k, v in cols_lower.items():
                if "base" in k:
                    base_col = v
                if "fine" in k or "finetune" in k:
                    ft_col = v
            if base_col and ft_col:
                out = piv[["plan_id", base_col, ft_col]].copy()
                out.columns = ["plan_id", "base_valid", "finetuned_valid"]
                out["base_valid"] = out["base_valid"].astype(int)
                out["finetuned_valid"] = out["finetuned_valid"].astype(int)
                return out

    return None


# ----------------------------
# Section 1: Experiment 1
# ----------------------------
def analyze_experiment1(df: pd.DataFrame, out_dir: Path):
    lines = []
    rows = []

    n = len(df)
    clean_valid = df["clean_valid"].astype(int).to_numpy()
    photo_valid = df["photo_valid"].astype(int).to_numpy()

    both_valid = int(((clean_valid == 1) & (photo_valid == 1)).sum())
    clean_only = int(((clean_valid == 1) & (photo_valid == 0)).sum())
    photo_only = int(((clean_valid == 0) & (photo_valid == 1)).sum())
    both_invalid = int(((clean_valid == 0) & (photo_valid == 0)).sum())

    rates = [
        ("clean_valid_rate", int(clean_valid.sum()), n),
        ("photo_valid_rate", int(photo_valid.sum()), n),
        ("photo_specific_degraded_rate", clean_only, n),
        ("photo_failure_rate", int((photo_valid == 0).sum()), n),
    ]

    # silent failures among all photo-invalid
    photo_invalid = df[df["photo_valid"].astype(int) == 0].copy()
    if len(photo_invalid) > 0:
        silent_mask = photo_invalid["photo_enclosed_free_ratio"].to_numpy() > 0.30
        silent_count = int(silent_mask.sum())
        rates.append(("silent_failure_share_among_photo_invalid", silent_count, len(photo_invalid)))
    else:
        silent_count = 0

    for name, k, denom in rates:
        lo, hi = wilson_ci(k, denom)
        rows.append({
            "metric": name,
            "count": k,
            "n": denom,
            "rate": k / denom if denom else np.nan,
            "ci_lo": lo,
            "ci_hi": hi,
        })

    mcn = mcnemar_exact_from_vectors(clean_valid, photo_valid)

    lines.append("=== Experiment 1: validity shift ===")
    lines.append(f"n = {n}")
    lines.append(f"both valid      : {both_valid}")
    lines.append(f"clean only valid: {clean_only}")
    lines.append(f"photo only valid: {photo_only}")
    lines.append(f"both invalid    : {both_invalid}")
    lines.append(f"McNemar exact p : {mcn['p']:.6g}")
    lines.append("")

    # All-plan wall_cc
    x = df["clean_wall_cc"].to_numpy()
    y = df["photo_wall_cc"].to_numpy()
    t_all = ttest_rel(y, x, nan_policy="omit")
    try:
        w_all = wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
        w_all_p = w_all.pvalue
    except Exception:
        w_all_p = np.nan
    try:
        sh_all = shapiro(y - x)
        sh_all_p = sh_all.pvalue
        sh_all_w = sh_all.statistic
    except Exception:
        sh_all_p = np.nan
        sh_all_w = np.nan
    d_all = cohens_d_paired(x, y)

    lines.append("=== Experiment 1: wall_cc shift (all plans) ===")
    lines.append(f"clean mean      : {x.mean():.4f}")
    lines.append(f"photo mean      : {y.mean():.4f}")
    lines.append(f"mean diff       : {(y - x).mean():.4f}")
    lines.append(f"paired t-test p : {t_all.pvalue:.6g}")
    lines.append(f"Wilcoxon p      : {w_all_p:.6g}" if not np.isnan(w_all_p) else "Wilcoxon p      : nan")
    lines.append(f"Shapiro W       : {sh_all_w:.6g}" if not np.isnan(sh_all_w) else "Shapiro W       : nan")
    lines.append(f"Shapiro p       : {sh_all_p:.6g}" if not np.isnan(sh_all_p) else "Shapiro p       : nan")
    lines.append(f"Cohen's d paired: {d_all:.4f}")
    lines.append("")

    # Degraded subset
    degraded = df[(df["clean_valid"].astype(int) == 1) & (df["photo_valid"].astype(int) == 0)].copy()
    if len(degraded) > 0:
        x2 = degraded["clean_wall_cc"].to_numpy()
        y2 = degraded["photo_wall_cc"].to_numpy()

        t_deg = ttest_rel(y2, x2, nan_policy="omit")
        try:
            w_deg = wilcoxon(y2, x2, zero_method="wilcox", alternative="two-sided")
            w_deg_p = w_deg.pvalue
            w_deg_stat = w_deg.statistic
        except Exception:
            w_deg_p = np.nan
            w_deg_stat = np.nan

        try:
            sh_deg = shapiro(y2 - x2)
            sh_deg_p = sh_deg.pvalue
            sh_deg_w = sh_deg.statistic
        except Exception:
            sh_deg_p = np.nan
            sh_deg_w = np.nan

        d_deg = cohens_d_paired(x2, y2)

        lines.append("=== Experiment 1: degraded subset only ===")
        lines.append(f"n               : {len(degraded)}")
        lines.append(f"clean mean      : {x2.mean():.4f}")
        lines.append(f"photo mean      : {y2.mean():.4f}")
        lines.append(f"mean diff       : {(y2 - x2).mean():.4f}")
        lines.append(f"paired t-test   : t={t_deg.statistic:.6g}, p={t_deg.pvalue:.6g}")
        lines.append(
            f"Wilcoxon        : W={w_deg_stat:.6g}, p={w_deg_p:.6g}" if not np.isnan(w_deg_p) else "Wilcoxon        : nan"
        )
        lines.append(
            f"Shapiro         : W={sh_deg_w:.6g}, p={sh_deg_p:.6g}" if not np.isnan(sh_deg_p) else "Shapiro         : nan"
        )
        lines.append(f"Cohen's d paired: {d_deg:.4f}")
        lines.append("")

        # Figures: histogram + boxplot
        fig1 = plt.figure(figsize=(7, 4.5))
        delta = y2 - x2
        plt.hist(delta, bins=20)
        plt.xlabel("Δ wall_cc (photo - clean)")
        plt.ylabel("Count")
        plt.title("Distribution of wall_cc change in degraded cases")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_delta_wallcc_hist.png", dpi=220, bbox_inches="tight")
        plt.close(fig1)

        fig2 = plt.figure(figsize=(6, 4.5))
        plt.boxplot([x2, y2], labels=["Clean", "Photo"])
        plt.ylabel("wall_cc")
        plt.title("Clean vs photo wall_cc in degraded cases")
        plt.tight_layout()
        plt.savefig(out_dir / "fig_wallcc_boxplot_degraded.png", dpi=220, bbox_inches="tight")
        plt.close(fig2)

    pd.DataFrame(rows).to_csv(out_dir / "table_key_proportion_cis.csv", index=False)
    (out_dir / "experiment1_stats.txt").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Section 2: Gate comparisons + overlap
# ----------------------------
def analyze_gate_tables(df: pd.DataFrame, out_dir: Path, thr_wall_cc: int, thr_enclosed: float, thr_outside: float):
    lines = []

    y_true = ((df["clean_valid"].astype(int) == 1) & (df["photo_valid"].astype(int) == 0)).astype(int).to_numpy()
    silent_mask = ((df["photo_valid"].astype(int) == 0) & (df["photo_enclosed_free_ratio"] > 0.30)).to_numpy()

    photo_wall = df["photo_wall_cc"].to_numpy()
    photo_encl = df["photo_enclosed_free_ratio"].to_numpy()
    photo_out = df["photo_outside_free_ratio"].to_numpy()
    photo_count = df["photo_enclosure_count"].to_numpy() if "photo_enclosure_count" in df.columns else np.zeros(len(df), dtype=int)

    # flag degraded = 1 when rule says invalid
    pred = {
        "full_gate": (
            (photo_count <= 0)
            | (photo_encl <= thr_enclosed)
            | (photo_out >= thr_outside)
            | (photo_wall >= thr_wall_cc)
        ).astype(int),
        "wall_cc_only": (photo_wall >= thr_wall_cc).astype(int),
        "enclosed_ratio_only": (photo_encl <= thr_enclosed).astype(int),
        "outside_ratio_only": (photo_out >= thr_outside).astype(int),
        "enclosed_and_outside": ((photo_encl <= thr_enclosed) | (photo_out >= thr_outside)).astype(int),
        "enclosure_count_only": (photo_count <= 0).astype(int),
    }

    if "photo_wall_area_ratio" in df.columns:
        # Optional; threshold is conservative and may flag none on this dataset
        pred["wall_area_only"] = (df["photo_wall_area_ratio"].to_numpy() <= 0.01).astype(int)

    rows_cmp = []
    for name, y_pred in pred.items():
        add_metric_row(rows_cmp, name, y_true, y_pred, silent_mask=silent_mask)
    cmp_df = pd.DataFrame(rows_cmp).sort_values("f1", ascending=False)
    cmp_df.to_csv(out_dir / "table_gate_comparison_bootstrap_cis.csv", index=False)

    # Component ablation
    pred_ab = {
        "full_gate": pred["full_gate"],
        "remove_enclosure_count": (
            (photo_encl <= thr_enclosed)
            | (photo_out >= thr_outside)
            | (photo_wall >= thr_wall_cc)
        ).astype(int),
        "remove_enclosed_ratio": (
            (photo_count <= 0)
            | (photo_out >= thr_outside)
            | (photo_wall >= thr_wall_cc)
        ).astype(int),
        "remove_outside_ratio": (
            (photo_count <= 0)
            | (photo_encl <= thr_enclosed)
            | (photo_wall >= thr_wall_cc)
        ).astype(int),
        "remove_wall_cc": (
            (photo_count <= 0)
            | (photo_encl <= thr_enclosed)
            | (photo_out >= thr_outside)
        ).astype(int),
    }

    rows_ab = []
    for name, y_pred in pred_ab.items():
        add_metric_row(rows_ab, name, y_true, y_pred, silent_mask=silent_mask)
    ab_df = pd.DataFrame(rows_ab).sort_values("f1", ascending=False)
    ab_df.to_csv(out_dir / "table_component_ablation_bootstrap_cis.csv", index=False)

    # Overlap breakdown among degraded cases
    degraded = df[(df["clean_valid"].astype(int) == 1) & (df["photo_valid"].astype(int) == 0)].copy()
    if len(degraded) > 0:
        viol_wall = degraded["photo_wall_cc"].to_numpy() >= thr_wall_cc
        viol_encl = degraded["photo_enclosed_free_ratio"].to_numpy() <= thr_enclosed
        viol_out = degraded["photo_outside_free_ratio"].to_numpy() >= thr_outside

        combo_rows = []
        for w, e, o in zip(viol_wall, viol_encl, viol_out):
            label_parts = []
            if w:
                label_parts.append("wall_cc")
            if e:
                label_parts.append("enclosed")
            if o:
                label_parts.append("outside")
            label = "+".join(label_parts) if label_parts else "none"
            combo_rows.append(label)

        overlap = pd.Series(combo_rows).value_counts().rename_axis("pattern").reset_index(name="count")
        overlap.to_csv(out_dir / "table_failure_overlap_breakdown.csv", index=False)

        lines.append("=== Failure overlap breakdown among degraded cases ===")
        for _, r in overlap.iterrows():
            lines.append(f"{r['pattern']}: {r['count']}")

    (out_dir / "gate_and_overlap_notes.txt").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Section 3: Preprocessing mitigation
# ----------------------------
def analyze_preprocessing(mit_df: pd.DataFrame | None, out_dir: Path):
    if mit_df is None:
        (out_dir / "preprocessing_stats.txt").write_text(
            "mitigation.csv not found or could not be parsed; preprocessing stats skipped.\n",
            encoding="utf-8",
        )
        return

    # normalize method names
    m = mit_df["method"].astype(str).str.lower().str.strip()
    m = m.replace({
        "none (baseline)": "none",
        "baseline": "none",
        "contrast enhancement (clahe)": "clahe",
        "contrast": "clahe",
        "deskewing (hough rotation)": "deskew",
        "rotation": "deskew",
        "both combined": "both",
        "combined": "both",
    })
    mit_df = mit_df.copy()
    mit_df["method_norm"] = m

    preferred_order = ["none", "clahe", "deskew", "both"]
    present = [x for x in preferred_order if x in set(mit_df["method_norm"])]

    piv = mit_df.pivot_table(index="plan_id", columns="method_norm", values="valid", aggfunc="first")
    piv = piv[[c for c in preferred_order if c in piv.columns]].dropna()

    lines = []
    lines.append("=== Preprocessing mitigation ===")
    lines.append(f"plans with complete method coverage: {len(piv)}")
    lines.append("")

    rows = []
    for method in piv.columns:
        k = int(piv[method].sum())
        n = len(piv)
        lo, hi = wilson_ci(k, n)
        rows.append({
            "method": method,
            "count_valid": k,
            "n": n,
            "rate": k / n if n else np.nan,
            "ci_lo": lo,
            "ci_hi": hi,
        })

    pd.DataFrame(rows).to_csv(out_dir / "table_preprocessing_recovery_cis.csv", index=False)

    if piv.shape[1] >= 3:
        q = cochrans_q_test(piv.to_numpy())
        lines.append(f"Cochran's Q: Q={q['Q']:.6g}, df={q['df']}, p={q['p']:.6g}")
        lines.append("")

    lines.append("Pairwise McNemar tests:")
    pair_rows = []
    for a, b in itertools.combinations(piv.columns.tolist(), 2):
        res = mcnemar_exact_from_vectors(piv[a].to_numpy(), piv[b].to_numpy())
        pair_rows.append({
            "method_a": a, "method_b": b,
            "a0_b1": res["x01"], "a1_b0": res["x10"], "p": res["p"]
        })
        lines.append(f"{a} vs {b}: 0->1={res['x01']}, 1->0={res['x10']}, p={res['p']:.6g}")

    pd.DataFrame(pair_rows).to_csv(out_dir / "table_preprocessing_pairwise_mcnemar.csv", index=False)
    (out_dir / "preprocessing_stats.txt").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Section 4: Base vs fine-tuned
# ----------------------------
def analyze_base_finetuned(base_ft_df: pd.DataFrame | None, out_dir: Path):
    if base_ft_df is None:
        (out_dir / "base_vs_finetuned_stats.txt").write_text(
            "No paired base/fine-tuned CSV found in /app/base_comparison; skipped inferential test.\n",
            encoding="utf-8",
        )
        return

    base = base_ft_df["base_valid"].astype(int).to_numpy()
    ft = base_ft_df["finetuned_valid"].astype(int).to_numpy()
    n = len(base_ft_df)

    res = mcnemar_exact_from_vectors(base, ft)

    rows = []
    for name, vec in [("base", base), ("finetuned", ft)]:
        k = int(vec.sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"model": name, "count_valid": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(out_dir / "table_base_vs_finetuned_cis.csv", index=False)

    lines = []
    lines.append("=== Base vs fine-tuned ===")
    lines.append(f"n = {n}")
    lines.append(f"base valid      : {int(base.sum())}")
    lines.append(f"finetuned valid : {int(ft.sum())}")
    lines.append(f"McNemar exact   : 0->1={res['x01']}, 1->0={res['x10']}, p={res['p']:.6g}")

    (out_dir / "base_vs_finetuned_stats.txt").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-csv", default="/app/paired_eval_out/paired_results.csv")
    ap.add_argument("--mitigation-csv", default="/app/mitigation_results/mitigation.csv")
    ap.add_argument("--base-comparison-dir", default="/app/base_comparison")
    ap.add_argument("--out-dir", default="/app/final_stats_out")
    ap.add_argument("--thr-wall-cc", type=int, default=30)
    ap.add_argument("--thr-enclosed", type=float, default=0.25)
    ap.add_argument("--thr-outside", type=float, default=0.75)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1
    paired_df = load_paired_results(Path(args.paired_csv))
    analyze_experiment1(paired_df, out_dir)

    # Gate tables + overlap
    analyze_gate_tables(
        paired_df, out_dir,
        thr_wall_cc=args.thr_wall_cc,
        thr_enclosed=args.thr_enclosed,
        thr_outside=args.thr_outside,
    )

    # Preprocessing
    mit_df = load_mitigation_results(Path(args.mitigation_csv))
    analyze_preprocessing(mit_df, out_dir)

    # Base vs fine-tuned
    base_ft_df = load_base_ft_results(Path(args.base_comparison_dir))
    analyze_base_finetuned(base_ft_df, out_dir)

    manifest = {
        "paired_csv": args.paired_csv,
        "mitigation_csv": args.mitigation_csv,
        "base_comparison_dir": args.base_comparison_dir,
        "thresholds": {
            "wall_cc": args.thr_wall_cc,
            "enclosed": args.thr_enclosed,
            "outside": args.thr_outside,
        },
        "outputs": sorted([p.name for p in out_dir.iterdir()]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()