import argparse
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_rule_preds(df, wall_thr=30, enclosed_thr=0.25, outside_thr=0.75, enclosure_thr=1):
    preds = {}

    if "photo_enclosure_count" in df.columns:
        preds["enclosure_count_only"] = (df["photo_enclosure_count"] > 0).astype(int)

    if "photo_enclosed_free_ratio" in df.columns:
        preds["enclosed_ratio_only"] = (df["photo_enclosed_free_ratio"] > enclosed_thr).astype(int)

    if "photo_outside_free_ratio" in df.columns:
        preds["outside_ratio_only"] = (df["photo_outside_free_ratio"] < outside_thr).astype(int)

    if "photo_wall_cc" in df.columns:
        preds["wall_cc_only"] = (df["photo_wall_cc"] < wall_thr).astype(int)

    if "photo_wall_area_ratio" in df.columns:
        # heuristic: nonzero wall area retained
        preds["wall_area_only"] = (df["photo_wall_area_ratio"] > 0).astype(int)

    # pairwise / combined gates
    if {"photo_wall_cc", "photo_enclosed_free_ratio"}.issubset(df.columns):
        preds["wall_cc_and_enclosed"] = (
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr)
        ).astype(int)

    if {"photo_wall_cc", "photo_outside_free_ratio"}.issubset(df.columns):
        preds["wall_cc_and_outside"] = (
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int)

    if {"photo_enclosed_free_ratio", "photo_outside_free_ratio"}.issubset(df.columns):
        preds["enclosed_and_outside"] = (
            (df["photo_enclosed_free_ratio"] > enclosed_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int)

    if {"photo_wall_cc", "photo_enclosed_free_ratio", "photo_outside_free_ratio"}.issubset(df.columns):
        preds["wall_cc_enclosed_outside"] = (
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int)

    if {"photo_enclosure_count", "photo_wall_cc", "photo_enclosed_free_ratio", "photo_outside_free_ratio"}.issubset(df.columns):
        preds["full_gate"] = (
            (df["photo_enclosure_count"] > 0) &
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int)

    return preds


def evaluate_rules(df, y_true, silent_mask, out_csv, wall_thr=30, enclosed_thr=0.25, outside_thr=0.75):
    preds = make_rule_preds(df, wall_thr, enclosed_thr, outside_thr)

    rows = []
    for name, pred_valid in preds.items():
        # Convert valid prediction to "broken detected" prediction
        pred_broken = 1 - pred_valid.astype(int)

        stats = confusion(y_true, pred_broken)

        silent_total = int(silent_mask.sum())
        silent_captured = int(((silent_mask == 1) & (pred_broken == 1)).sum())
        silent_missed = int(((silent_mask == 1) & (pred_broken == 0)).sum())
        silent_recall = safe_div(silent_captured, silent_total)

        rows.append({
            "method": name,
            **stats,
            "silent_total": silent_total,
            "silent_captured": silent_captured,
            "silent_missed": silent_missed,
            "silent_recall": silent_recall,
        })

    out_df = pd.DataFrame(rows).sort_values(["f1", "silent_recall", "recall"], ascending=False)
    out_df.to_csv(out_csv, index=False)
    return out_df


def threshold_ablation(df, y_true, silent_mask, out_csv, out_plot, thr_values):
    rows = []

    for thr in thr_values:
        pred_valid = (df["photo_wall_cc"] < thr).astype(int)
        pred_broken = 1 - pred_valid

        stats = confusion(y_true, pred_broken)

        silent_total = int(silent_mask.sum())
        silent_captured = int(((silent_mask == 1) & (pred_broken == 1)).sum())
        silent_recall = safe_div(silent_captured, silent_total)

        rows.append({
            "threshold": thr,
            **stats,
            "silent_total": silent_total,
            "silent_captured": silent_captured,
            "silent_recall": silent_recall,
        })

    out_df = pd.DataFrame(rows).sort_values("threshold")
    out_df.to_csv(out_csv, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(out_df["threshold"], out_df["precision"], marker="o", label="precision")
    plt.plot(out_df["threshold"], out_df["recall"], marker="o", label="recall")
    plt.plot(out_df["threshold"], out_df["f1"], marker="o", label="f1")
    plt.plot(out_df["threshold"], out_df["silent_recall"], marker="o", label="silent_recall")
    best = out_df.loc[out_df["f1"].idxmax()]
    plt.axvline(best["threshold"], linestyle="--", label=f"best={int(best['threshold'])}")
    plt.xlabel("wall_cc threshold")
    plt.ylabel("Score")
    plt.title("wall_cc Threshold Ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=220, bbox_inches="tight")
    plt.close()

    return out_df


def component_ablation(df, y_true, silent_mask, out_csv, wall_thr=30, enclosed_thr=0.25, outside_thr=0.75):
    required = {"photo_enclosure_count", "photo_wall_cc", "photo_enclosed_free_ratio", "photo_outside_free_ratio"}
    if not required.issubset(df.columns):
        return None

    rules = {
        "full_gate": (
            (df["photo_enclosure_count"] > 0) &
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int),

        "remove_enclosure_count": (
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int),

        "remove_wall_cc": (
            (df["photo_enclosure_count"] > 0) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int),

        "remove_enclosed_ratio": (
            (df["photo_enclosure_count"] > 0) &
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_outside_free_ratio"] < outside_thr)
        ).astype(int),

        "remove_outside_ratio": (
            (df["photo_enclosure_count"] > 0) &
            (df["photo_wall_cc"] < wall_thr) &
            (df["photo_enclosed_free_ratio"] > enclosed_thr)
        ).astype(int),
    }

    rows = []
    for name, pred_valid in rules.items():
        pred_broken = 1 - pred_valid
        stats = confusion(y_true, pred_broken)

        silent_total = int(silent_mask.sum())
        silent_captured = int(((silent_mask == 1) & (pred_broken == 1)).sum())
        silent_recall = safe_div(silent_captured, silent_total)

        rows.append({
            "ablation": name,
            **stats,
            "silent_total": silent_total,
            "silent_captured": silent_captured,
            "silent_recall": silent_recall,
        })

    out_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out_df.to_csv(out_csv, index=False)
    return out_df


def grid_ablation(df, y_true, silent_mask, out_csv, wall_values, enclosed_values, outside_values):
    rows = []

    for w, e, o in itertools.product(wall_values, enclosed_values, outside_values):
        if not {"photo_wall_cc", "photo_enclosed_free_ratio", "photo_outside_free_ratio"}.issubset(df.columns):
            break

        pred_valid = (
            (df["photo_wall_cc"] < w) &
            (df["photo_enclosed_free_ratio"] > e) &
            (df["photo_outside_free_ratio"] < o)
        ).astype(int)

        pred_broken = 1 - pred_valid
        stats = confusion(y_true, pred_broken)

        silent_total = int(silent_mask.sum())
        silent_captured = int(((silent_mask == 1) & (pred_broken == 1)).sum())
        silent_recall = safe_div(silent_captured, silent_total)

        rows.append({
            "wall_thr": w,
            "enclosed_thr": e,
            "outside_thr": o,
            **stats,
            "silent_total": silent_total,
            "silent_captured": silent_captured,
            "silent_recall": silent_recall,
        })

    out_df = pd.DataFrame(rows).sort_values(["f1", "silent_recall", "recall"], ascending=False)
    out_df.to_csv(out_csv, index=False)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wall-thr", type=int, default=30)
    ap.add_argument("--enclosed-thr", type=float, default=0.25)
    ap.add_argument("--outside-thr", type=float, default=0.75)
    ap.add_argument("--silent-ratio-thr", type=float, default=0.3)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.paired)

    # Main thesis target: degraded = clean ok, photo broken
    if "degraded_clean_valid_photo_broken" not in df.columns:
        raise RuntimeError("paired_results.csv missing column: degraded_clean_valid_photo_broken")

    y_degraded = df["degraded_clean_valid_photo_broken"].astype(int)

    # Secondary target: any photo invalid
    if "photo_valid" not in df.columns:
        raise RuntimeError("paired_results.csv missing column: photo_valid")

    y_photo_invalid = (df["photo_valid"] == 0).astype(int)

    # Silent failures among photo-invalid
    if "photo_enclosed_free_ratio" not in df.columns:
        raise RuntimeError("paired_results.csv missing column: photo_enclosed_free_ratio")

    silent_mask = ((df["photo_valid"] == 0) & (df["photo_enclosed_free_ratio"] > args.silent_ratio_thr)).astype(int)

    # -------------------------
    # TABLE A: gate comparison for degraded detection
    # -------------------------
    evaluate_rules(
        df=df,
        y_true=y_degraded,
        silent_mask=silent_mask,
        out_csv=out_dir / "table_gate_comparison_degraded.csv",
        wall_thr=args.wall_thr,
        enclosed_thr=args.enclosed_thr,
        outside_thr=args.outside_thr,
    )

    # -------------------------
    # TABLE B: gate comparison for photo-invalid detection
    # -------------------------
    evaluate_rules(
        df=df,
        y_true=y_photo_invalid,
        silent_mask=silent_mask,
        out_csv=out_dir / "table_gate_comparison_photo_invalid.csv",
        wall_thr=args.wall_thr,
        enclosed_thr=args.enclosed_thr,
        outside_thr=args.outside_thr,
    )

    # -------------------------
    # TABLE C: wall_cc threshold ablation
    # -------------------------
    threshold_ablation(
        df=df,
        y_true=y_degraded,
        silent_mask=silent_mask,
        out_csv=out_dir / "table_wallcc_threshold_ablation.csv",
        out_plot=out_dir / "plot_wallcc_threshold_ablation.png",
        thr_values=list(range(10, 51, 5)),
    )

    # -------------------------
    # TABLE D: component ablation
    # -------------------------
    component_ablation(
        df=df,
        y_true=y_degraded,
        silent_mask=silent_mask,
        out_csv=out_dir / "table_component_ablation.csv",
        wall_thr=args.wall_thr,
        enclosed_thr=args.enclosed_thr,
        outside_thr=args.outside_thr,
    )

    # -------------------------
    # TABLE E: grid ablation
    # -------------------------
    grid_ablation(
        df=df,
        y_true=y_degraded,
        silent_mask=silent_mask,
        out_csv=out_dir / "table_grid_ablation.csv",
        wall_values=[20, 25, 30, 35, 40],
        enclosed_values=[0.20, 0.25, 0.30],
        outside_values=[0.70, 0.75, 0.80],
    )

    print("[OK] Wrote comparison + ablation outputs to:", out_dir)


if __name__ == "__main__":
    main()