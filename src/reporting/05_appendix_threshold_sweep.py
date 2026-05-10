# make_appendix_threshold_sweep.py
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

def safe_div(a, b):
    return a / b if b else 0.0

def latex_escape(text):
    s = str(text)
    for k, v in {"_": r"\_", "%": r"\%", "&": r"\&"}.items():
        s = s.replace(k, v)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_thr", type=int, default=5)
    ap.add_argument("--max_thr", type=int, default=80)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["clean_valid"] = boolify(df["clean_valid"])
    df["photo_valid"] = boolify(df["photo_valid"])

    if "photo_wall_cc" not in df.columns:
        raise ValueError("CSV must contain photo_wall_cc")

    # Ground-truth target for Experiment 5 style screening:
    # degraded = clean valid, photo invalid
    df["is_degraded"] = ((df["clean_valid"] == 1) & (df["photo_valid"] == 0)).astype(int)

    rows = []
    for thr in range(args.min_thr, args.max_thr + 1):
        # gate pass if wall_cc < thr; flagged if wall_cc >= thr
        pred_flag = (df["photo_wall_cc"] >= thr).astype(int)

        tp = int(((pred_flag == 1) & (df["is_degraded"] == 1)).sum())
        fp = int(((pred_flag == 1) & (df["is_degraded"] == 0)).sum())
        fn = int(((pred_flag == 0) & (df["is_degraded"] == 1)).sum())
        tn = int(((pred_flag == 0) & (df["is_degraded"] == 0)).sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        pass_rate = safe_div(int((pred_flag == 0).sum()), len(df))

        rows.append({
            "threshold": thr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pass_rate": pass_rate,
        })

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "threshold_sweep_wall_cc.csv", index=False)

    best = res.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    best.to_frame().T.to_csv(out_dir / "best_threshold.csv", index=False)

    # concise latex table: top 10 by F1
    top = res.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).head(10).copy()
    top["precision"] = top["precision"].map(lambda x: f"{x:.3f}")
    top["recall"] = top["recall"].map(lambda x: f"{x:.3f}")
    top["f1"] = top["f1"].map(lambda x: f"{x:.3f}")
    top["pass_rate"] = top["pass_rate"].map(lambda x: f"{x:.3f}")

    cols = ["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "pass_rate"]
    lines = []
    lines.append(r"\begin{tabular}{rrrrrrrrr}")
    lines.append(r"\hline")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\hline")
    for _, row in top[cols].iterrows():
        lines.append(" & ".join(latex_escape(v) for v in row.tolist()) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (out_dir / "threshold_sweep_top10.tex").write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.plot(res["threshold"], res["f1"])
    plt.xlabel("wall_cc threshold")
    plt.ylabel("F1")
    plt.title("wall_cc threshold sweep: F1")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep_f1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(res["threshold"], res["recall"])
    plt.xlabel("wall_cc threshold")
    plt.ylabel("Recall")
    plt.title("wall_cc threshold sweep: recall")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep_recall.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(res["threshold"], res["pass_rate"])
    plt.xlabel("wall_cc threshold")
    plt.ylabel("Pass rate")
    plt.title("wall_cc threshold sweep: pass rate")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep_pass_rate.png", dpi=200)
    plt.close()

    print(f"Best threshold by F1: {int(best['threshold'])}")
    print(f"Wrote outputs to {out_dir}")

if __name__ == "__main__":
    main()