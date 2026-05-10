# make_appendix_b_stats.py
import argparse
from pathlib import Path
import pandas as pd

OUTCOME_ORDER = ["Stable", "Degraded", "Baseline failure", "Photo recovery"]

def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if str(series.dtype).startswith(("int", "float")):
        return series.fillna(0).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map({
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0
    }).fillna(0).astype(int)

def infer_outcome(row: pd.Series) -> str:
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

def latex_escape(text: str) -> str:
    s = str(text)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def format_num(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{x:.{decimals}f}"

def make_latex_table(df: pd.DataFrame, caption: str, label: str, out_path: Path):
    cols = list(df.columns)
    align = "l" * 2 + "r" * (len(cols) - 2)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        vals = [latex_escape(v) for v in row.tolist()]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def summarise_metric(df: pd.DataFrame, metric: str, decimals=2) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_ORDER:
        sub = df[df["outcome"] == outcome]
        if len(sub) == 0:
            continue
        s = sub[metric].dropna()
        rows.append({
            "Outcome group": outcome,
            "Metric": metric,
            "n": str(len(s)),
            "Mean": format_num(s.mean(), decimals),
            "Median": format_num(s.median(), decimals),
            "SD": format_num(s.std(ddof=1), decimals) if len(s) > 1 else "",
            "Min": format_num(s.min(), decimals),
            "Max": format_num(s.max(), decimals),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Full paired evaluation CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = ["plan_id", "clean_valid", "photo_valid", "clean_wall_cc", "photo_wall_cc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["clean_valid"] = boolify(df["clean_valid"])
    df["photo_valid"] = boolify(df["photo_valid"])
    df["outcome"] = df.apply(infer_outcome, axis=1)
    df["delta_wall_cc"] = df["photo_wall_cc"] - df["clean_wall_cc"]

    # Table B.1: wall_cc only
    wall_metrics = ["clean_wall_cc", "photo_wall_cc", "delta_wall_cc"]
    wall_frames = [summarise_metric(df, m, decimals=2) for m in wall_metrics]
    table_b1 = pd.concat(wall_frames, ignore_index=True)

    b1_csv = out_dir / "appendix_b1_wall_cc_stats.csv"
    table_b1.to_csv(b1_csv, index=False)

    make_latex_table(
        table_b1,
        caption="Descriptive statistics for wall connected component counts by paired outcome group.",
        label="tab:appendix_b1_wallcc",
        out_path=out_dir / "appendix_b1_wall_cc_stats.tex"
    )

    # Table B.2: selected additional topology metrics
    optional_metrics = [
        "clean_enclosure_count",
        "photo_enclosure_count",
        "clean_enclosed_free_ratio",
        "photo_enclosed_free_ratio",
        "clean_outside_free_ratio",
        "photo_outside_free_ratio",
        "clean_wall_area_ratio",
        "photo_wall_area_ratio",
    ]
    available_optional = [m for m in optional_metrics if m in df.columns]

    if available_optional:
        extra_frames = []
        for m in available_optional:
            decimals = 3 if "ratio" in m else 2
            extra_frames.append(summarise_metric(df, m, decimals=decimals))
        table_b2 = pd.concat(extra_frames, ignore_index=True)

        b2_csv = out_dir / "appendix_b2_selected_topology_stats.csv"
        table_b2.to_csv(b2_csv, index=False)

        make_latex_table(
            table_b2,
            caption="Descriptive statistics for selected topology metrics by paired outcome group.",
            label="tab:appendix_b2_topology",
            out_path=out_dir / "appendix_b2_selected_topology_stats.tex"
        )

        print(f"Wrote: {b2_csv}")
        print(f"Wrote: {out_dir / 'appendix_b2_selected_topology_stats.tex'}")
    else:
        print("No optional topology metric columns found for Table B.2.")

    print(f"Wrote: {b1_csv}")
    print(f"Wrote: {out_dir / 'appendix_b1_wall_cc_stats.tex'}")

if __name__ == "__main__":
    main()