# make_appendix_paired_roster.py
import argparse
from pathlib import Path
import pandas as pd

EXPECTED_COLS = [
    "plan_id",
    "clean_valid",
    "photo_valid",
    "clean_wall_cc",
    "photo_wall_cc",
]

OPTIONAL_COLS = [
    "degraded_clean_valid_photo_broken",
    "clean_enclosure_count",
    "photo_enclosure_count",
    "clean_enclosed_free_ratio",
    "photo_enclosed_free_ratio",
    "clean_outside_free_ratio",
    "photo_outside_free_ratio",
    "clean_wall_area_ratio",
    "photo_wall_area_ratio",
]

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

def latex_escape(text):
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

def make_longtable(df, out_tex, caption, label):
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)

    header = " & ".join(latex_escape(c) for c in cols) + r" \\"
    body_lines = []
    for _, row in df.iterrows():
        vals = [latex_escape(v) for v in row.tolist()]
        body_lines.append(" & ".join(vals) + r" \\")
    body = "\n".join(body_lines)

    tex = rf"""
\begin{{center}}
\small
\begin{{longtable}}{{{align}}}
\caption{{{latex_escape(caption)}}}\label{{{label}}} \\
\hline
{header}
\hline
\endfirsthead
\hline
{header}
\hline
\endhead
\hline
\endfoot
{body}
\end{{longtable}}
\end{{center}}
""".strip()

    out_tex.write_text(tex, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Full paired evaluation CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--sort_by", default="plan_id", help="Column to sort by")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["clean_valid"] = boolify(df["clean_valid"])
    df["photo_valid"] = boolify(df["photo_valid"])

    df["outcome"] = df.apply(infer_outcome, axis=1)
    df["delta_wall_cc"] = df["photo_wall_cc"] - df["clean_wall_cc"]

    keep_cols = [
        "plan_id", "outcome", "clean_valid", "photo_valid",
        "clean_wall_cc", "photo_wall_cc", "delta_wall_cc"
    ]
    for c in OPTIONAL_COLS:
        if c in df.columns:
            keep_cols.append(c)

    roster = df[keep_cols].copy()

    if args.sort_by in roster.columns:
        roster = roster.sort_values(args.sort_by)

    roster_csv = out_dir / "appendix_paired_roster.csv"
    roster.to_csv(roster_csv, index=False)

    summary = (
        roster.groupby("outcome", dropna=False)
        .agg(
            n=("plan_id", "count"),
            mean_clean_wall_cc=("clean_wall_cc", "mean"),
            mean_photo_wall_cc=("photo_wall_cc", "mean"),
            mean_delta_wall_cc=("delta_wall_cc", "mean"),
        )
        .reset_index()
    )
    summary["pct"] = 100 * summary["n"] / len(roster)
    summary_csv = out_dir / "appendix_outcome_summary.csv"
    summary.to_csv(summary_csv, index=False)

    latex_df = roster[["plan_id", "outcome", "clean_valid", "photo_valid", "clean_wall_cc", "photo_wall_cc", "delta_wall_cc"]].copy()
    make_longtable(
        latex_df,
        out_dir / "appendix_paired_roster.tex",
        caption="Condensed paired evaluation results for all floorplans.",
        label="tab:appendix_paired_roster"
    )

    print(f"Wrote: {roster_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {out_dir / 'appendix_paired_roster.tex'}")

if __name__ == "__main__":
    main()