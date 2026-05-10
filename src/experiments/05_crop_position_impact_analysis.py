import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, kruskal
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def add_position_groups(df):
    y = pd.to_numeric(df["crop_center_y_norm"], errors="coerce")
    x = pd.to_numeric(df["crop_center_x_norm"], errors="coerce")
    area = pd.to_numeric(df["crop_area_frac"], errors="coerce")

    df["vertical_band"] = pd.cut(
        y,
        bins=[-np.inf, 1/3, 2/3, np.inf],
        labels=["top", "middle", "bottom"]
    )

    df["horizontal_band"] = pd.cut(
        x,
        bins=[-np.inf, 1/3, 2/3, np.inf],
        labels=["left", "middle", "right"]
    )

    df["crop_size_band"] = pd.cut(
        area,
        bins=[-np.inf, 0.20, 0.40, 0.60, np.inf],
        labels=["<=20%", "20-40%", "40-60%", ">60%"]
    )

    # edge pattern
    def edge_pattern(row):
        flags = []
        for col, name in [("touch_top", "T"), ("touch_bottom", "B"), ("touch_left", "L"), ("touch_right", "R")]:
            v = row.get(col, np.nan)
            if pd.notna(v) and int(v) == 1:
                flags.append(name)
        return "".join(flags) if flags else "none"

    df["edge_pattern"] = df.apply(edge_pattern, axis=1)
    return df

def group_summary(df, group_col, metrics):
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        tmp = df.groupby(group_col, dropna=False)[m].agg(["count", "mean", "median", "std"]).reset_index()
        tmp.insert(0, "metric", m)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def write_report(path, df, summaries):
    lines = []
    lines.append("TEST 2: CROP POSITION / FLOOR POSITION IMPACT REPORT")
    lines.append("=" * 75)
    lines.append(f"n_samples = {len(df)}")
    lines.append("")

    for title, sdf in summaries.items():
        lines.append(title)
        lines.append("-" * len(title))
        if sdf.empty:
            lines.append("No summary available.")
        else:
            lines.append(sdf.to_string(index=False))
        lines.append("")

    metrics = [m for m in ["photo_wall_iou", "photo_wall_f1", "photo_wall_cc", "delta_wall_iou", "delta_wall_f1", "delta_wall_cc"] if m in df.columns]

    if SCIPY_OK:
        for group_col in ["vertical_band", "horizontal_band", "crop_size_band", "touch_any_edge"]:
            if group_col not in df.columns:
                continue
            lines.append(f"Statistical tests for {group_col}")
            lines.append("-" * (22 + len(group_col)))
            for metric in metrics:
                grouped = []
                labels = []
                for g, sub in df.groupby(group_col, dropna=False):
                    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                    if len(vals) >= 3:
                        grouped.append(vals.values)
                        labels.append(g)
                if len(grouped) >= 2:
                    try:
                        stat, p = kruskal(*grouped)
                        lines.append(f"{metric}: Kruskal-Wallis H={stat:.6f}, p={p:.6g}, groups={labels}")
                    except Exception as e:
                        lines.append(f"{metric}: test error: {e}")
            lines.append("")

        # focused top vs bottom comparison
        if "vertical_band" in df.columns:
            top = df[df["vertical_band"] == "top"]
            bottom = df[df["vertical_band"] == "bottom"]
            lines.append("Top vs bottom comparison")
            lines.append("------------------------")
            for metric in metrics:
                a = pd.to_numeric(top[metric], errors="coerce").dropna()
                b = pd.to_numeric(bottom[metric], errors="coerce").dropna()
                if len(a) >= 3 and len(b) >= 3:
                    try:
                        u, p = mannwhitneyu(a, b, alternative="two-sided")
                        lines.append(f"{metric}: top_mean={a.mean():.6f}, bottom_mean={b.mean():.6f}, U={u:.6f}, p={p:.6g}")
                    except Exception as e:
                        lines.append(f"{metric}: test error: {e}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-csv", required=True, help="Output CSV from test1_homography_impact.py")
    parser.add_argument("--out-report", required=True, help="Path to text report")
    parser.add_argument("--out-vertical-summary", required=True, help="Path to vertical-band summary CSV")
    parser.add_argument("--out-horizontal-summary", required=True, help="Path to horizontal-band summary CSV")
    parser.add_argument("--out-size-summary", required=True, help="Path to crop-size summary CSV")
    parser.add_argument("--out-edge-summary", required=True, help="Path to edge summary CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.enriched_csv)
    df = add_position_groups(df)

    metrics = [
        "clean_wall_iou", "photo_wall_iou", "delta_wall_iou",
        "clean_wall_f1", "photo_wall_f1", "delta_wall_f1",
        "clean_wall_cc", "photo_wall_cc", "delta_wall_cc",
    ]

    vertical_summary = group_summary(df, "vertical_band", metrics)
    horizontal_summary = group_summary(df, "horizontal_band", metrics)
    size_summary = group_summary(df, "crop_size_band", metrics)
    edge_summary = group_summary(df, "edge_pattern", metrics)

    os.makedirs(Path(args.out_report).parent, exist_ok=True)
    vertical_summary.to_csv(args.out_vertical_summary, index=False)
    horizontal_summary.to_csv(args.out_horizontal_summary, index=False)
    size_summary.to_csv(args.out_size_summary, index=False)
    edge_summary.to_csv(args.out_edge_summary, index=False)
    write_report(args.out_report, df, {
        "Vertical band summaries": vertical_summary,
        "Horizontal band summaries": horizontal_summary,
        "Crop size summaries": size_summary,
        "Edge pattern summaries": edge_summary,
    })

    print(f"Saved report: {args.out_report}")
    print(f"Saved vertical summary: {args.out_vertical_summary}")
    print(f"Saved horizontal summary: {args.out_horizontal_summary}")
    print(f"Saved size summary: {args.out_size_summary}")
    print(f"Saved edge summary: {args.out_edge_summary}")

if __name__ == "__main__":
    main()