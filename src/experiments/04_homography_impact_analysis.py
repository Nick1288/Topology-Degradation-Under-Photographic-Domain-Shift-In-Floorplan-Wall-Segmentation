import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, mannwhitneyu, ttest_ind
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =========================================================
# Helpers
# =========================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_id_col(df):
    candidates = ["plan_id", "sample_id", "id", "leaf_id", "folder", "name"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect ID column. Available columns: {list(df.columns)}")

def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return np.nan
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def angle_deg(a, b, c):
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def quad_orthogonality_error_deg(quad):
    # quad assumed [tl, tr, br, bl]
    pts = np.asarray(quad, dtype=np.float64)
    if pts.shape != (4, 2):
        return np.nan
    angs = [
        angle_deg(pts[3], pts[0], pts[1]),
        angle_deg(pts[0], pts[1], pts[2]),
        angle_deg(pts[1], pts[2], pts[3]),
        angle_deg(pts[2], pts[3], pts[0]),
    ]
    return float(np.nanmean([abs(a - 90.0) for a in angs]))

def homography_feature_block(H, prefix):
    H = np.asarray(H, dtype=np.float64)
    A = H[:2, :2]

    out = {}
    out[f"{prefix}_h00"] = H[0, 0]
    out[f"{prefix}_h01"] = H[0, 1]
    out[f"{prefix}_h02"] = H[0, 2]
    out[f"{prefix}_h10"] = H[1, 0]
    out[f"{prefix}_h11"] = H[1, 1]
    out[f"{prefix}_h12"] = H[1, 2]
    out[f"{prefix}_h20"] = H[2, 0]
    out[f"{prefix}_h21"] = H[2, 1]
    out[f"{prefix}_perspective_mag"] = float(np.sqrt(H[2, 0] ** 2 + H[2, 1] ** 2))
    out[f"{prefix}_translation_mag"] = float(np.sqrt(H[0, 2] ** 2 + H[1, 2] ** 2))

    try:
        s = np.linalg.svd(A, compute_uv=False)
        out[f"{prefix}_affine_sv1"] = float(s[0])
        out[f"{prefix}_affine_sv2"] = float(s[-1])
        out[f"{prefix}_affine_cond"] = float(s[0] / s[-1]) if s[-1] != 0 else np.inf
    except Exception:
        out[f"{prefix}_affine_sv1"] = np.nan
        out[f"{prefix}_affine_sv2"] = np.nan
        out[f"{prefix}_affine_cond"] = np.nan

    out[f"{prefix}_affine_det"] = float(np.linalg.det(A))
    out[f"{prefix}_scale_proxy"] = float(np.sqrt(abs(np.linalg.det(A)))) if np.isfinite(np.linalg.det(A)) else np.nan
    return out

def compute_sample_features(sample_dir: Path):
    sample_id = sample_dir.name

    h_root_path = sample_dir / "H_root.npy"
    h_floor_path = sample_dir / "H_floor.npy"
    crop_meta_path = sample_dir / "crop_meta.json"
    rect_meta_path = sample_dir / "rectification_meta.json"

    if not all(p.exists() for p in [h_root_path, h_floor_path, crop_meta_path, rect_meta_path]):
        return None

    H_root = np.load(str(h_root_path))
    H_floor = np.load(str(h_floor_path))
    crop_meta = load_json(crop_meta_path)
    rect_meta = load_json(rect_meta_path)

    root_id = crop_meta.get("root_id", sample_id)
    bbox = crop_meta.get("crop_box_rectified_xyxy", [None, None, None, None])
    x1, y1, x2, y2 = bbox
    crop_w = crop_meta.get("crop_w", None)
    crop_h = crop_meta.get("crop_h", None)

    rect_w = rect_meta.get("rect_w", None)
    rect_h = rect_meta.get("rect_h", None)

    row = {
        "plan_id": sample_id,
        "root_id": root_id,
        "rect_method": rect_meta.get("method", None),
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "rect_w": rect_w,
        "rect_h": rect_h,
    }

    if None not in [x1, y1, x2, y2, rect_w, rect_h] and rect_w and rect_h:
        full_area = rect_w * rect_h
        crop_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        row["crop_area_frac"] = crop_area / full_area if full_area > 0 else np.nan
        row["crop_center_x_norm"] = ((x1 + x2) / 2.0) / rect_w
        row["crop_center_y_norm"] = ((y1 + y2) / 2.0) / rect_h
        row["touch_top"] = int(y1 <= 0)
        row["touch_left"] = int(x1 <= 0)
        row["touch_bottom"] = int(y2 >= rect_h)
        row["touch_right"] = int(x2 >= rect_w)
        row["touch_any_edge"] = int(any([row["touch_top"], row["touch_left"], row["touch_bottom"], row["touch_right"]]))
    else:
        row["crop_area_frac"] = np.nan
        row["crop_center_x_norm"] = np.nan
        row["crop_center_y_norm"] = np.nan
        row["touch_top"] = np.nan
        row["touch_left"] = np.nan
        row["touch_bottom"] = np.nan
        row["touch_right"] = np.nan
        row["touch_any_edge"] = np.nan

    row.update(homography_feature_block(H_root, "root"))
    row.update(homography_feature_block(H_floor, "floor"))

    src_quad = rect_meta.get("src_quad", None)
    if src_quad is not None and rect_meta.get("orig_w") and rect_meta.get("orig_h"):
        src_area = polygon_area(src_quad)
        orig_area = rect_meta["orig_w"] * rect_meta["orig_h"]
        row["root_src_quad_area_frac"] = src_area / orig_area if orig_area > 0 else np.nan
        row["root_quad_orthogonality_error_deg"] = quad_orthogonality_error_deg(src_quad)
    else:
        row["root_src_quad_area_frac"] = np.nan
        row["root_quad_orthogonality_error_deg"] = np.nan

    return row

def collect_feature_table(data_root):
    rows = []
    for p in sorted(Path(data_root).iterdir()):
        if not p.is_dir():
            continue
        if p.name.endswith("__ROOT"):
            continue
        row = compute_sample_features(p)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No valid sample folders found with homography metadata.")
    return pd.DataFrame(rows)

def add_delta_columns(df):
    if "clean_wall_iou" in df.columns and "photo_wall_iou" in df.columns:
        df["delta_wall_iou"] = df["photo_wall_iou"] - df["clean_wall_iou"]
    if "clean_wall_f1" in df.columns and "photo_wall_f1" in df.columns:
        df["delta_wall_f1"] = df["photo_wall_f1"] - df["clean_wall_f1"]
    if "clean_wall_cc" in df.columns and "photo_wall_cc" in df.columns:
        df["delta_wall_cc"] = df["photo_wall_cc"] - df["clean_wall_cc"]
    return df

def safe_qcut(series, q=4):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.nunique() < q:
        return pd.Series([np.nan] * len(series), index=series.index)
    try:
        return pd.qcut(s, q=q, labels=[f"Q{i+1}" for i in range(q)], duplicates="drop")
    except Exception:
        return pd.Series([np.nan] * len(series), index=series.index)

def summarize_by_bin(df, bin_col, metric_cols):
    out = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        tmp = df.groupby(bin_col, dropna=False)[metric].agg(["count", "mean", "median", "std"]).reset_index()
        tmp.insert(0, "metric", metric)
        out.append(tmp)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)

def correlation_block(df, feature_cols, outcome_cols):
    rows = []
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce")
        for out in outcome_cols:
            if out not in df.columns:
                continue
            y = pd.to_numeric(df[out], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 5:
                continue
            if SCIPY_OK:
                rho, p = spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            rows.append({
                "feature": feat,
                "outcome": out,
                "n": int(mask.sum()),
                "spearman_rho": rho,
                "p_value": p,
            })
    return pd.DataFrame(rows)

def write_text_report(path, df, bin_summaries, corr_df, severity_col):
    lines = []
    lines.append("TEST 1: HOMOGRAPHY IMPACT REPORT")
    lines.append("=" * 70)
    lines.append(f"n_samples = {len(df)}")
    lines.append("")

    if severity_col in df.columns:
        lines.append(f"Severity column used: {severity_col}")
        lines.append(df[severity_col].describe().to_string())
        lines.append("")

    for title, bdf in bin_summaries.items():
        lines.append(title)
        lines.append("-" * len(title))
        if bdf.empty:
            lines.append("No summary available.")
        else:
            lines.append(bdf.to_string(index=False))
        lines.append("")

    if not corr_df.empty:
        lines.append("Spearman correlations")
        lines.append("---------------------")
        lines.append(corr_df.sort_values(["outcome", "spearman_rho"], ascending=[True, True]).to_string(index=False))
        lines.append("")

    # High vs low severity comparison
    if severity_col in df.columns:
        sev = pd.to_numeric(df[severity_col], errors="coerce")
        q1 = sev.quantile(0.25)
        q4 = sev.quantile(0.75)
        low = df[sev <= q1]
        high = df[sev >= q4]

        lines.append("Low vs high severity comparison")
        lines.append("-------------------------------")
        lines.append(f"low_n  = {len(low)}")
        lines.append(f"high_n = {len(high)}")
        for metric in ["photo_wall_iou", "photo_wall_f1", "photo_wall_cc", "delta_wall_iou", "delta_wall_f1", "delta_wall_cc"]:
            if metric not in df.columns:
                continue
            a = pd.to_numeric(low[metric], errors="coerce").dropna()
            b = pd.to_numeric(high[metric], errors="coerce").dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            lines.append(f"{metric}: low_mean={a.mean():.6f}, high_mean={b.mean():.6f}")
            if SCIPY_OK:
                try:
                    t_stat, t_p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
                    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
                    lines.append(f"  Welch t-test: t={t_stat:.6f}, p={t_p:.6g}")
                    lines.append(f"  Mann-Whitney: U={u_stat:.6f}, p={u_p:.6g}")
                except Exception as e:
                    lines.append(f"  Test error: {e}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to cubi_ft_with_h")
    parser.add_argument("--eval-csv", required=True, help="Path to paired eval CSV")
    parser.add_argument("--out-csv", required=True, help="Path to write enriched CSV")
    parser.add_argument("--out-report", required=True, help="Path to write text report")
    parser.add_argument("--out-bin-summary", required=True, help="Path to write quartile summary CSV")
    args = parser.parse_args()

    feat_df = collect_feature_table(args.data_root)

    eval_df = pd.read_csv(args.eval_csv)
    eval_id_col = detect_id_col(eval_df)
    if eval_id_col != "plan_id":
        eval_df = eval_df.rename(columns={eval_id_col: "plan_id"})

    merged = eval_df.merge(feat_df, on="plan_id", how="left")
    merged = add_delta_columns(merged)

    # Main severity score: keep it simple and interpretable
    # Stronger perspective + worse conditioning + bigger orthogonality error
    merged["floor_affine_cond_log"] = np.log10(pd.to_numeric(merged["floor_affine_cond"], errors="coerce").replace([np.inf, -np.inf], np.nan))
    merged["floor_perspective_mag_log"] = np.log10(pd.to_numeric(merged["floor_perspective_mag"], errors="coerce") + 1e-12)
    merged["root_quad_ortho_err"] = pd.to_numeric(merged["root_quad_orthogonality_error_deg"], errors="coerce")

    severity_parts = []
    for col in ["floor_affine_cond_log", "floor_perspective_mag_log", "root_quad_ortho_err"]:
        s = pd.to_numeric(merged[col], errors="coerce")
        z = (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) and np.isfinite(s.std(ddof=0)) else s * np.nan
        severity_parts.append(z.rename(col + "_z"))
        merged[col + "_z"] = z

    merged["distortion_severity_score"] = pd.concat(severity_parts, axis=1).mean(axis=1, skipna=True)
    merged["distortion_severity_bin"] = safe_qcut(merged["distortion_severity_score"], q=4)

    metric_cols = [
        "clean_wall_iou", "photo_wall_iou", "delta_wall_iou",
        "clean_wall_f1", "photo_wall_f1", "delta_wall_f1",
        "clean_wall_cc", "photo_wall_cc", "delta_wall_cc",
    ]

    bin_summary = summarize_by_bin(merged, "distortion_severity_bin", metric_cols)

    corr_features = [
        "distortion_severity_score",
        "floor_affine_cond",
        "floor_perspective_mag",
        "root_quad_orthogonality_error_deg",
        "crop_area_frac",
        "crop_center_y_norm",
    ]
    corr_outcomes = ["photo_wall_iou", "photo_wall_f1", "photo_wall_cc", "delta_wall_iou", "delta_wall_f1", "delta_wall_cc"]
    corr_df = correlation_block(merged, corr_features, corr_outcomes)

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    bin_summary.to_csv(args.out_bin_summary, index=False)
    write_text_report(args.out_report, merged, {"Quartile summaries": bin_summary}, corr_df, "distortion_severity_score")

    print(f"Saved enriched CSV: {args.out_csv}")
    print(f"Saved report: {args.out_report}")
    print(f"Saved bin summary: {args.out_bin_summary}")

if __name__ == "__main__":
    main()