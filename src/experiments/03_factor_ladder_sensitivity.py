import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from common.image import read_bgr_white_background
from common.model import SegmentationHead, bgr_to_tensor, load_checkpoint
from common.topology import compute_topology_metrics, is_structurally_valid


BORDER_MAP = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
}

INTERP_MAP = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


def apply_spatial_photometric(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = cv2.GaussianBlur(img, (5, 5), 1.5)

    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = xx / max(w - 1, 1)
    yy = yy / max(h - 1, 1)

    ax = rng.uniform(-0.15, 0.15)
    ay = rng.uniform(-0.15, 0.15)
    bias = rng.uniform(0.82, 0.95)

    illum = bias + ax * (xx - 0.5) + ay * (yy - 0.5)

    cx = rng.uniform(0.2, 0.8)
    cy = rng.uniform(0.2, 0.8)
    sx = rng.uniform(0.12, 0.24)
    sy = rng.uniform(0.12, 0.24)
    shadow_strength = rng.uniform(0.05, 0.18)

    shadow = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    illum *= (1.0 - shadow_strength * shadow)

    out = np.clip(out.astype(np.float32) * illum[..., None], 0, 255).astype(np.uint8)

    _, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def crop_fixed(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx = int(round(w * frac))
    cy = int(round(h * frac))
    if cx * 2 >= w or cy * 2 >= h:
        return img
    out = img[cy:h - cy, cx:w - cx]
    if out.size == 0:
        return img
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_mask_valid_region(img: np.ndarray, valid_mask: np.ndarray, trim_frac: float = 0.02) -> np.ndarray:
    h, w = img.shape[:2]
    ys, xs = np.where(valid_mask > 0)
    if len(xs) < 100 or len(ys) < 100:
        return img

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad_x = int(round((x1 - x0 + 1) * trim_frac))
    pad_y = int(round((y1 - y0 + 1) * trim_frac))

    x0 = max(0, x0 + pad_x)
    x1 = min(w - 1, x1 - pad_x)
    y0 = max(0, y0 + pad_y)
    y1 = min(h - 1, y1 - pad_y)

    if x1 <= x0 or y1 <= y0:
        return img

    out = img[y0:y1 + 1, x0:x1 + 1]
    if out.size == 0:
        return img
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_warp_with_policy(
    img: np.ndarray,
    rng: np.random.RandomState,
    border_mode_name: str,
    crop_policy_name: str,
    interp_name: str,
) -> np.ndarray:
    h, w = img.shape[:2]

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    dst = np.float32([
        [rng.uniform(0.06, 0.16) * w, rng.uniform(0.04, 0.12) * h],
        [w - rng.uniform(0.10, 0.20) * w, rng.uniform(0.02, 0.10) * h],
        [w - rng.uniform(0.04, 0.14) * w, h - rng.uniform(0.08, 0.18) * h],
        [rng.uniform(0.08, 0.18) * w, h - rng.uniform(0.03, 0.12) * h],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    interp = INTERP_MAP[interp_name]
    border_mode = BORDER_MAP[border_mode_name]

    warped = cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=interp,
        borderMode=border_mode,
        borderValue=(255, 255, 255),
    )

    # For mask-based crop, always compute valid region using constant-zero mask warp
    mask = np.full((h, w), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if crop_policy_name == "none":
        return warped
    elif crop_policy_name == "fixed1":
        return crop_fixed(warped, 0.01)
    elif crop_policy_name == "fixed2":
        return crop_fixed(warped, 0.02)
    elif crop_policy_name == "mask":
        return crop_mask_valid_region(warped, warped_mask, trim_frac=0.02)
    else:
        raise ValueError(f"Unknown crop policy: {crop_policy_name}")


def apply_structural_capture_loss(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = cv2.GaussianBlur(img, (7, 7), 2.0)

    h, w = out.shape[:2]
    scale = rng.uniform(0.55, 0.75)
    sw = max(32, int(round(w * scale)))
    sh = max(32, int(round(h * scale)))

    out = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    q = int(rng.uniform(35, 50))
    _, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


def apply_degradation(
    img_bgr: np.ndarray,
    level: int,
    rng: np.random.RandomState,
    border_mode_name: str,
    crop_policy_name: str,
    interp_name: str,
) -> np.ndarray:
    out = img_bgr.copy()

    if level >= 1:
        out = apply_spatial_photometric(out, rng)

    if level >= 2:
        out = apply_warp_with_policy(out, rng, border_mode_name, crop_policy_name, interp_name)

    if level >= 3:
        out = apply_structural_capture_loss(out, rng)

    return out


def summarise_variant(dfv: pd.DataFrame, thr_wall_cc: int) -> dict:
    row = {
        "variant": dfv["variant"].iloc[0],
        "border_mode": dfv["border_mode"].iloc[0],
        "crop_policy": dfv["crop_policy"].iloc[0],
        "interp": dfv["interp"].iloc[0],
    }

    for lvl in [0, 1, 2, 3]:
        d = dfv[dfv["level"] == lvl]
        row[f"L{lvl}_valid_rate"] = float(d["valid"].mean())
        row[f"L{lvl}_wall_cc_mean"] = float(d["wall_cc"].mean())
        row[f"L{lvl}_enclosed_mean"] = float(d["enclosed_free_ratio"].mean())
        row[f"L{lvl}_outside_mean"] = float(d["outside_free_ratio"].mean())

    piv = dfv.pivot(index="plan_id", columns="level", values="wall_cc")
    piv.columns = [f"L{c}" for c in piv.columns]

    for col in ["L0", "L1", "L2", "L3"]:
        if col not in piv.columns:
            row[f"{col}_missing"] = 1

    piv["delta_01"] = piv["L1"] - piv["L0"]
    piv["delta_12"] = piv["L2"] - piv["L1"]
    piv["delta_23"] = piv["L3"] - piv["L2"]

    row["median_delta_01"] = float(piv["delta_01"].median())
    row["median_delta_12"] = float(piv["delta_12"].median())
    row["median_delta_23"] = float(piv["delta_23"].median())

    row["pct_increase_01"] = float((piv["delta_01"] > 0).mean())
    row["pct_increase_12"] = float((piv["delta_12"] > 0).mean())
    row["pct_increase_23"] = float((piv["delta_23"] > 0).mean())

    l0_valid = piv["L0"] < thr_wall_cc
    row["cross_after_L1"] = int((l0_valid & (piv["L1"] >= thr_wall_cc)).sum())
    row["cross_after_L2"] = int((l0_valid & (piv["L2"] >= thr_wall_cc)).sum())
    row["cross_after_L3"] = int((l0_valid & (piv["L3"] >= thr_wall_cc)).sum())
    row["baseline_valid_count"] = int(l0_valid.sum())

    return row


def save_disagreement_lists(raw_df: pd.DataFrame, out_dir: Path):
    # Compare L2 validity between border policies under the same crop/interp
    focus = raw_df[raw_df["level"] == 2].copy()

    base_cols = ["plan_id", "crop_policy", "interp"]
    piv = focus.pivot_table(
        index=base_cols,
        columns="border_mode",
        values="valid",
        aggfunc="first",
    ).reset_index()

    border_cols = [c for c in ["constant", "replicate", "reflect"] if c in piv.columns]
    if len(border_cols) < 2:
        return

    disagreement_rows = []
    for _, r in piv.iterrows():
        vals = [r[c] for c in border_cols if pd.notna(r[c])]
        if len(set(vals)) > 1:
            disagreement_rows.append(r)

    if disagreement_rows:
        out = pd.DataFrame(disagreement_rows)
        out.to_csv(out_dir / "l2_border_disagreements.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-ids", required=True, help="Path to degraded_ids.txt")
    ap.add_argument("--clean-raster-root", required=True, help="Root containing {plan_id}/clean_rgb.png")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pth)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--img-size", type=int, default=896)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roi-pad", type=int, default=30)

    ap.add_argument("--use-doors", action="store_true")
    ap.add_argument("--door-dilate", type=int, default=0)

    ap.add_argument("--thr-wall-cc", type=int, default=30)
    ap.add_argument("--thr-enclosed", type=float, default=0.25)
    ap.add_argument("--thr-outside", type=float, default=0.75)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--border-modes", nargs="+", default=["constant", "replicate", "reflect"])
    ap.add_argument("--crop-policies", nargs="+", default=["none", "fixed1", "mask"])
    ap.add_argument("--interps", nargs="+", default=["linear"])
    args = ap.parse_args()

    for b in args.border_modes:
        if b not in BORDER_MAP:
            raise ValueError(f"Unsupported border mode: {b}")
    for c in args.crop_policies:
        if c not in {"none", "fixed1", "fixed2", "mask"}:
            raise ValueError(f"Unsupported crop policy: {c}")
    for i in args.interps:
        if i not in INTERP_MAP:
            raise ValueError(f"Unsupported interpolation: {i}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degraded_ids = [x.strip() for x in Path(args.degraded_ids).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit and args.limit > 0:
        degraded_ids = degraded_ids[:args.limit]
    if len(degraded_ids) == 0:
        raise RuntimeError("No degraded ids loaded.")

    clean_root = Path(args.clean_raster_root)
    if not clean_root.exists():
        raise FileNotFoundError(f"Missing clean raster root: {clean_root}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = SegmentationHead(num_classes=3).to(device).eval()
    load_checkpoint(model, args.ckpt, device)

    raw_rows = []
    summary_rows = []

    with torch.no_grad():
        for border_mode in args.border_modes:
            for crop_policy in args.crop_policies:
                for interp_name in args.interps:
                    variant = f"b-{border_mode}__c-{crop_policy}__i-{interp_name}"
                    print(f"[RUN] {variant}")

                    for plan_idx, plan_id in enumerate(degraded_ids):
                        cp = clean_root / plan_id / "clean_rgb.png"
                        if not cp.exists():
                            continue

                        img = read_bgr_white_background(cp)
                        if img is None:
                            continue

                        img_rs = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

                        for level in [0, 1, 2, 3]:
                            # Deterministic per plan, per level, per variant
                            variant_seed = (
                                args.seed
                                + 100000 * (args.border_modes.index(border_mode) + 1)
                                + 10000 * (args.crop_policies.index(crop_policy) + 1)
                                + 1000 * (args.interps.index(interp_name) + 1)
                                + 10 * plan_idx
                                + level
                            )
                            rng = np.random.RandomState(variant_seed)

                            inp = apply_degradation(
                                img_rs, level, rng,
                                border_mode_name=border_mode,
                                crop_policy_name=crop_policy,
                                interp_name=interp_name,
                            )

                            x = bgr_to_tensor(inp).to(device)
                            logits = model(x)
                            pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

                            wall = (pred == 1).astype(np.uint8)
                            door = (pred == 2).astype(np.uint8)

                            metrics = compute_topology_metrics(
                                wall, door,
                                roi_pad=args.roi_pad,
                                use_doors=args.use_doors,
                                door_dilate=args.door_dilate,
                            )
                            ok = is_structurally_valid(metrics, args.thr_wall_cc, args.thr_enclosed, args.thr_outside)

                            raw_rows.append({
                                "variant": variant,
                                "border_mode": border_mode,
                                "crop_policy": crop_policy,
                                "interp": interp_name,
                                "plan_id": plan_id,
                                "level": level,
                                "valid": int(ok),
                                "wall_cc": metrics["wall_cc"],
                                "enclosure_count": metrics["enclosure_count"],
                                "enclosed_free_ratio": metrics["enclosed_free_ratio"],
                                "outside_free_ratio": metrics["outside_free_ratio"],
                                "wall_area_ratio": metrics["wall_area_ratio"],
                            })

                    dfv = pd.DataFrame([r for r in raw_rows if r["variant"] == variant])
                    if len(dfv) > 0:
                        summary_rows.append(summarise_variant(dfv, args.thr_wall_cc))

    if len(raw_rows) == 0:
        raise RuntimeError("No rows produced.")

    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)

    raw_df.to_csv(out_dir / "sensitivity_raw_results.csv", index=False)
    summary_df.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    save_disagreement_lists(raw_df, out_dir)

    # Compact text summary
    lines = []
    lines.append("Sensitivity study summary")
    lines.append("")

    for _, r in summary_df.sort_values(["border_mode", "crop_policy", "interp"]).iterrows():
        lines.append(f"Variant: {r['variant']}")
        lines.append(
            f"  L2 valid={r['L2_valid_rate']:.3f}, wall_cc={r['L2_wall_cc_mean']:.2f}, "
            f"enclosed={r['L2_enclosed_mean']:.3f}, outside={r['L2_outside_mean']:.3f}, "
            f"cross={int(r['cross_after_L2'])}/{int(r['baseline_valid_count'])}"
        )
        lines.append(
            f"  L3 valid={r['L3_valid_rate']:.3f}, wall_cc={r['L3_wall_cc_mean']:.2f}, "
            f"enclosed={r['L3_enclosed_mean']:.3f}, outside={r['L3_outside_mean']:.3f}, "
            f"cross={int(r['cross_after_L3'])}/{int(r['baseline_valid_count'])}"
        )
        lines.append(
            f"  median deltas: 01={r['median_delta_01']:.2f}, 12={r['median_delta_12']:.2f}, 23={r['median_delta_23']:.2f}"
        )
        lines.append("")

    (out_dir / "sensitivity_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Wrote:")
    print(" -", out_dir / "sensitivity_raw_results.csv")
    print(" -", out_dir / "sensitivity_summary.csv")
    print(" -", out_dir / "sensitivity_summary.txt")
    if (out_dir / "l2_border_disagreements.csv").exists():
        print(" -", out_dir / "l2_border_disagreements.csv")


if __name__ == "__main__":
    main()
