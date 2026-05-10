import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.image import bgr_to_rgb, crop_box, read_bgr, union_bbox
from common.model import SegmentationHead, load_checkpoint, run_segmentation
from common.preprocessing import preprocess_contrast


# =========================
# Difference visualization
# =========================

def make_diff_overlay(mask_before: np.ndarray, mask_after: np.ndarray):
    """
    Red = pixels added by CLAHE
    Blue = pixels removed by CLAHE
    White = unchanged wall
    Black = background
    """
    added = (mask_after == 1) & (mask_before == 0)
    removed = (mask_after == 0) & (mask_before == 1)
    common = (mask_after == 1) & (mask_before == 1)

    h, w = mask_before.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[common] = (255, 255, 255)
    vis[added] = (220, 40, 40)      # red
    vis[removed] = (40, 80, 220)    # blue

    return vis


def make_diff_overlay_on_photo(photo_bgr: np.ndarray, mask_before: np.ndarray, mask_after: np.ndarray, alpha=0.55):
    """
    Overlay added and removed wall pixels on the CLAHE image.
    Red = added, Blue = removed
    """
    base = bgr_to_rgb(photo_bgr).astype(np.float32)
    out = base.copy()

    added = (mask_after == 1) & (mask_before == 0)
    removed = (mask_after == 0) & (mask_before == 1)

    red = np.array([220, 40, 40], dtype=np.float32)
    blue = np.array([40, 80, 220], dtype=np.float32)

    out[added] = (1 - alpha) * out[added] + alpha * red
    out[removed] = (1 - alpha) * out[removed] + alpha * blue

    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# Candidate picking
# =========================

def pick_candidates(df: pd.DataFrame, n_recovered: int, n_failed: int):
    recovered = df[(df["none_valid"] == 0) & (df["contrast_valid"] == 1)].copy()
    failed = df[(df["none_valid"] == 0) & (df["contrast_valid"] == 0)].copy()

    if len(recovered) == 0:
        print("[WARN] No recovered CLAHE examples found.")
    if len(failed) == 0:
        print("[WARN] No failed CLAHE examples found.")

    # Recovered: prefer strong changes and cleaner post-CLAHE structure
    if len(recovered) > 0:
        recovered["delta_wall_cc"] = recovered["none_wall_cc"] - recovered["contrast_wall_cc"]
        recovered = recovered.sort_values(
            by=["delta_wall_cc", "contrast_enclosed_free_ratio", "contrast_wall_cc"],
            ascending=[False, False, True]
        )

    # Failed: prefer cases where CLAHE still didn't help much or structure stayed bad
    if len(failed) > 0:
        failed["delta_wall_cc"] = failed["none_wall_cc"] - failed["contrast_wall_cc"]
        failed = failed.sort_values(
            by=["delta_wall_cc", "contrast_wall_cc", "contrast_enclosed_free_ratio"],
            ascending=[False, False, True]
        )

    recovered_ids = list(recovered["plan_id"].head(n_recovered)) if len(recovered) > 0 else []
    failed_ids = list(failed["plan_id"].head(n_failed)) if len(failed) > 0 else []
    return recovered_ids, failed_ids


# =========================
# Figure rendering
# =========================

def save_candidate_figure(
    plan_id: str,
    label: str,
    photo_root: Path,
    model: nn.Module,
    device: torch.device,
    img_size: int,
    out_path: Path,
    title_suffix: str = "",
):
    photo_path = photo_root / plan_id / "photo_rect.png"
    orig = read_bgr(photo_path)
    clahe = preprocess_contrast(orig)

    wall_none, _ = run_segmentation(orig, model, device, img_size)
    wall_clahe, _ = run_segmentation(clahe, model, device, img_size)

    orig_rs = cv2.resize(orig, (img_size, img_size), interpolation=cv2.INTER_AREA)
    clahe_rs = cv2.resize(clahe, (img_size, img_size), interpolation=cv2.INTER_AREA)

    box = union_bbox(wall_none, wall_clahe, pad=30)

    orig_crop = crop_box(orig_rs, box)
    clahe_crop = crop_box(clahe_rs, box)
    wall_none_crop = crop_box((wall_none * 255).astype(np.uint8), box)
    wall_clahe_crop = crop_box((wall_clahe * 255).astype(np.uint8), box)

    wall_none_bin = crop_box(wall_none, box)
    wall_clahe_bin = crop_box(wall_clahe, box)

    diff_mask_vis = make_diff_overlay(wall_none_bin, wall_clahe_bin)
    diff_photo_vis = make_diff_overlay_on_photo(clahe_crop, wall_none_bin, wall_clahe_bin, alpha=0.60)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))

    axs[0, 0].imshow(bgr_to_rgb(orig_crop))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("(a) Original photograph", fontsize=10)

    axs[0, 1].imshow(bgr_to_rgb(clahe_crop))
    axs[0, 1].axis("off")
    axs[0, 1].set_title("(b) CLAHE-enhanced photograph", fontsize=10)

    axs[0, 2].imshow(diff_photo_vis)
    axs[0, 2].axis("off")
    axs[0, 2].set_title("(c) Change regions over CLAHE image", fontsize=10)

    axs[1, 0].imshow(wall_none_crop, cmap="gray", vmin=0, vmax=255)
    axs[1, 0].axis("off")
    axs[1, 0].set_title("(d) Wall mask without CLAHE", fontsize=10)

    axs[1, 1].imshow(wall_clahe_crop, cmap="gray", vmin=0, vmax=255)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("(e) Wall mask with CLAHE", fontsize=10)

    axs[1, 2].imshow(diff_mask_vis)
    axs[1, 2].axis("off")
    axs[1, 2].set_title("(f) Difference map: added=red, removed=blue", fontsize=10)

    plt.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[OK] Wrote: {out_path}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to mitigation.csv")
    ap.add_argument("--photo_root", required=True, help="Root containing <plan_id>/photo_rect.png")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint")
    ap.add_argument("--out_dir", required=True, help="Folder to write output figures")
    ap.add_argument("--img_size", type=int, default=896)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_recovered", type=int, default=3, help="How many recovered candidates to export")
    ap.add_argument("--n_failed", type=int, default=3, help="How many failed candidates to export")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    recovered_ids, failed_ids = pick_candidates(df, args.n_recovered, args.n_failed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = SegmentationHead(num_classes=3).to(device).eval()
    load_checkpoint(model, args.ckpt, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, plan_id in enumerate(recovered_ids, start=1):
        save_candidate_figure(
            plan_id=plan_id,
            label="CLAHE recovery candidate",
            photo_root=Path(args.photo_root),
            model=model,
            device=device,
            img_size=args.img_size,
            out_path=out_dir / f"recovered_{i:02d}_{plan_id}.png",
        )

    for i, plan_id in enumerate(failed_ids, start=1):
        save_candidate_figure(
            plan_id=plan_id,
            label="CLAHE non-recovery candidate",
            photo_root=Path(args.photo_root),
            model=model,
            device=device,
            img_size=args.img_size,
            out_path=out_dir / f"failed_{i:02d}_{plan_id}.png",
        )

    print("[OK] Done.")
    print(f"[INFO] Recovered candidates: {recovered_ids}")
    print(f"[INFO] Failed candidates: {failed_ids}")
    print(f"[INFO] Output folder: {out_dir}")


if __name__ == "__main__":
    main()
