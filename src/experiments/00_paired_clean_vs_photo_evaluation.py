import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from common.image import read_bgr_white_background
from common.model import SegmentationHead, load_checkpoint, run_segmentation
from common.topology import compute_topology_metrics, is_structurally_valid


def save_bin(path: Path, mask01: np.ndarray):
    cv2.imwrite(str(path), (mask01.astype(np.uint8) * 255))


def read_bin(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Missing mask: {path}")
    return (mask > 0).astype(np.uint8)


def make_overlay(base_bgr: np.ndarray, wall01: np.ndarray, door01: np.ndarray, text: str):
    overlay = base_bgr.copy()
    wall_color = np.array([255, 0, 0], dtype=np.uint8)
    door_color = np.array([0, 0, 255], dtype=np.uint8)

    wall_mask = wall01.astype(bool)
    if wall_mask.any():
        overlay[wall_mask] = (
            overlay[wall_mask].astype(np.float32) * 0.65
            + wall_color.astype(np.float32) * 0.35
        ).astype(np.uint8)

    door_mask = door01.astype(bool)
    if door_mask.any():
        overlay[door_mask] = (
            overlay[door_mask].astype(np.float32) * 0.45
            + door_color.astype(np.float32) * 0.55
        ).astype(np.uint8)

    cv2.putText(overlay, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-pred-root", required=True, help="Folder containing per-plan photo predictions")
    parser.add_argument("--clean-raster-root", required=True, help="Folder containing <plan_id>/clean_rgb.png")
    parser.add_argument("--ckpt", required=True, help="Fine-tuned checkpoint")
    parser.add_argument("--out-root", required=True, help="Output folder for paired results")
    parser.add_argument("--img-size", type=int, default=896)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--roi-pad", type=int, default=30)
    parser.add_argument("--use-doors", action="store_true")
    parser.add_argument("--door-dilate", type=int, default=0)
    parser.add_argument("--thr-wall-cc", type=int, default=30)
    parser.add_argument("--thr-enclosed", type=float, default=0.25)
    parser.add_argument("--thr-outside", type=float, default=0.75)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--max-overlays", type=int, default=50)
    args = parser.parse_args()

    photo_pred_root = Path(args.photo_pred_root)
    clean_root = Path(args.clean_raster_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    preds_clean_root = out_root / "preds_clean"
    preds_clean_root.mkdir(parents=True, exist_ok=True)
    overlays_root = out_root / "overlays"
    overlays_root.mkdir(parents=True, exist_ok=True)

    plan_dirs = sorted(path for path in photo_pred_root.iterdir() if path.is_dir())
    if args.limit and args.limit > 0:
        plan_dirs = plan_dirs[:args.limit]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = SegmentationHead(num_classes=3).to(device).eval()
    load_checkpoint(model, args.ckpt, device)

    rows = []
    degraded_overlays_saved = 0

    for plan_dir in plan_dirs:
        plan_id = plan_dir.name
        photo_wall_path = plan_dir / "pred_walllike.png"
        photo_door_path = plan_dir / "pred_door.png"
        photo_img_path = plan_dir / "photo_rect.png"
        clean_img_path = clean_root / plan_id / "clean_rgb.png"

        if not photo_wall_path.exists() or not clean_img_path.exists():
            continue

        photo_wall = read_bin(photo_wall_path)
        photo_door = read_bin(photo_door_path) if photo_door_path.exists() else np.zeros_like(photo_wall, np.uint8)
        photo_metrics = compute_topology_metrics(
            photo_wall,
            photo_door,
            roi_pad=args.roi_pad,
            use_doors=args.use_doors,
            door_dilate=args.door_dilate,
        )
        photo_ok = is_structurally_valid(photo_metrics, args.thr_wall_cc, args.thr_enclosed, args.thr_outside)

        clean_img = read_bgr_white_background(clean_img_path)
        if clean_img is None:
            continue

        clean_wall, clean_door = run_segmentation(clean_img, model, device, args.img_size)
        clean_metrics = compute_topology_metrics(
            clean_wall,
            clean_door,
            roi_pad=args.roi_pad,
            use_doors=args.use_doors,
            door_dilate=args.door_dilate,
        )
        clean_ok = is_structurally_valid(clean_metrics, args.thr_wall_cc, args.thr_enclosed, args.thr_outside)
        degraded = clean_ok and not photo_ok

        clean_out_dir = preds_clean_root / plan_id
        clean_out_dir.mkdir(parents=True, exist_ok=True)
        clean_resized = cv2.resize(clean_img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(clean_out_dir / "clean_rgb.png"), clean_resized)
        save_bin(clean_out_dir / "pred_walllike.png", clean_wall)
        save_bin(clean_out_dir / "pred_door.png", clean_door)

        rows.append({
            "plan_id": plan_id,
            "clean_valid": int(clean_ok),
            "photo_valid": int(photo_ok),
            "degraded_clean_valid_photo_broken": int(degraded),
            "clean_wall_cc": clean_metrics["wall_cc"],
            "clean_enclosure_count": clean_metrics["enclosure_count"],
            "clean_enclosed_free_ratio": clean_metrics["enclosed_free_ratio"],
            "clean_outside_free_ratio": clean_metrics["outside_free_ratio"],
            "clean_wall_area_ratio": clean_metrics["wall_area_ratio"],
            "photo_wall_cc": photo_metrics["wall_cc"],
            "photo_enclosure_count": photo_metrics["enclosure_count"],
            "photo_enclosed_free_ratio": photo_metrics["enclosed_free_ratio"],
            "photo_outside_free_ratio": photo_metrics["outside_free_ratio"],
            "photo_wall_area_ratio": photo_metrics["wall_area_ratio"],
        })

        if args.save_overlays and degraded and degraded_overlays_saved < args.max_overlays:
            photo_img = cv2.imread(str(photo_img_path)) if photo_img_path.exists() else None
            if photo_img is None:
                photo_img = np.full((args.img_size, args.img_size, 3), 255, np.uint8)
            else:
                photo_img = cv2.resize(photo_img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

            clean_overlay = make_overlay(clean_resized, clean_wall, clean_door, f"CLEAN valid=1 cc={clean_metrics['wall_cc']}")
            photo_overlay = make_overlay(photo_img, photo_wall, photo_door, f"PHOTO valid=0 cc={photo_metrics['wall_cc']}")
            cv2.imwrite(str(overlays_root / f"{plan_id}_clean_vs_photo.png"), np.concatenate([clean_overlay, photo_overlay], axis=1))
            degraded_overlays_saved += 1

    if not rows:
        raise RuntimeError("No paired rows produced. Check input paths and plan IDs.")

    out_csv = out_root / "paired_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    n_clean_ok = sum(row["clean_valid"] for row in rows)
    n_photo_ok = sum(row["photo_valid"] for row in rows)
    n_degraded = sum(row["degraded_clean_valid_photo_broken"] for row in rows)

    print("Pairs evaluated:", n)
    print(f"Clean valid: {n_clean_ok}/{n} ({n_clean_ok / n:.1%})")
    print(f"Photo valid: {n_photo_ok}/{n} ({n_photo_ok / n:.1%})")
    print(f"Degraded: {n_degraded}/{n} ({n_degraded / n:.1%})")
    print("[OUT]", out_csv)


if __name__ == "__main__":
    main()
