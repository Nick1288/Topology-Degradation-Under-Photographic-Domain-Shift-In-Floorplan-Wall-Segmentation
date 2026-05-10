import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from common.model import SegmentationHead, load_backbone_only, load_checkpoint, run_segmentation
from common.topology import compute_topology_metrics, is_structurally_valid


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo-root", default="/app/data/cubi_ft")
    ap.add_argument("--base-ckpt", default="/app/model_best_val_loss_var.pkl")
    ap.add_argument("--finetuned-ckpt", default="/app/runs_w3d_door/best.pth")
    ap.add_argument("--out-dir", default="/app/base_comparison")
    ap.add_argument("--img-size", type=int, default=896)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roi-pad", type=int, default=30)
    ap.add_argument("--use-doors", action="store_true")
    ap.add_argument("--door-dilate", type=int, default=0)
    ap.add_argument("--thr-wall-cc", type=int, default=30)
    ap.add_argument("--thr-enclosed", type=float, default=0.25)
    ap.add_argument("--thr-outside", type=float, default=0.75)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    photo_root = Path(args.photo_root)
    photo_paths = sorted(photo_root.glob("*/photo_rect.png"))
    if args.limit and args.limit > 0:
        photo_paths = photo_paths[:args.limit]

    if len(photo_paths) == 0:
        raise RuntimeError(f"No photo_rect.png files found under {photo_root}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Base model: same architecture, ONLY backbone loaded from CubiCasa pkl
    base_model = SegmentationHead(num_classes=3).to(device).eval()
    load_backbone_only(base_model, args.base_ckpt, device=device)

    # Finetuned model: full checkpoint load
    ft_model = SegmentationHead(num_classes=3).to(device).eval()
    load_checkpoint(ft_model, args.finetuned_ckpt, device)

    rows = []

    for photo_path in photo_paths:
        plan_id = photo_path.parent.name
        img = cv2.imread(str(photo_path))
        if img is None:
            continue

        base_wall, base_door = run_segmentation(img, base_model, device, args.img_size)
        base_metrics = compute_topology_metrics(
            base_wall, base_door,
            roi_pad=args.roi_pad,
            use_doors=args.use_doors,
            door_dilate=args.door_dilate,
        )
        base_ok = is_structurally_valid(base_metrics, args.thr_wall_cc, args.thr_enclosed, args.thr_outside)

        ft_wall, ft_door = run_segmentation(img, ft_model, device, args.img_size)
        ft_metrics = compute_topology_metrics(
            ft_wall, ft_door,
            roi_pad=args.roi_pad,
            use_doors=args.use_doors,
            door_dilate=args.door_dilate,
        )
        ft_ok = is_structurally_valid(ft_metrics, args.thr_wall_cc, args.thr_enclosed, args.thr_outside)

        rows.append({
            "plan_id": plan_id,

            "base_valid": int(base_ok),
            "base_wall_cc": base_metrics["wall_cc"],
            "base_enclosure_count": base_metrics["enclosure_count"],
            "base_enclosed_free_ratio": base_metrics["enclosed_free_ratio"],
            "base_outside_free_ratio": base_metrics["outside_free_ratio"],

            "ft_valid": int(ft_ok),
            "ft_wall_cc": ft_metrics["wall_cc"],
            "ft_enclosure_count": ft_metrics["enclosure_count"],
            "ft_enclosed_free_ratio": ft_metrics["enclosed_free_ratio"],
            "ft_outside_free_ratio": ft_metrics["outside_free_ratio"],

            "wall_cc_improvement": base_metrics["wall_cc"] - ft_metrics["wall_cc"],
        })

    if len(rows) == 0:
        raise RuntimeError("No rows produced.")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)

    lines = []
    lines.append(f"Sample size: {len(df)}")
    lines.append("")
    lines.append("Base model (CubiCasa backbone only + untrained custom head):")
    lines.append(f"  valid_rate: {df['base_valid'].mean():.3f}")
    lines.append(f"  mean_wall_cc: {df['base_wall_cc'].mean():.3f}")
    lines.append("")
    lines.append("Finetuned model:")
    lines.append(f"  valid_rate: {df['ft_valid'].mean():.3f}")
    lines.append(f"  mean_wall_cc: {df['ft_wall_cc'].mean():.3f}")
    lines.append("")
    lines.append("Improvement:")
    lines.append(f"  validity_gain: {(df['ft_valid'].mean() - df['base_valid'].mean()):.3f}")
    lines.append(f"  mean_wall_cc_reduction: {df['wall_cc_improvement'].mean():.3f}")
    lines.append("")
    both_fail = ((df["base_valid"] == 0) & (df["ft_valid"] == 0)).sum()
    base_only_fail = ((df["base_valid"] == 0) & (df["ft_valid"] == 1)).sum()
    ft_only_fail = ((df["base_valid"] == 1) & (df["ft_valid"] == 0)).sum()
    lines.append("Failure breakdown:")
    lines.append(f"  both_fail: {both_fail}")
    lines.append(f"  base_only_fail (finetuning fixed): {base_only_fail}")
    lines.append(f"  ft_only_fail (regression): {ft_only_fail}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Wrote:")
    print(" -", csv_path)
    print(" -", out_dir / "summary.txt")


if __name__ == "__main__":
    main()
