import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from common.model import SegmentationHead, load_checkpoint, run_segmentation
from common.preprocessing import preprocess_contrast, preprocess_deskew
from common.topology import compute_topology_metrics, is_structurally_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-ids", default="/app/paired_eval_out/analysis/degraded_ids.txt")
    ap.add_argument("--photo-root", default="/app/data/cubi_ft")
    ap.add_argument("--ckpt", default="/app/runs_w3d_door/best.pth")
    ap.add_argument("--out-dir", default="/app/mitigation_results")
    ap.add_argument("--img-size", type=int, default=896)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roi-pad", type=int, default=30)
    ap.add_argument("--use-doors", action="store_true")
    ap.add_argument("--door-dilate", type=int, default=0)
    ap.add_argument("--thr-wall-cc", type=int, default=30)
    ap.add_argument("--thr-enclosed", type=float, default=0.25)
    ap.add_argument("--thr-outside", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    degraded_ids = [x.strip() for x in Path(args.degraded_ids).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit and args.limit > 0:
        degraded_ids = degraded_ids[:args.limit]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = SegmentationHead(num_classes=3).to(device).eval()
    load_checkpoint(model, args.ckpt, device)

    rows = []

    for plan_id in degraded_ids:
        photo_path = Path(args.photo_root) / plan_id / "photo_rect.png"
        if not photo_path.exists():
            continue

        img = cv2.imread(str(photo_path))
        if img is None:
            continue

        versions = {
            "none": img,
            "contrast": preprocess_contrast(img),
            "deskew": preprocess_deskew(img),
            "both": preprocess_contrast(preprocess_deskew(img)),
        }

        row = {"plan_id": plan_id}

        for name, img_variant in versions.items():
            wall, door = run_segmentation(img_variant, model, device, args.img_size)
            metrics = compute_topology_metrics(
                wall, door,
                roi_pad=args.roi_pad,
                use_doors=args.use_doors,
                door_dilate=args.door_dilate,
            )
            ok = is_structurally_valid(metrics, args.thr_wall_cc, args.thr_enclosed, args.thr_outside)

            row[f"{name}_valid"] = int(ok)
            row[f"{name}_wall_cc"] = metrics["wall_cc"]
            row[f"{name}_enclosed_free_ratio"] = metrics["enclosed_free_ratio"]
            row[f"{name}_outside_free_ratio"] = metrics["outside_free_ratio"]

        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No rows produced.")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "mitigation.csv"
    df.to_csv(csv_path, index=False)

    baseline_failures = (df["none_valid"] == 0).sum()
    recovered_contrast = ((df["none_valid"] == 0) & (df["contrast_valid"] == 1)).sum()
    recovered_deskew = ((df["none_valid"] == 0) & (df["deskew_valid"] == 1)).sum()
    recovered_both = ((df["none_valid"] == 0) & (df["both_valid"] == 1)).sum()

    lines = []
    lines.append(f"Sample size: {len(df)}")
    lines.append("")
    for name in ["none", "contrast", "deskew", "both"]:
        lines.append(f"{name}: valid_rate={df[f'{name}_valid'].mean():.3f}")
    lines.append("")
    lines.append(f"baseline_failures: {baseline_failures}")
    if baseline_failures > 0:
        lines.append(f"recovered_contrast: {recovered_contrast}/{baseline_failures} ({recovered_contrast/baseline_failures:.3f})")
        lines.append(f"recovered_deskew: {recovered_deskew}/{baseline_failures} ({recovered_deskew/baseline_failures:.3f})")
        lines.append(f"recovered_both: {recovered_both}/{baseline_failures} ({recovered_both/baseline_failures:.3f})")
    else:
        lines.append("No baseline failures in the tested set.")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Wrote:")
    print(" -", csv_path)
    print(" -", out_dir / "summary.txt")


if __name__ == "__main__":
    main()
