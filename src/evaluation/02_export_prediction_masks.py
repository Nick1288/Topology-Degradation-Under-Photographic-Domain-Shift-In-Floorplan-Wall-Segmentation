import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from common.model import SegmentationHead, bgr_to_tensor, load_checkpoint

def save_bin(path: Path, m: np.ndarray):
    out = (m.astype(np.uint8) * 255)
    cv2.imwrite(str(path), out)

def make_overlay(base_bgr: np.ndarray, walllike01: np.ndarray, door01: np.ndarray,
                 alpha_wall: float = 0.35, alpha_door: float = 0.55) -> np.ndarray:
    """
    base_bgr: HxWx3 uint8
    walllike01, door01: HxW uint8 in {0,1}
    Returns overlay image (uint8).
    """
    overlay = base_bgr.copy()

    # Color layers (BGR). Pick high-contrast colors.
    wall_color = np.array([255, 0, 0], dtype=np.uint8)   # Blue
    door_color = np.array([0, 0, 255], dtype=np.uint8)   # Red

    # Apply walllike
    wmask = walllike01.astype(bool)
    if wmask.any():
        overlay[wmask] = (overlay[wmask].astype(np.float32) * (1 - alpha_wall) +
                          wall_color.astype(np.float32) * alpha_wall).astype(np.uint8)

    # Apply door (draw on top)
    dmask = door01.astype(bool)
    if dmask.any():
        overlay[dmask] = (overlay[dmask].astype(np.float32) * (1 - alpha_door) +
                          door_color.astype(np.float32) * alpha_door).astype(np.uint8)

    return overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="e.g. /app/data/cubi_ft")
    ap.add_argument("--ckpt", required=True, help="e.g. /app/runs_w3d_door/best.pth")
    ap.add_argument("--out-root", required=True, help="e.g. /app/preds_w3d_best")
    ap.add_argument("--img-size", type=int, default=896, help="must match training img-size")
    ap.add_argument("--split", choices=["all", "train", "val"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-mode", choices=["folder", "flat"], default="folder",
                    help="folder = subdirs with photo_rect.png, flat = direct PNGs")
    ap.add_argument("--overlay-root", default="", help="optional; default = <out-root>/../overlays")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # overlays folder: .../cubi/overlays/<id>_overlay.png
    if args.overlay_root.strip():
        overlay_root = Path(args.overlay_root)
    else:
        overlay_root = out_root.parent / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "folder":
        items = sorted([d for d in data_root.iterdir() if d.is_dir()])
        if len(items) == 0:
            raise RuntimeError(f"No folders found in {data_root}")

        if args.split != "all":
            split_idx = int(0.9 * len(items))
            if args.split == "train":
                items = items[:split_idx]
            elif args.split == "val":
                items = items[split_idx:]
    else:
        items = sorted(data_root.glob("*.png"))
        if len(items) == 0:
            raise RuntimeError(f"No PNGs found in {data_root}")

    if args.limit and args.limit > 0:
        items = items[:args.limit]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = SegmentationHead(num_classes=3).to(device).eval()
    load_checkpoint(model, args.ckpt, device)

    with torch.no_grad():
        for i, item in enumerate(items):
            if args.input_mode == "folder":
                img_path = item / "photo_rect.png"
                out_name = item.name
            else:
                img_path = item
                out_name = item.stem

            img = cv2.imread(str(img_path))
            if img is None:
                print("[SKIP] Missing image:", img_path)
                continue

            img_rs = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            x = bgr_to_tensor(img_rs).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

            walllike = (pred == 1).astype(np.uint8)
            door = (pred == 2).astype(np.uint8)

            out_dir = out_root / out_name
            out_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_dir / "photo_rect.png"), img_rs)
            save_bin(out_dir / "pred_walllike.png", walllike)
            save_bin(out_dir / "pred_door.png", door)

            # --- OVERLAY OUTPUT ---
            overlay = make_overlay(img_rs, walllike, door)
            overlay_path = overlay_root / f"{out_name}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

    print("[DONE] Export + overlays complete.")
    print(f"[OVERLAYS] {overlay_root}")

if __name__ == "__main__":
    main()
