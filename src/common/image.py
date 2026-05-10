from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def read_bgr_white_background(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = (a.astype(np.float32) / 255.0)[..., None]
        rgb = cv2.merge([r, g, b]).astype(np.float32)
        white = np.full_like(rgb, 255.0, dtype=np.float32)
        comp = np.clip(rgb * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)
        return cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def bgr_to_rgb(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def union_bbox(mask_a: np.ndarray, mask_b: np.ndarray, pad: int = 30):
    ys1, xs1 = np.where(mask_a > 0)
    ys2, xs2 = np.where(mask_b > 0)
    xs_all = np.concatenate([xs1, xs2]) if (len(xs1) + len(xs2)) else np.array([], dtype=np.int32)
    ys_all = np.concatenate([ys1, ys2]) if (len(ys1) + len(ys2)) else np.array([], dtype=np.int32)

    h, w = mask_a.shape[:2]
    if len(xs_all) == 0 or len(ys_all) == 0:
        return 0, 0, w, h

    x0 = max(0, int(xs_all.min()) - pad)
    y0 = max(0, int(ys_all.min()) - pad)
    x1 = min(w, int(xs_all.max()) + 1 + pad)
    y1 = min(h, int(ys_all.max()) + 1 + pad)
    return x0, y0, x1, y1


def crop_box(img: np.ndarray, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()

