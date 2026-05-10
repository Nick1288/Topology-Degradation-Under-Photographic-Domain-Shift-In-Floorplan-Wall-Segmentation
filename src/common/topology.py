from __future__ import annotations

import cv2
import numpy as np


def connected_components_count(mask: np.ndarray) -> int:
    n, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(max(0, n - 1))


def obstacle_roi(obstacles: np.ndarray, pad: int = 30):
    ys, xs = np.where(obstacles > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = obstacles.shape
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w - 1, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h - 1, int(ys.max()) + pad)
    return x0, y0, x1, y1


def flood_outside(free: np.ndarray) -> np.ndarray:
    h, w = free.shape
    outside = np.zeros((h, w), np.uint8)
    remaining = free.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    seeds = []
    for x in range(w):
        if remaining[0, x] == 1:
            seeds.append((x, 0))
        if remaining[h - 1, x] == 1:
            seeds.append((x, h - 1))
    for y in range(h):
        if remaining[y, 0] == 1:
            seeds.append((0, y))
        if remaining[y, w - 1] == 1:
            seeds.append((w - 1, y))

    for sx, sy in seeds:
        if outside[sy, sx] == 1:
            continue
        tmp = remaining.copy()
        cv2.floodFill(tmp, mask, (sx, sy), 2)
        newly_outside = (tmp == 2).astype(np.uint8)
        outside = np.maximum(outside, newly_outside)
        remaining[newly_outside == 1] = 0
    return outside


def compute_topology_metrics(wall01: np.ndarray, door01: np.ndarray, roi_pad: int, use_doors: bool, door_dilate: int):
    wall = wall01.astype(np.uint8)
    door = door01.astype(np.uint8)

    if use_doors and door_dilate > 0:
        k = 2 * door_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        door = cv2.dilate(door, kernel, iterations=1)

    obstacles = np.maximum(wall, door) if use_doors else wall.copy()
    h, w = obstacles.shape
    roi = obstacle_roi(obstacles, pad=roi_pad)
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w - 1, h - 1)

    obs_c = obstacles[y0:y1 + 1, x0:x1 + 1]
    wall_c = wall[y0:y1 + 1, x0:x1 + 1]
    free = (1 - obs_c).astype(np.uint8)
    outside = flood_outside(free)
    enclosed = (free & (1 - outside)).astype(np.uint8)

    n_enc, _, _, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=8)
    free_total = float(free.sum() + 1e-6)

    return {
        "roi": f"{x0},{y0},{x1},{y1}",
        "wall_cc": connected_components_count(wall_c),
        "wall_area_ratio": float(wall_c.sum() / ((wall_c.shape[0] * wall_c.shape[1]) + 1e-6)),
        "enclosure_count": int(max(0, n_enc - 1)),
        "enclosed_free_ratio": float(enclosed.sum() / free_total),
        "outside_free_ratio": float(outside.sum() / free_total),
    }


def is_structurally_valid(metrics: dict, wall_cc_threshold: int, enclosed_threshold: float, outside_threshold: float) -> bool:
    return (
        metrics["enclosure_count"] > 0
        and metrics["enclosed_free_ratio"] > enclosed_threshold
        and metrics["outside_free_ratio"] < outside_threshold
        and metrics["wall_cc"] < wall_cc_threshold
    )

