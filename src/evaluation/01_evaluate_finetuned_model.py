#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuned CubiCasa evaluator for photographed floorplan inputs.

The evaluator uses sliding-window inference, wall sealing, flood-fill interior
recovery, door extraction, and wall-band constraints. It exports wall, door,
interior, stair, debug, and overlay outputs for each input image.
"""

import os, glob, argparse, time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
from floortrans.models import get_model as ft_get_model

# ---------------- I/O helpers ----------------

def normpath_abs(p: str) -> str:
    return Path(p).resolve().as_posix()

def imread_safe(p: str, flags=cv2.IMREAD_COLOR):
    p = normpath_abs(p)
    try:
        arr = np.fromfile(p, dtype=np.uint8)
        if arr.size > 0:
            return cv2.imdecode(arr, flags)
    except Exception:
        pass
    return cv2.imread(p, flags)

def mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")

def _save_bundle(stem_base: str, name: str, bin01: np.ndarray, prob: np.ndarray):
    cv2.imwrite(f"{stem_base}_{name}.png", (bin01 * 255).astype(np.uint8))

def _resize_keep_aspect(img, long_side):
    H, W = img.shape[:2]
    if max(H, W) == long_side:
        return img, 1.0
    scale = long_side / float(max(H, W))
    new_W = int(round(W * scale))
    new_H = int(round(H * scale))
    return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR), scale

# ---------------- Core helpers ----------------

def _u8(x):
    return (x > 0).astype(np.uint8)

def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    return k

def _morph_open(bin01: np.ndarray, k: int, it: int = 1) -> np.ndarray:
    k = _ensure_odd(k)
    ker = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(bin01.astype(np.uint8), cv2.MORPH_OPEN, ker, iterations=int(it))

def _morph_close(bin01: np.ndarray, k: int, it: int = 1) -> np.ndarray:
    k = _ensure_odd(k)
    ker = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(bin01.astype(np.uint8), cv2.MORPH_CLOSE, ker, iterations=int(it))

def _skeletonize_u8(bin01: np.ndarray) -> np.ndarray:
    """Pure OpenCV morphological skeletonization. Input/Output: uint8 0/1."""
    img = _u8(bin01) * 255
    skel = np.zeros_like(img, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)

def _count_cc(bin01: np.ndarray, min_area: int = 1) -> int:
    m = _u8(bin01)
    if m.sum() == 0: return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1: return 0
    if min_area <= 1: return n - 1
    cnt = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            cnt += 1
    return cnt

def _largest_k_cc(bin01: np.ndarray, k: int = 4, min_area: int = 10):
    m = _u8(bin01)
    if m.sum() == 0: return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    pairs = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= int(min_area):
            pairs.append((a, i))
    pairs.sort(reverse=True)
    return pairs[:int(k)]

def _cut_oriented_line(mask01: np.ndarray, cy: int, cx: int, dy: float, dx: float,
                       length: int = 21, thickness: int = 7) -> np.ndarray:
    out = mask01.copy().astype(np.uint8)
    H, W = out.shape

    # normalize direction
    n = (dx*dx + dy*dy) ** 0.5
    if n < 1e-6:
        return out

    dx /= n; dy /= n

    # perpendicular direction (cut line direction)
    px, py = -dy, dx

    half = length // 2
    x0 = int(round(cx - px * half))
    y0 = int(round(cy - py * half))
    x1 = int(round(cx + px * half))
    y1 = int(round(cy + py * half))

    # clip roughly
    x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
    y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))

    cv2.line(out, (x0, y0), (x1, y1), 0, thickness=int(thickness))
    return out

# ---------------- Endpoint Logic Helpers ----------------

def _neighbors8_degree(skel01: np.ndarray) -> np.ndarray:
    s = _u8(skel01)
    k = np.ones((3, 3), np.uint8)
    n = cv2.filter2D(s, -1, k, borderType=cv2.BORDER_CONSTANT)
    return (n - s).astype(np.uint8)

def _endpoint_points(skel01: np.ndarray):
    deg = _neighbors8_degree(skel01)
    ys, xs = np.where((skel01 > 0) & (deg == 1))
    return list(zip(ys.tolist(), xs.tolist()))

def _cluster_points(points, r: int = 3):
    if not points: return []
    r = int(r)
    clusters = []
    unused = set(range(len(points)))
    while unused:
        i = next(iter(unused))
        unused.remove(i)
        cy, cx = points[i]
        cluster = [(cy, cx)]
        changed = True
        while changed:
            changed = False
            to_add = []
            for j in list(unused):
                y, x = points[j]
                if max(abs(y - cy), abs(x - cx)) <= r:
                    to_add.append(j)
            if to_add:
                changed = True
                for j in to_add:
                    unused.remove(j)
                    cluster.append(points[j])
                cy = int(round(sum(p[0] for p in cluster) / len(cluster)))
                cx = int(round(sum(p[1] for p in cluster) / len(cluster)))
        clusters.append(cluster)
    return clusters

def endpt_seeds_from_prob(pr_endpt: np.ndarray,
                          door01: np.ndarray,
                          thr: float = 0.55,
                          nms_r: int = 6,
                          door_band_ks: int = 21,
                          max_seeds_per_comp: int = 6,
                          min_comp_area: int = 60) -> np.ndarray:
    """
    Returns sparse endpoint seeds (uint8 0/1) from endpoint probability map.
    - restricts to a dilated door band so endpoints off-door don't count
    - NMS to avoid clusters
    """
    door = (door01 > 0).astype(np.uint8)
    if door.sum() == 0:
        return np.zeros_like(door, np.uint8)

    ks = int(door_band_ks)
    if ks < 3: ks = 3
    if ks % 2 == 0: ks += 1
    band = cv2.dilate(door, np.ones((ks, ks), np.uint8), 1).astype(bool)

    # candidate map
    cand = (pr_endpt > float(thr)) & band
    cand = cand.astype(np.uint8)
    if cand.sum() == 0:
        return np.zeros_like(door, np.uint8)

    # split into door components; pick seeds per component only
    nC, lab, stats, _ = cv2.connectedComponentsWithStats(door, 8)
    seeds = np.zeros_like(door, np.uint8)

    rr = int(max(1, nms_r))

    for cid in range(1, nC):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < int(min_comp_area):
            continue
        comp = (lab == cid)
        comp_cand = cand & comp

        if comp_cand.sum() == 0:
            continue

        # greedy peak pick using probability as score
        score = pr_endpt.copy()
        score[~comp] = -1.0
        score[~band] = -1.0
        score[~comp_cand.astype(bool)] = -1.0

        picked = 0
        while picked < int(max_seeds_per_comp):
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[y, x] < float(thr):
                break
            seeds[y, x] = 1
            picked += 1
            y0, y1 = max(0, y-rr), min(score.shape[0], y+rr+1)
            x0, x1 = max(0, x-rr), min(score.shape[1], x+rr+1)
            score[y0:y1, x0:x1] = -1.0

    return seeds.astype(np.uint8)

def _best_neck_point_from_endpoints(comp01: np.ndarray, endpt_seeds01: np.ndarray,
                                   min_pair_dist: int = 10,
                                   sample_step: int = 1):
    """
    Given a door component and endpoint seeds, find a neck point to cut:
    - choose best endpoint pair (far enough apart)
    - along their connecting line, find minimum DT (pinch)
    Returns (cy, cx, y1, x1, y2, x2) or None.
    """
    comp = (comp01 > 0).astype(np.uint8)
    if comp.sum() == 0:
        return None

    ys, xs = np.where((endpt_seeds01 > 0) & (comp > 0))
    pts = list(zip(ys.tolist(), xs.tolist()))
    if len(pts) < 2:
        return None

    # DT inside comp
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3).astype(np.float32)
    dmax = float(dist.max())
    if dmax <= 0:
        return None

    best = None
    best_score = -1e9

    # try pairs (cap for safety)
    pts = pts[:12]
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            y1, x1 = pts[i]
            y2, x2 = pts[j]
            if max(abs(y2-y1), abs(x2-x1)) < int(min_pair_dist):
                continue

            L = int(max(abs(x2-x1), abs(y2-y1)) / max(1, int(sample_step))) + 1
            xs_l = np.linspace(x1, x2, L).astype(np.int32)
            ys_l = np.linspace(y1, y2, L).astype(np.int32)
            
            # Bounds check
            H, W = comp.shape
            xs_l = np.clip(xs_l, 0, W-1)
            ys_l = np.clip(ys_l, 0, H-1)
            
            line = dist[ys_l, xs_l]

            # ignore endpoints themselves
            core = line[2:-2] if line.size > 6 else line
            if core.size == 0:
                continue

            valley = float(core.min())
            peak_min = float(min(dist[y1, x1], dist[y2, x2]))
            # If endpoints are on border, peak_min might be 0, clamp it
            peak_min = max(1.0, peak_min)

            # deeper pinch => better; also prefer wider doors (peak_min higher)
            score = (1.0 - valley / peak_min) + 0.15 * peak_min
            if score > best_score:
                best_score = score
                idx = int(np.argmin(core))
                # map back to full line index
                neck_i = idx + (2 if line.size > 6 else 0)
                best = (int(ys_l[neck_i]), int(xs_l[neck_i]), y1, x1, y2, x2)

    return best

def save_endpoint_debug(stem_base: str,
                        bgr_scaled: np.ndarray,
                        door01: np.ndarray,
                        pr_endpt: np.ndarray,
                        seeds01: np.ndarray,
                        thr: float = 0.55):
    H, W = door01.shape

    # 1) Seeds overlay on doors
    vis = bgr_scaled.copy()
    door_mask = (door01 > 0)
    vis[door_mask] = (0, 255, 0)  # doors green

    ys, xs = np.where(seeds01 > 0)
    for (y, x) in zip(ys.tolist(), xs.tolist()):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)  # seeds red dots

    # show raw endpoint pixels above thr as small blue dots too (optional but useful)
    cand = (pr_endpt > float(thr))
    cy, cx = np.where(cand)
    # downsample the dots so it doesn't become a solid blob
    step = 4
    for (y, x) in zip(cy[::step].tolist(), cx[::step].tolist()):
        cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1)  # cand blue

    cv2.imwrite(f"{stem_base}_endpt_on_doors.jpg", vis)

    # 2) Save endpoint prob heatmap (grayscale)
    heat = np.clip(pr_endpt * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{stem_base}_endpt_prob.png", heat)

    # 3) Save seeds mask
    cv2.imwrite(f"{stem_base}_endpt_seeds.png", (seeds01.astype(np.uint8) * 255))

def _count_cc_minarea(bin01: np.ndarray, min_area: int = 45) -> int:
    m = (bin01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    cnt = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            cnt += 1
    return cnt

def _cut_disk(mask01: np.ndarray, cy: int, cx: int, rad: int = 2) -> np.ndarray:
    out = mask01.copy()
    H, W = out.shape
    rad = int(max(0, rad))
    y0, y1 = max(0, cy-rad), min(H, cy+rad+1)
    x0, x1 = max(0, cx-rad), min(W, cx+rad+1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - cy)**2 + (xx - cx)**2 <= rad**2
    out[y0:y1, x0:x1][disk] = 0
    return out.astype(np.uint8)

def split_component_by_endpoints(comp01: np.ndarray,
                                 endpt_seeds01: np.ndarray,
                                 *,
                                 cut_r: int = 2,
                                 min_piece_area: int = 45,
                                 require_gain: int = 1) -> (np.ndarray, bool):
    """
    Attempt to split a single connected door component using endpoints.
    Accept only if CC count increases by >= require_gain.
    """
    comp = (comp01 > 0).astype(np.uint8)
    base_cc = _count_cc_minarea(comp, min_area=min_piece_area)

    best = _best_neck_point_from_endpoints(comp, endpt_seeds01)
    if best is None:
        return comp, False

    cy, cx, y1, x1, y2, x2 = best
    dy, dx = (y2 - y1), (x2 - x1)

    cand = _cut_oriented_line(comp, cy, cx, dy, dx, length=21, thickness=7)

    new_cc = _count_cc_minarea(cand, min_area=min_piece_area)

    if new_cc >= base_cc + int(require_gain):
        return cand.astype(np.uint8), True
    return comp, False

def split_doors_by_endpoints(
    door01: np.ndarray,
    pr_endpt: np.ndarray,
    *,
    endpt_thr: float = 0.55,
    endpt_nms_r: int = 6,
    endpt_band_ks: int = 21,
    min_comp_area: int = 60,
    cut_r: int = 2,
    min_piece_area: int = 45,
    max_passes: int = 2
) -> np.ndarray:
    """
    Component-wise endpoint-guided splitting with strict acceptance gating.
    - does NOT assume "2 endpoints per door"
    - will only split if it increases CC count
    """
    door = (door01 > 0).astype(np.uint8)
    if door.sum() == 0:
        return door

    out = door.copy()

    # precompute endpoint seeds once against current door
    # (we will recompute per pass as topology changes)
    for _ in range(int(max_passes)):
        seeds = endpt_seeds_from_prob(
            pr_endpt=pr_endpt,
            door01=out,
            thr=endpt_thr,
            nms_r=endpt_nms_r,
            door_band_ks=endpt_band_ks,
            max_seeds_per_comp=6,
            min_comp_area=min_comp_area
        )

        nC, lab, stats, _ = cv2.connectedComponentsWithStats(out, 8)
        changed = False

        for cid in range(1, nC):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < int(min_comp_area):
                continue

            comp_mask = (lab == cid)
            comp = (out & comp_mask.astype(np.uint8)).astype(np.uint8)
            comp_seeds = (seeds & comp).astype(np.uint8)

            # need at least 2 seeds to attempt a split
            if int(comp_seeds.sum()) < 2:
                continue

            new_comp, did = split_component_by_endpoints(
                comp, comp_seeds,
                cut_r=cut_r,
                min_piece_area=min_piece_area,
                require_gain=1
            )
            if did:
                out[comp_mask] = 0
                out[comp_mask] = new_comp[comp_mask].astype(np.uint8)
                changed = True

        if not changed:
            break

    return out.astype(np.uint8)

# ---------------- Model Definition ----------------
class FineTunedEvalWrapper20(nn.Module):
    def __init__(self, arch="hg_furukawa_original", n_out=20):
        super().__init__()
        self.base = ft_get_model(arch, 51)
        # Match checkpoint upsample structure
        self.base.upsample = nn.ConvTranspose2d(51, 51, kernel_size=4, stride=4, padding=0, bias=True)
        
        # Head taps 256-ch features + 2 Coord channels = 258
        self.head = nn.Conv2d(256 + 2, n_out, kernel_size=1, bias=True)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(n_out, n_out, kernel_size=1, bias=True),
        )

    def forward(self, x):
        feats = {}
        def hook_fn(module, inp, out):
            feats["f256"] = inp[0]

        h = self.base.conv4_.register_forward_hook(hook_fn)
        try:
            _ = self.base(x)
        finally:
            h.remove()

        f256 = feats.get("f256", None)
        if f256 is None:
            raise RuntimeError("Failed to capture 256ch features.")

        # --- CoordConv Injection (Replicates training logic) ---
        N, C, H, W = f256.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=f256.device),
            torch.linspace(-1, 1, W, device=f256.device),
            indexing='ij'
        )
        grid = torch.stack((yy, xx), dim=0).unsqueeze(0).repeat(N, 1, 1, 1)
        f_coord = torch.cat([f256, grid], dim=1) # [N, 258, H, W]
        # -----------------------------------------------------

        y = self.head(f_coord)
        y = self.up(y)
        return y


def load_model_safe(arch: str, weights: str, device: torch.device):
    print(f"[build] arch={arch} n_classes=20")
    model = FineTunedEvalWrapper20(arch, n_out=20)

    if not (weights and os.path.exists(weights)):
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"[load] weights: {weights}")
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)

    # Hard sanity checks so you fail fast with a useful message
    if "head.weight" not in state_dict or "up.1.weight" not in state_dict:
        missing = [k for k in ["head.weight", "up.1.weight"] if k not in state_dict]
        raise RuntimeError(
            "This eval wrapper expects a checkpoint with separate 'head' and 'up' modules. "
            f"Missing keys: {missing}. If you trained a different layout, say so and we match it."
        )
    if "base.conv4_.weight" in state_dict:
        w = state_dict["base.conv4_.weight"]
        if w.shape[0] != 51:
            raise RuntimeError(f"Unexpected base.conv4_ out_ch={w.shape[0]} (expected 51).")

    model.load_state_dict(state_dict, strict=True)
    print("[load] Strict load successful.")

    model.eval().to(device)
    return model

# ---------------- Door Logic ----------------

def door_mask_from_prob(pr_blob: np.ndarray, thr: float = 0.25, blur_ks: int = 3, close_k: int = 5) -> np.ndarray:
    p = pr_blob.astype(np.float32)
    k = _ensure_odd(blur_ks)
    if k >= 3:
        p = cv2.GaussianBlur(p, (k, k), 0)
    
    # Aggressive threshold
    door = (p > float(thr)).astype(np.uint8)
    
    # Directional closing preserves thin vertical and horizontal doors.
    # 1. Close vertically.
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    door = cv2.morphologyEx(door, cv2.MORPH_CLOSE, v_kernel)
    
    # 2. Close horizontally.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    door = cv2.morphologyEx(door, cv2.MORPH_CLOSE, h_kernel)
    
    # 3. Final generic clean-up
    ck = _ensure_odd(close_k)
    if ck >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        door = cv2.morphologyEx(door, cv2.MORPH_CLOSE, kernel, iterations=1)
        
    return door.astype(np.uint8)

def apply_cut_safely(door_bin01: np.ndarray, cut_prob: np.ndarray, center_prob: np.ndarray = None, *,
                     cut_thr: float = 0.85, door_band_ks: int = 31, protect_center_thr: float = 0.50) -> np.ndarray:
    door_bin01 = _u8(door_bin01)
    if door_bin01.sum() == 0: return door_bin01
    ks = _ensure_odd(int(door_band_ks))
    door_band = cv2.dilate(door_bin01, np.ones((ks, ks), np.uint8), 1).astype(bool)
    cut_mask = (cut_prob > float(cut_thr)) & door_band
    if center_prob is not None:
        protect = (center_prob > float(protect_center_thr))
        cut_mask = cut_mask & (~protect)
    out = door_bin01.copy()
    out[cut_mask] = 0
    return out.astype(np.uint8)

# ---------------- Wall/Interior Logic ----------------

def support_guided_close(wall01: np.ndarray, pr_wall: np.ndarray, close_k: int = 11,
                         close_it: int = 1, support_thr: float = 0.22) -> np.ndarray:
    wall = (wall01 > 0).astype(np.uint8)
    k = _ensure_odd(close_k)
    if k < 3: return wall
    ker = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, ker, iterations=int(close_it)).astype(np.uint8)
    added = (closed == 1) & (wall == 0)
    keep_added = added & (pr_wall > float(support_thr))
    out = wall.copy()
    out[keep_added] = 1
    return out.astype(np.uint8)

def build_wall_obstacles(pr_wall: np.ndarray, t_wall: float = 0.40, close_k: int = 5, close_it: int = 1,
                         dilate_k: int = 3, dilate_it: int = 1, open_k: int = 0, open_it: int = 1,
                         do_plug: bool = False, plug_border_band: int = 24, plug_support_thr: float = 0.22,
                         plug_close_k: int = 11, micro_close_k: int = 3, micro_close_it: int = 1,
                         sg_close_k: int = 13, sg_close_it: int = 1, sg_support_thr: float = 0.22) -> np.ndarray:
    wall = (pr_wall > float(t_wall)).astype(np.uint8)
    if int(open_k) >= 3:
        wall = _morph_open(wall, int(open_k), int(open_it))
    if int(micro_close_k) >= 3:
        wall = _morph_close(wall, int(micro_close_k), int(micro_close_it))
    wall = support_guided_close(wall, pr_wall, close_k=int(sg_close_k),
                                close_it=int(sg_close_it), support_thr=float(sg_support_thr))
    return wall.astype(np.uint8)

def floodfill_outside_from_border(free01: np.ndarray, step: int = 8) -> np.ndarray:
    free = (free01 > 0).astype(np.uint8)
    H, W = free.shape[:2]
    im = (free * 255).astype(np.uint8)
    mask = np.zeros((H + 2, W + 2), np.uint8)
    st = max(1, int(step))
    for x in range(0, W, st):
        if im[0, x] == 255: cv2.floodFill(im, mask, (x, 0), 128)
        if im[H - 1, x] == 255: cv2.floodFill(im, mask, (x, H - 1), 128)
    for y in range(0, H, st):
        if im[y, 0] == 255: cv2.floodFill(im, mask, (0, y), 128)
        if im[y, W - 1] == 255: cv2.floodFill(im, mask, (W - 1, y), 128)
    return (im == 128).astype(np.uint8)

def make_interior_by_floodfill(wall01: np.ndarray):
    wall = (wall01 > 0).astype(np.uint8)
    free = (wall == 0).astype(np.uint8)
    outside = floodfill_outside_from_border(free)
    interior = (free > 0) & (outside == 0)
    return interior.astype(np.uint8), outside.astype(np.uint8), free.astype(np.uint8)

def vote_fix_interior(interior01: np.ndarray, outside01: np.ndarray, pr_room: np.ndarray = None,
                      pr_sem_room: np.ndarray = None, pr_door: np.ndarray = None, pr_stair: np.ndarray = None,
                      t_room: float = 0.45, t_sem_room: float = 0.50, t_door: float = 0.55, t_stair: float = 0.55,
                      min_component_area: int = 1200, border_margin: int = 12, use_distance_guard: bool = True,
                      dist_min: int = 18) -> np.ndarray:
    interior = (interior01 > 0).astype(np.uint8)
    outside = (outside01 > 0).astype(np.uint8)
    H, W = interior.shape[:2]
    cand = (outside == 1).astype(np.uint8)
    bm = max(0, int(border_margin))
    if bm > 0:
        cand[:bm, :] = 0; cand[-bm:, :] = 0; cand[:, :bm] = 0; cand[:, -bm:] = 0
    if bool(use_distance_guard):
        valid = np.zeros((H, W), np.uint8)
        valid[bm:H - bm, bm:W - bm] = 1
        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
        cand[dist < float(dist_min)] = 0
    vote = np.zeros((H, W), np.uint8)
    if pr_room is not None: vote |= (pr_room > float(t_room)).astype(np.uint8)
    if pr_sem_room is not None: vote |= (pr_sem_room > float(t_sem_room)).astype(np.uint8)
    if pr_door is not None: vote |= (pr_door > float(t_door)).astype(np.uint8)
    if pr_stair is not None: vote |= (pr_stair > float(t_stair)).astype(np.uint8)
    cand = ((cand > 0) & (vote > 0)).astype(np.uint8)
    if cand.sum() == 0: return interior
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    promote = np.zeros_like(interior, np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_component_area): continue
        promote[lab == i] = 1
    out = interior.copy()
    out[promote > 0] = 1
    return out.astype(np.uint8)

def interior_pipeline_floodfill_plus_vote(pr_wall: np.ndarray, pr_room: np.ndarray = None,
                                          pr_sem_room: np.ndarray = None, pr_door: np.ndarray = None,
                                          pr_stair: np.ndarray = None, t_wall: float = 0.40, close_k: int = 5,
                                          dilate_k: int = 3, do_plug: bool = True, plug_border_band: int = 24,
                                          plug_support_thr: float = 0.22, plug_close_k: int = 11,
                                          t_room: float = 0.45, t_sem_room: float = 0.50,
                                          min_component_area: int = 1200, border_margin: int = 12,
                                          dist_min: int = 18) -> dict:
    wall01 = build_wall_obstacles(pr_wall=pr_wall, t_wall=t_wall, close_k=close_k, dilate_k=dilate_k,
                                  do_plug=do_plug, plug_border_band=plug_border_band,
                                  plug_support_thr=plug_support_thr, plug_close_k=plug_close_k)
    interior01, outside01, free01 = make_interior_by_floodfill(wall01)
    interior_ref = vote_fix_interior(interior01=interior01, outside01=outside01, pr_room=pr_room,
                                     pr_sem_room=pr_sem_room, pr_door=pr_door, pr_stair=pr_stair,
                                     t_room=t_room, t_sem_room=t_sem_room, min_component_area=min_component_area,
                                     border_margin=border_margin, use_distance_guard=True, dist_min=dist_min)
    return {"wall01": wall01, "free01": free01, "outside01": outside01, "interior01": interior01,
            "interior_refined01": interior_ref}
def kmeans_labels_spatial(X_f32, K, xy_f32=None, xy_weight=1.5, attempts=5):
    """
    Spatial K-Means to prevent 'confetti' shattering.
    """
    X = X_f32.astype(np.float32)
    if xy_f32 is not None:
        # Weighted coordinate injection
        X = np.concatenate([X, xy_weight * xy_f32], axis=1)
    
    # Standardize
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    _, labels, _ = cv2.kmeans(Xn, int(K), None, criteria, int(attempts), cv2.KMEANS_PP_CENTERS)
    return labels.reshape(-1).astype(np.int32)


def refine_doors_using_embeddings(
    door_bin01: np.ndarray, 
    emb_map: np.ndarray, 
    min_cc_area: int = 120, 
    seam_dilate: int = 3,
    # Conservative threshold to avoid splitting a single door component.
    split_dist_thr: float = 1.2  
) -> np.ndarray:
    """
    Splits merged door blobs using embeddings.
    Conservative settings: Only splits if embeddings strongly disagree.
    """
    door = (door_bin01 > 0).astype(np.uint8)
    if door.sum() == 0: return door
    
    out = door.copy()
    nC, lab, stats, _ = cv2.connectedComponentsWithStats(door, 8)
    
    for cid in range(1, nC):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < int(min_cc_area): continue
        
        # Get bounding box
        y0 = stats[cid, cv2.CC_STAT_TOP]
        x0 = stats[cid, cv2.CC_STAT_LEFT]
        h  = stats[cid, cv2.CC_STAT_HEIGHT]
        w  = stats[cid, cv2.CC_STAT_WIDTH]
        y1, x1 = y0 + h, x0 + w
        
        # Crop
        blob_mask = (lab[y0:y1, x0:x1] == cid).astype(np.uint8)
        emb_crop  = emb_map[:, y0:y1, x0:x1] 
        
        ys, xs = np.where(blob_mask > 0)
        if ys.size < 50: continue 
        
        # 1. Embeddings [N, C]
        features = emb_crop[:, ys, xs].T 
        
        # 2. Coordinates [N, 2]
        coords = np.stack([ys / h, xs / w], axis=1).astype(np.float32)
        
        # Keep spatial weight low so long components are not split only by distance.
        labels = kmeans_labels_spatial(features, K=2, xy_f32=coords, xy_weight=0.1)
        
        mask0 = (labels == 0)
        mask1 = (labels == 1)
        
        if mask0.sum() == 0 or mask1.sum() == 0: continue

        center0 = features[mask0].mean(axis=0)
        center1 = features[mask1].mean(axis=0)
        
        # Euclidean distance in embedding space
        dist = np.linalg.norm(center0 - center1)
        
        # Rejection: If embeddings are too similar, DO NOT SPLIT.
        if dist < split_dist_thr:
            continue 
            
        # If we passed the check, execute the split
        m1 = np.zeros_like(blob_mask)
        m2 = np.zeros_like(blob_mask)
        m1[ys[mask0], xs[mask0]] = 1
        m2[ys[mask1], xs[mask1]] = 1
        
        k3 = np.ones((3,3), np.uint8)
        m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, k3)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k3)
        
        if m1.sum() < (area * 0.10) or m2.sum() < (area * 0.10):
            continue 
            
        overlap = cv2.dilate(m1, k3) & cv2.dilate(m2, k3)
        seam = cv2.dilate(overlap, np.ones((seam_dilate, seam_dilate), np.uint8))
        
        crop_out = out[y0:y1, x0:x1]
        crop_out[seam > 0] = 0
        out[y0:y1, x0:x1] = crop_out

    return out

# ---------------- Sliding Window ----------------

class SlidingWindowInferencer:
    """
    Sliding-window inference for the 20-channel fine-tuned model.
    """
    def __init__(self, model, crop_size=768, stride=512, batch_size=2):
        self.model = model
        self.crop_size = int(crop_size)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.device = next(model.parameters()).device

    @staticmethod
    def _split_probs_20(logits: torch.Tensor) -> torch.Tensor:
        # Unpack ALL 8 heads
        room2, blob2, wall2, center2, cut2, endpt1, sem5, inst4 = split_heads(logits)

        pr_room   = torch.softmax(room2,   dim=1)  # [B,2,H,W]
        pr_blob   = torch.softmax(blob2,   dim=1)  # [B,2,H,W]
        pr_wall   = torch.softmax(wall2,   dim=1)  # [B,2,H,W]
        pr_center = torch.softmax(center2, dim=1)  # [B,2,H,W]
        pr_cut    = torch.softmax(cut2,    dim=1)  # [B,2,H,W]
        
        # single-channel probability
        pr_endpt  = torch.sigmoid(endpt1)          # [B,1,H,W]
        
        pr_sem    = torch.softmax(sem5,    dim=1)  # [B,5,H,W]
        
        # Raw embeddings (no activation needed for distance calculations)
        inst_emb  = inst4                          # [B,4,H,W]

        return torch.cat([pr_room, pr_blob, pr_wall, pr_center, pr_cut, pr_endpt, pr_sem, inst_emb], dim=1)

    def predict(self, bgr_full_res: np.ndarray) -> np.ndarray:
        H, W = bgr_full_res.shape[:2]
        # Canvas must hold 20 channels now
        canvas = torch.zeros((20, H, W), device=self.device, dtype=torch.float32)
        counts = torch.zeros((1,  H, W), device=self.device, dtype=torch.float32)

        y_steps = list(range(0, H - self.crop_size + 1, self.stride)) or [0]
        if y_steps[-1] != H - self.crop_size:
            y_steps.append(H - self.crop_size)

        x_steps = list(range(0, W - self.crop_size + 1, self.stride)) or [0]
        if x_steps[-1] != W - self.crop_size:
            x_steps.append(W - self.crop_size)

        coords = [(y, x) for y in y_steps for x in x_steps]

        for i in range(0, len(coords), self.batch_size):
            batch_coords = coords[i:i + self.batch_size]
            batch_imgs = []

            for y, x in batch_coords:
                crop = bgr_full_res[y:y + self.crop_size, x:x + self.crop_size]
                ts = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
                batch_imgs.append(ts)

            x_batch = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                logits = self.model(x_batch)          # [B,20,h,w]
                probs  = self._split_probs_20(logits) # [B,20,h,w]

            for j, (y, x) in enumerate(batch_coords):
                canvas[:, y:y + self.crop_size, x:x + self.crop_size] += probs[j]
                counts[:, y:y + self.crop_size, x:x + self.crop_size] += 1.0

        final = canvas / torch.clamp(counts, min=1.0)
        return final.cpu().numpy()


def split_heads(logits: torch.Tensor):
    """
    logits: [B,20,H,W]
    Layout: room2(2), blob2(2), wall2(2), center2(2), cut2(2), endpt1(1), sem5(5), inst4(4)
    """
    return (
        logits[:, 0:2],    # room2
        logits[:, 2:4],    # blob2
        logits[:, 4:6],    # wall2
        logits[:, 6:8],    # center2
        logits[:, 8:10],   # cut2
        logits[:, 10:11],  # endpt1
        logits[:, 11:16],  # sem5
        logits[:, 16:20],
    )

def extract_rescue_v6_probs(probs20: np.ndarray) -> dict:
    """
    probs20: [20,H,W]
    """
    if not (probs20.ndim == 3 and probs20.shape[0] == 20):
        raise ValueError(f"Expected [20,H,W], got {probs20.shape}")

    return {
        "pr_room":   probs20[1],
        "pr_blob":   probs20[3],
        "pr_wall":   probs20[5],
        "pr_center": probs20[7],
        "pr_cut":    probs20[9],
        "pr_endpt":  probs20[10],
        "pr_sem":    probs20[11:16],
        "inst_emb":  probs20[16:20], # Keep raw embeddings
    }


# ---------------- Main Execution ----------------

def run_on_image(model, bgr, stem, long_side=1024, out_dir="pred_labels"):
    os.makedirs(out_dir, exist_ok=True)
    H_orig, W_orig = bgr.shape[:2]
    
    # Preserve a minimum short side so wide images are not over-compressed.
    target_short = 768
    current_short = min(H_orig, W_orig)
    
    if current_short < target_short:
        # Upsample small images
        scale = target_short / current_short
    else:
        # For huge images, we can shrink, but don't go below 768 on the short side
        # Check what scale 'long_side' would give
        scale_long = long_side / max(H_orig, W_orig)
        new_short = min(H_orig, W_orig) * scale_long
        
        if new_short < 720: # Allow a little slack, but not much (484 is too low)
            print(f"[Warn] Boosting resolution! Standard scaling would crush height to {int(new_short)}.")
            scale = 768 / min(H_orig, W_orig)
        else:
            scale = scale_long

    new_W = int(round(W_orig * scale))
    new_H = int(round(H_orig * scale))
    bgr_scaled = cv2.resize(bgr, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    
    print(f"[{os.path.basename(stem)}] Input: {W_orig}x{H_orig} -> Inference: {new_W}x{new_H} (Scale: {scale:.3f})")
    # -----------------------------------------

    t0 = time.time()
    inferencer = SlidingWindowInferencer(model, crop_size=768, stride=512, batch_size=2)
    
    ph = (768 - (new_H % 768)) % 768
    pw = (768 - (new_W % 768)) % 768
    bgr_pad = cv2.copyMakeBorder(bgr_scaled, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    probs_pad = inferencer.predict(bgr_pad)
    probs = probs_pad[:, :new_H, :new_W]
    print(f"[debug] Inference done ({time.time() - t0:.1f}s)")
    stem_base = os.path.join(out_dir, os.path.basename(stem))

    P = extract_rescue_v6_probs(probs)
    pr_room   = P["pr_room"]
    pr_blob   = P["pr_blob"]
    pr_wall   = P["pr_wall"]
    pr_center = P["pr_center"]
    pr_cut    = P["pr_cut"]
    pr_endpt  = P["pr_endpt"]
    pr_sem    = P["pr_sem"]
    inst_emb  = P["inst_emb"]

    pr_stair = pr_sem[4]

    # Walls & Interior
    pack = interior_pipeline_floodfill_plus_vote(
        pr_wall=pr_wall, pr_room=None, pr_sem_room=None, pr_door=pr_blob, pr_stair=pr_stair,
        t_wall=0.40, close_k=5, dilate_k=3, do_plug=True, plug_border_band=24,
        plug_support_thr=0.22, plug_close_k=11, t_room=0.45, t_sem_room=0.50,
        min_component_area=1200, border_margin=12, dist_min=18
    )
    bin_walls = pack["wall01"].astype(np.uint8)
    free01 = pack["free01"].astype(np.uint8)
    outside01 = pack["outside01"].astype(np.uint8)
    interior = pack["interior_refined01"].astype(np.uint8)

    # Door detection with conservative post-processing.
    bin_doors = door_mask_from_prob(pr_blob, thr=0.25, blur_ks=3, close_k=5)

    # 1) CUT (Center protected)
    # Re-enabled: At high resolution, the cut head is often accurate enough to help
    bin_doors = apply_cut_safely(bin_doors, pr_cut, pr_center, cut_thr=0.85, door_band_ks=31, protect_center_thr=0.50)

    # 2) EMBEDDING SPLIT
    # With higher resolution, the embeddings for "top door" and "side door" will be far apart
    bin_doors = refine_doors_using_embeddings(
        bin_doors, 
        inst_emb, 
        min_cc_area=120, 
        seam_dilate=3,
        split_dist_thr=0.65 
    )

    # 3) Endpoint Clean-up
    bin_doors = split_doors_by_endpoints(
        door01=bin_doors,
        pr_endpt=pr_endpt,
        endpt_thr=0.60,
        endpt_nms_r=6,
        endpt_band_ks=21,
        min_comp_area=60,
        cut_r=2,
        min_piece_area=45,
        max_passes=1
    ).astype(np.uint8)

    # Wall Constrain & Save
    wall_band = cv2.dilate(bin_walls, np.ones((5, 5), np.uint8), iterations=1)
    bin_stairs = (pr_stair > 0.50).astype(np.uint8)

    _save_bundle(stem_base, "wall_labels",  bin_walls, pr_wall)
    _save_bundle(stem_base, "door_labels",  bin_doors, pr_blob)
    _save_bundle(stem_base, "stair_labels", bin_stairs, pr_stair)
    cv2.imwrite(f"{stem_base}_interior.png", (interior * 255).astype(np.uint8))
    cv2.imwrite(f"{stem_base}_room_labels.png", (interior * 255).astype(np.uint8))
    cv2.imwrite(f"{stem_base}_outside_dbg.png", (outside01 * 255).astype(np.uint8))
    cv2.imwrite(f"{stem_base}_free_dbg.png",    (free01    * 255).astype(np.uint8))

    # Overlay
    vis = bgr_scaled.copy()
    vis[bin_walls > 0] = (0, 0, 255)
    vis[bin_doors > 0] = (0, 255, 0)
    vis[bin_stairs > 0] = (0, 165, 255)
    vis[pr_cut > 0.85] = (255, 0, 255)
    mask = interior > 0
    if mask.any():
        cyan = np.array([255, 255, 0], dtype=np.float32)
        roi = vis[mask].astype(np.float32)
        vis[mask] = (roi * 0.6 + cyan * 0.4).astype(np.uint8)
    cv2.imwrite(f"{stem_base}_overlay.jpg", vis)
    print(f"[{stem_base}] Saved overlay and masks.")

def build_argparser():
    ap = argparse.ArgumentParser("Fine-tuned CubiCasa evaluator")
    ap.add_argument("--mode", type=str, required=True, choices=["single", "photo"])
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--log-path", type=str, default="./eval_results")
    ap.add_argument("--long_side", type=int, default=1024, help="Inference resolution")
    ap.add_argument("--photo_path", type=str)
    ap.add_argument("--photo_root", type=str)
    return ap

def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    model = load_model_safe("hg_furukawa_original", args.weights, device)
    run_root = Path(normpath_abs(args.log_path)) / stamp() / "pred_labels"
    mkdir(str(run_root))

    if args.mode == "single":
        img = imread_safe(args.photo_path)
        if img is None: raise FileNotFoundError(f"Could not read image: {args.photo_path}")
        p = Path(args.photo_path)
        stem = p.parent.name if p.name == "photo_rect.png" else p.stem
        run_on_image(model, img, str(run_root / stem), args.long_side, str(run_root))
    elif args.mode == "photo":
        pairs = sorted(glob.glob(os.path.join(normpath_abs(args.photo_root), "*", "photo_rect.png")))
        if not pairs: raise FileNotFoundError(f"No photo_rect.png found under: {args.photo_root}")
        for p in pairs:
            img = imread_safe(p)
            if img is None:
                print(f"[warn] Could not read: {p}")
                continue
            stem = Path(p).parent.name
            run_on_image(model, img, str(run_root / stem), args.long_side, str(run_root))

if __name__ == "__main__":
    main()
