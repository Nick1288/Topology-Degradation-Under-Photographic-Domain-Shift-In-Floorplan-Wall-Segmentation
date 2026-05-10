#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a rectified + content-cropped photo benchmark from in-the-wild shots.
Additionally, derive filled FLOORS (rooms area) from either GT room masks or predicted walls.

For each <ID>.* photo:
  1) Detect & warp the page to top-down (HSV/Lab page finder, with edge-quad fallback),
  2) Run inner content crop (projection crop + CLAHE/adaptive fallback),
  3) If GT exists under data_root/<subset>/<ID>/F1_scaled_{room,icon}.png:
       - resize GT to original photo size,
       - warp with the same homography,
       - apply the exact same crop box,
  4) Optionally, if --walls-root is provided and a file <walls_root>/<ID>_walls.png exists:
       - warp & crop walls with the same homography/box,
       - thicken/close walls, remove page background, fill interiors → floors
  5) Save to out_root/<ID>/:
       - photo_rect.png
       - photo_room.png / photo_icon.png (if GT available)
       - floors.png (from GT rooms if present; otherwise from walls if supplied)
       - debug.jpg  (orig | rectified | final-crop)
"""

import os, re
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2

# ---------- configuration ----------
SUBSETS = ["high_quality", "high_quality_architectural", "colorful"]
EXTS    = ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG","*.heic","*.HEIC")

# ---------- utils ----------
def _mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _imread_color(p: Path): return cv2.imread(str(p), cv2.IMREAD_COLOR)
def _imread_u16(p: Path):   return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

def _extract_id_from_name(stem: str) -> Optional[str]:
    m = re.findall(r"\d+", stem)
    return (sorted(m, key=len)[-1].lstrip("0") or "0") if m else None

def _find_gt_paths(data_root: Path, id_: str):
    subset = next((s for s in SUBSETS if (data_root / s / id_).is_dir()), None)
    if subset is None: return None, None
    base = data_root / subset / id_
    room = base / "F1_scaled_room.png"
    icon = base / "F1_scaled_icon.png"
    return (room if room.is_file() else None, icon if icon.is_file() else None)

def _order_pts(pts4: np.ndarray) -> np.ndarray:
    s = pts4.sum(axis=1); d = np.diff(pts4, axis=1).ravel()
    tl = pts4[np.argmin(s)]; br = pts4[np.argmax(s)]
    tr = pts4[np.argmin(d)]; bl = pts4[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _save_triptych(orig, rect, cropped, out_path: Path):
    def to3(im):
        return im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h = max(orig.shape[0], rect.shape[0], cropped.shape[0])
    def pad_h(im):
        im = to3(im)
        if im.shape[0] == h: return im
        pad = np.zeros((h - im.shape[0], im.shape[1], 3), np.uint8)
        return np.vstack([im, pad])
    vis = np.hstack([pad_h(orig), pad_h(rect), pad_h(cropped)])
    cv2.imwrite(str(out_path), vis)

def _page_mask_from_rect_path(rect_path: Path) -> np.ndarray:
    bgr = _imread_color(rect_path)
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    page = (((hsv[...,2] > 170) | (lab[...,0] > 180)) & (hsv[...,1] < 80)).astype(np.uint8)*255
    page = cv2.medianBlur(page, 5)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), 2)
    return page
def emit_floors_for_id(id_: str, rect_png: Path, walls_root: Path, out_dir: Path):
    walls_path = walls_root / f"{id_}_walls.png"
    if not walls_path.is_file():
        print(f"[SKIP] no walls for {id_}: {walls_path}"); return
    page = _page_mask_from_rect_path(rect_png)
    walls = cv2.imread(str(walls_path), cv2.IMREAD_GRAYSCALE)
    if walls is None:
        print(f"[SKIP] unreadable walls: {walls_path}"); return

    # strengthen & close walls just a bit
    walls = cv2.medianBlur(walls, 3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k, 1)
    walls = cv2.dilate(walls, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)

    # floors = page minus walls, then flood-fill outside
    page_in = (page > 0).astype(np.uint8)*255
    free = cv2.bitwise_and(page_in, cv2.bitwise_not(walls))
    h,w = free.shape
    pad = cv2.copyMakeBorder(free, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((h+2,w+2), np.uint8)
    cv2.floodFill(pad, mask, (0,0), 255)
    outside = pad[1:-1,1:-1]
    floors = cv2.bitwise_and(free, cv2.bitwise_not(outside))
    floors = cv2.morphologyEx(floors, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)

    cv2.imwrite(str(out_dir / "floors.png"), floors, [cv2.IMWRITE_PNG_COMPRESSION, 1])


# ---------- robust rectifier (page first, then fallback to edge-quad) ----------
def _quad_is_ok(quad: np.ndarray, H0: int, W0: int) -> bool:
    area = cv2.contourArea(quad.astype(np.float32))
    if area < 0.25 * (H0 * W0):
        return False
    pts = _order_pts(quad.astype(np.float32))
    tl,tr,br,bl = pts
    sides = [np.linalg.norm(tr - tl), np.linalg.norm(br - tr),
             np.linalg.norm(bl - br), np.linalg.norm(tl - bl)]
    if any(s < 0.20 * max(H0, W0) for s in sides):
        return False
    def ang(a, b, c):
        v1, v2 = a-b, c-b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        cos = np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)
        return float(np.degrees(np.arccos(cos)))
    angs = [ang(tl,tr,br), ang(tr,br,bl), ang(br,bl,tl), ang(bl,tl,tr)]
    return min(angs) >= 10.0

def _rectify_edge_quad(bgr: np.ndarray):
    H0, W0 = bgr.shape[:2]
    scale = 1000.0 / max(H0, W0)
    small = cv2.resize(bgr, (int(W0*scale), int(H0*scale))) if scale < 1.0 else bgr.copy()
    if scale >= 1.0: scale = 1.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    med = np.median(gray)
    tries = [(int(0.66*med), int(1.33*med)), (30,120), (10,80), (50,180)]
    best_quad, best_area = None, -1.0
    for lo, hi in tries:
        e = cv2.Canny(gray, lo, hi, L2gradient=True)
        e = cv2.dilate(e, np.ones((3,3), np.uint8), 1)
        cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx): continue
            quad_full = (approx.reshape(-1,2).astype(np.float32) / scale)
            if not _quad_is_ok(quad_full, H0, W0): continue
            area = cv2.contourArea(quad_full.astype(np.float32))
            if area > best_area:
                best_area, best_quad = area, quad_full
        if best_quad is not None: break
    if best_quad is None: return None
    tl,tr,br,bl = _order_pts(best_quad)
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    Wd, Hd = int(max(wA,wB)), int(max(hA,hB))
    Wd = max(800, min(Wd, 6000)); Hd = max(800, min(Hd, 6000))
    dst = np.array([[0,0],[Wd-1,0],[Wd-1,Hd-1],[0,Hd-1]], np.float32)
    Hmat = cv2.getPerspectiveTransform(_order_pts(best_quad), dst)
    warped = cv2.warpPerspective(bgr, Hmat, (Wd, Hd), flags=cv2.INTER_CUBIC)
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped, Hmat

def _rectify_page_hsv_lab(bgr: np.ndarray, inflate=1.06, min_fill=0.30):
    H0, W0 = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) 
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    v = hsv[..., 2]; s = hsv[..., 1]; L = lab[..., 0]
    bright = (v > 170) | (L > 180) # v corresponds to the brightness, we are finding a white piece of paper, so it has to be bright, use lab, L to make doubly sure it is a bright paper
    low_sat = (s < 80) # s corresponds to how rich the color is, white has a low s, so we find that
    mask = (bright & low_sat).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), theta = rect
    if rw < 20 or rh < 20: return None
    rw, rh = float(rw)*inflate, float(rh)*inflate
    box = cv2.boxPoints(((cx, cy), (rw, rh), theta)).astype(np.float32)
    src = _order_pts(box)
    w = int(round(np.linalg.norm(src[1] - src[0]))); w = max(w, 2)
    h = int(round(np.linalg.norm(src[3] - src[0]))); h = max(h, 2)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, Hm, (w, h), flags=cv2.INTER_CUBIC)
    if float(w*h) < min_fill * float(H0*W0): return None
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped, Hm

def rectify_photo(bgr: np.ndarray):
    r = _rectify_page_hsv_lab(bgr)
    if r is not None: return r
    r = _rectify_edge_quad(bgr)
    if r is not None: return r
    return bgr, np.eye(3, dtype=np.float32)

# ---------- inner content crop ----------
def crop_plan_from_rectified(bgr: np.ndarray,
                             pad: int = 18,
                             frac: float = 0.0025,
                             page_margin_frac: float = 0.02,
                             min_page_keep: float = 0.55):
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    V, S, L = hsv[..., 2], hsv[..., 1], lab[..., 0]

    page = ((V > 170) | (L > 180)) & (S < 80)
    page = page.astype(np.uint8) * 255
    page = cv2.medianBlur(page, 5)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 2)

    cnts, _ = cv2.findContours(page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if (w * h) >= (min_page_keep * H * W):
            m = int(round(page_margin_frac * min(H, W)))
            x1 = max(0, x - m); y1 = max(0, y - m)
            x2 = min(W - 1, x + w - 1 + m)
            y2 = min(H - 1, y + h - 1 + m)
            inset = max(2, m // 6)
            x1 = min(x1 + inset, W - 2); y1 = min(y1 + inset, H - 2)
            x2 = max(x2 - inset, x1 + 1); y2 = max(y2 - inset, y1 + 1)
            return bgr[y1:y2 + 1, x1:x2 + 1], (x1, y1, x2, y2)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(gray, 7, 40, 40)
    v = np.median(g)
    lo = int(max(0, (1.0 - 0.33) * v)); hi = int(min(255, (1.0 + 0.33) * v))
    e = cv2.Canny(g, lo, hi, L2gradient=True)

    rows = e.sum(axis=1) // 255
    cols = e.sum(axis=0) // 255
    r_thr = max(4, int(frac * W))
    c_thr = max(4, int(frac * H))

    def bounds(arr, thr):
        idx = np.where(arr > thr)[0]
        return (0, len(arr) - 1) if idx.size == 0 else (int(idx[0]), int(idx[-1]))

    y1, y2 = bounds(rows, r_thr)
    x1, x2 = bounds(cols, c_thr)
    big_pad = max(pad, int(0.02 * min(H, W)))
    x1 = max(0, x1 - big_pad); y1 = max(0, y1 - big_pad)
    x2 = min(W - 1, x2 + big_pad); y2 = min(H - 1, y2 + big_pad)

    if (x2 - x1) < 0.45 * W or (y2 - y1) < 0.45 * H:
        return bgr, (0, 0, W - 1, H - 1)

    return bgr[y1:y2 + 1, x1:x2 + 1], (x1, y1, x2, y2)

def _warp_label_u16(label_u16: np.ndarray, Hmat: np.ndarray, out_size: Tuple[int,int]) -> np.ndarray:
    return cv2.warpPerspective(label_u16, Hmat, out_size, flags=cv2.INTER_NEAREST).astype(np.uint16)

# ---------- floors-from-walls helpers ----------
def _estimate_wall_thickness(walls_u8: np.ndarray, default_t: int = 6) -> int:
    w = (walls_u8 > 0).astype(np.uint8) * 255
    A = int(cv2.countNonZero(w))
    if A < 150: return default_t
    skel = np.zeros_like(w)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    tmp = w.copy()
    while True:
        eroded = cv2.erode(tmp, elem)
        dil = cv2.dilate(eroded, elem)
        sub = cv2.subtract(tmp, dil)
        skel = cv2.bitwise_or(skel, sub)
        tmp = eroded
        if cv2.countNonZero(tmp) == 0:
            break
    L = max(1, cv2.countNonZero(skel))
    t = int(np.clip(round(A / (2.0 * L)), 2, 18))
    return t

def _close_and_thicken(walls_u8: np.ndarray, t: int) -> np.ndarray:
    t = max(2, int(t))
    kL = int(np.clip(4 * t, 7, 31))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kL, max(1, t // 2)))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, t // 2), kL))
    out = cv2.morphologyEx(walls_u8, cv2.MORPH_CLOSE, kh, 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kv, 1)
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, t//2), max(3, t//2))), 1)
    return out

def _page_mask_from_rect(rect_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2Lab)
    V, S, L = hsv[...,2], hsv[...,1], lab[...,0]
    page = (((V > 170) | (L > 180)) & (S < 80)).astype(np.uint8) * 255
    page = cv2.medianBlur(page, 5)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), 2)
    if cv2.countNonZero(page) < 0.2 * page.size:
        page[:] = 255  # fallback: assume full canvas is page
    return page

def _floors_from_walls(rect_bgr: np.ndarray, walls_u8: np.ndarray) -> np.ndarray:
    """
    Convert outline/band walls to filled floors:
      1) page mask,
      2) thicken/close walls,
      3) remove walls from page, flood-fill from border to get background,
      4) floors = page_without_background.
    """
    H, W = rect_bgr.shape[:2]
    page = _page_mask_from_rect(rect_bgr)
    walls = (walls_u8 > 0).astype(np.uint8) * 255
    if walls.shape[:2] != (H, W):
        walls = cv2.resize(walls, (W, H), interpolation=cv2.INTER_NEAREST)

    t = _estimate_wall_thickness(walls)
    walls_solid = _close_and_thicken(walls, t)

    interior = cv2.bitwise_and(page, cv2.bitwise_not(walls_solid))

    # flood-fill from border to remove exterior/background
    ff = interior.copy()
    h, w = ff.shape[:2]
    pad = cv2.copyMakeBorder(ff, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(pad, mask, (0,0), 128)  # fill outside with 128
    outside = (pad[1:-1,1:-1] == 128).astype(np.uint8) * 255

    floors = cv2.bitwise_and(interior, cv2.bitwise_not(outside))
    floors = cv2.morphologyEx(floors, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    floors = cv2.medianBlur(floors, 3)
    return floors

# ---------- per-photo pipeline ----------
def process_one(photo_path: Path,
                data_root: Path,
                out_root: Path,
                crop_pad: int,
                crop_frac: float,
                walls_root: Optional[Path]):
    bgr = _imread_color(photo_path)
    if bgr is None:
        print(f"[SKIP] unreadable: {photo_path}")
        return
    id_ = _extract_id_from_name(photo_path.stem)
    if id_ is None:
        print(f"[SKIP] no numeric ID in {photo_path.name}")
        return
    out_dir = out_root / id_; _mkdir(out_dir)

    # 1) rectify
    rect, H = rectify_photo(bgr) # de-warp the photo
    Wd, Hd = rect.shape[1], rect.shape[0]

    room_w = icon_w = None
    room_gt, icon_gt = _find_gt_paths(data_root, id_)
    if room_gt is not None and icon_gt is not None:
        room = _imread_u16(room_gt); icon = _imread_u16(icon_gt)
        if room is not None and icon is not None:
            Ph, Pw = bgr.shape[:2]
            room_rs = cv2.resize(room, (Pw, Ph), interpolation=cv2.INTER_NEAREST)
            icon_rs = cv2.resize(icon, (Pw, Ph), interpolation=cv2.INTER_NEAREST)
            room_w = _warp_label_u16(room_rs, H, (Wd, Hd))
            icon_w = _warp_label_u16(icon_rs, H, (Wd, Hd))

    # 3) content crop and apply same box to labels
    rect_c, (x1,y1,x2,y2) = crop_plan_from_rectified(rect, pad=crop_pad, frac=crop_frac)
    if room_w is not None: room_w = room_w[y1:y2+1, x1:x2+1]
    if icon_w is not None: icon_w = icon_w[y1:y2+1, x1:x2+1]

    # 3b) if walls provided, bring them through the same warp+crop so floors can be derived
    walls_c = None
    if walls_root is not None:
        # try <walls_root>/<ID>_walls.png first, then <walls_root>/<ID>/walls.png
        cand1 = walls_root / f"{id_}_walls.png"
        cand2 = walls_root / id_ / "walls.png"
        wsrc = cand1 if cand1.is_file() else (cand2 if cand2.is_file() else None)
        if wsrc is not None:
            walls0 = cv2.imread(str(wsrc), cv2.IMREAD_GRAYSCALE)
            if walls0 is not None:
                # warp+crop walls to match rect_c
                walls_warp = cv2.warpPerspective(walls0, H, (Wd, Hd), flags=cv2.INTER_NEAREST)
                walls_c = walls_warp[y1:y2+1, x1:x2+1]

    # 4) save base outputs
    cv2.imwrite(str(out_dir / "photo_rect.png"), rect_c)
    if room_w is not None: cv2.imwrite(str(out_dir / "photo_room.png"), room_w)
    if icon_w is not None: cv2.imwrite(str(out_dir / "photo_icon.png"), icon_w)
    _save_triptych(bgr, rect, rect_c, out_dir / "debug.jpg")

    # 5) derive FLOORS
    floors = None
    # Preferred: derive directly from GT rooms if available (room foreground == 1)
    if room_w is not None:
        # GT rooms in CubiCasa finetune convention: 0 background, 1 room foreground
        floors = ((room_w == 1).astype(np.uint8) * 255)
    elif walls_c is not None:
        floors = _floors_from_walls(rect_c, walls_c)

    if floors is not None:
        cv2.imwrite(str(out_dir / "floors.png"), floors)

    print(f"[OK] {photo_path.name} -> {out_dir}  {'(floors)' if floors is not None else '(no floors)'}")

# ---------- CLI ----------
def main():
    import argparse
    ap = argparse.ArgumentParser("Build in-the-wild photo benchmark for CubiCasa5k (+ floors)")
    ap.add_argument("--photo-root", required=True)
    ap.add_argument("--data-root",  required=True)
    ap.add_argument("--out-root",   required=True)
    ap.add_argument("--crop-pad", type=int, default=18)
    ap.add_argument("--crop-frac", type=float, default=0.0025)
    ap.add_argument("--walls-root", type=str, default=None,
                    help="Folder with predicted walls (expects <ID>_walls.png or <ID>/walls.png). Optional.")
    args = ap.parse_args()

    photo_root = Path(args.photo_root)
    data_root  = Path(args.data_root)
    out_root   = Path(args.out_root)
    assert photo_root.is_dir(), f"Bad photo-root: {photo_root}"
    assert data_root.is_dir(),  f"Bad data-root: {data_root}"
    _mkdir(out_root)

    walls_root = Path(args.walls_root) if args.walls_root else None
    if walls_root is not None and not walls_root.exists():
        print(f"[WARN] walls-root not found: {walls_root}")
        walls_root = None

    # ---- CASE 1: already-rectified (<root>/<ID>/photo_rect.png) ----
    rects = sorted(photo_root.glob("*/photo_rect.png"))
    if rects:
        print(f"[INFO] Found {len(rects)} rectified photos under {photo_root}, using rectified-mode.")
        for rect_path in rects:
            id_ = rect_path.parent.name
            out_dir = out_root / id_
            _mkdir(out_dir)
            rect = _imread_color(rect_path)
            if rect is None:
                print(f"[SKIP] unreadable: {rect_path}")
                continue

            rect_c, _box = crop_plan_from_rectified(rect, pad=args.crop_pad, frac=args.crop_frac)
            cv2.imwrite(str(out_dir / "photo_rect.png"), rect_c)

            if walls_root is not None:
                emit_floors_for_id(id_, out_dir / "photo_rect.png", walls_root, out_dir)

            _save_triptych(rect, rect, rect_c, out_dir / "debug.jpg")
            print(f"[OK] {id_} -> {out_dir}")
        return

    # Raw photos directly under photo-root.
    print(f"[INFO] No rectified photos found, falling back to RAW-mode from {photo_root}")
    exts = ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG","*.heic","*.HEIC")
    photos = []
    for pat in exts:
        photos.extend(sorted(photo_root.glob(pat)))

    if not photos:
        print(f"[WARN] No raw photos found in {photo_root}")
        return

    for p in photos:
        process_one(
            photo_path=p,
            data_root=data_root,
            out_root=out_root,
            crop_pad=args.crop_pad,
            crop_frac=args.crop_frac,
            walls_root=walls_root,
        )



if __name__ == "__main__":
    main()
