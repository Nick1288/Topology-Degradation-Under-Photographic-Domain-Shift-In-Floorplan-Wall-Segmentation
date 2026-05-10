"""
extract_svg_masks.py

Purpose
- For each cubi_ft split folder (e.g., 10_t / 10_t_r / 10_t_l), pick the correct floor
  from the corresponding high_quality_architectural/<base_id>/model.svg, then export masks.

What it exports per split folder (in OUT_ROOT/<split_folder>/):
- obstacle_label.png   : binary mask of walls and windows
- icon_label.png       : binary mask of doors (Door + Threshold)
- floor_pick.txt       : scores for each floor + chosen floor
- debug_obstacle_floorK.png, debug_doors_floorK.png (optional) for all floors

How it picks floor
- Rasterize per-floor "obstacle" layer and compare edges to the split folder's clean_rgb.png.
  Highest score wins.

Notes
- This assumes split folders already contain clean_rgb.png.
- IDs in SKIP_IDS are intentionally skipped.

Dependencies
pip install lxml cairosvg opencv-python numpy
Optional (fallback renderer):
- Inkscape on PATH (if CairoSVG fails on some SVGs)

Run
python extract_svg_masks.py
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import cv2
from lxml import etree
import cairosvg

# ==============================
# EDIT THESE PATHS
# ==============================
CUBI_FT_ROOT = Path(r"C:\Users\nicho\CubiCasa5k\data\cubi_ft_rasterized")
HQ_ROOT = Path(r"C:\Users\nicho\CubiCasa5k\data\cubicasa5k\high_quality_architectural")
OUT_ROOT = Path(r"C:\Users\nicho\CubiCasa5k\data\cubi_ft_masks")  # output mirror (new)
# ==============================

# Raster size for mask export.
RASTER_W = 1024
RASTER_H = 1024

# If True, keep per-floor debug rasters; else delete after exporting chosen masks
KEEP_DEBUG_FLOORS = True

SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}

# Extract numeric base id from folder name like "10_t", "10_t_r", "13565_anything"
ID_RE = re.compile(r"^(\d+)")


# Your "skip these ids, these are problematic"
SKIP_IDS = {
    "102",
    "1128",
    "10092",
    "10450",
    "10530",
    "10550",
    "10589",
    "10723",
    "10731",
    "10768",
    "11062",
    "11136",
    "11345",
    "13142",
    "13550",
    "13565",
    "13802",
    "14129",
}
# "11154R is invalid" will be skipped automatically because it doesn't match numeric prefix.


# -----------------------------------------
# SVG helpers
# -----------------------------------------
def _class_tokens(elem):
    return set((elem.get("class") or "").split())


def floor_index_from_class(g):
    # looks for tokens like "Floor-1"
    for t in _class_tokens(g):
        if t.startswith("Floor-"):
            try:
                return int(t.split("-")[1])
            except Exception:
                return None
    return None


def force_visible(elem):
    style = elem.get("style") or ""
    style = re.sub(r"display\s*:\s*none\s*;?", "", style, flags=re.I).strip()
    if style:
        elem.set("style", style)
    else:
        elem.attrib.pop("style", None)


def detect_floor_indices(svg_bytes: bytes) -> list[int]:
    parser = etree.XMLParser(remove_comments=True, recover=True, huge_tree=True)
    root = etree.fromstring(svg_bytes, parser=parser)

    floors = root.xpath(
        ".//svg:g[contains(concat(' ', normalize-space(@class), ' '), ' Floorplan ')]",
        namespaces=NSMAP,
    )
    idxs = sorted(
        {
            floor_index_from_class(g)
            for g in floors
            if floor_index_from_class(g) is not None
        }
    )
    return idxs

def subtract_binary_masks(obstacle_u8: np.ndarray, remove_u8: np.ndarray) -> np.ndarray:
    """
    obstacle_u8: uint8 mask, either {0,255} or any grayscale where >0 means obstacle
    remove_u8  : uint8 mask, either {0,255} or any grayscale where >0 means remove

    Returns uint8 {0,255} where remove pixels are cleared from obstacle.
    No dilation. Pure subtraction.
    """
    if obstacle_u8.ndim != 2 or remove_u8.ndim != 2:
        raise ValueError("Masks must be 2D grayscale")

    if obstacle_u8.shape != remove_u8.shape:
        raise ValueError(f"Shape mismatch: obstacle {obstacle_u8.shape} vs remove {remove_u8.shape}")

    obs = obstacle_u8 > 0
    rem = remove_u8 > 0

    out = (obs & (~rem)).astype(np.uint8) * 255
    return out


def extract_floor_and_layers_svg(
    svg_bytes: bytes,
    keep_floor_idx: int,
    match_mode: str,
) -> bytes:
    """
    match_mode:
      - "walls"
      - "windows"
      - "doors"

    Keeps only the chosen floor and only the matched geometry.
    IMPORTANT: if a matching element is a group <g>, we keep the entire subtree,
    otherwise you delete the child <path>/<rect> and the raster becomes blank.
    """

    parser = etree.XMLParser(remove_comments=True, recover=True, huge_tree=True)
    root = etree.fromstring(svg_bytes, parser=parser)

    # 1) keep only selected floor group
    floors = root.xpath(
        ".//svg:g[contains(concat(' ', normalize-space(@class), ' '), ' Floorplan ')]",
        namespaces=NSMAP,
    )

    selected_floor_g = None
    for g in floors:
        idx = floor_index_from_class(g)
        if idx is not None and idx == keep_floor_idx:
            selected_floor_g = g
        else:
            parent = g.getparent()
            if parent is not None:
                parent.remove(g)

    if selected_floor_g is None:
        selected_floor_g = root

    def cls_tokens(el):
        return set((el.get("class") or "").split())

    def any_ancestor(el, fn) -> bool:
        p = el.getparent()
        while p is not None:
            if fn(p):
                return True
            p = p.getparent()
        return False

    # ---- Match rules ----
    def is_wall_group(el) -> bool:
        # Wall elements can appear as "Wall" or "Wall External".
        toks = cls_tokens(el)
        if "Wall" in toks:
            return True
        # sometimes "Wall External" becomes two tokens, still covers by above
        # id might be "Wall" on some files
        if (el.get("id") or "") == "Wall":
            return True
        return False

    def is_window_group(el) -> bool:
        cls = (el.get("class") or "")
        eid = (el.get("id") or "")
        if "Window" in cls:
            return True
        if eid == "Window":
            return True
        return False

    def is_window(el) -> bool:
        return is_window_group(el) or any_ancestor(el, is_window_group)

    def is_door_group(el) -> bool:
        cls = (el.get("class") or "")
        eid = (el.get("id") or "")
        if "Door" in cls:
            return True
        if "Threshold" in cls:
            return True
        if eid in ("Door", "Threshold"):
            return True
        return False

    def is_door(el) -> bool:
        return is_door_group(el) or any_ancestor(el, is_door_group)

    if match_mode == "walls":
        match_fn = is_wall_group
    elif match_mode == "windows":
        match_fn = is_window
    elif match_mode == "doors":
        match_fn = is_door
    else:
        raise ValueError(f"Unknown match_mode: {match_mode}")

    # 3) Collect elements to keep: matched elements + their ancestors
    # PLUS: if matched element is a <g>, keep its whole subtree (descendants).
    to_keep = {root}

    drawables = selected_floor_g.xpath(
        ".//svg:path | .//svg:rect | .//svg:line | .//svg:polygon | .//svg:polyline | "
        ".//svg:circle | .//svg:ellipse | .//svg:text | .//svg:g",
        namespaces=NSMAP,
    )

    for el in drawables:
        if match_fn(el):
            # keep this element + ancestors
            cur = el
            while cur is not None:
                to_keep.add(cur)
                cur = cur.getparent()

            # CRITICAL: if it's a group, keep everything inside it
            if el.tag.endswith("}g"):
                for d in el.xpath(".//*", namespaces=NSMAP):
                    to_keep.add(d)

    # 4) Prune everything not in keep-set (post-order)
    for el in list(root.xpath(".//*", namespaces=NSMAP))[::-1]:
        if el not in to_keep:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    # 5) ensure visibility
    for el in list(to_keep):
        if isinstance(el.tag, str):
            force_visible(el)

    return etree.tostring(root, encoding="utf-8", xml_declaration=True)


# -----------------------------------------
# Rasterization with fallback
# -----------------------------------------
def _inkscape_available() -> bool:
    try:
        subprocess.run(
            ["inkscape", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


INKSCAPE_OK = _inkscape_available()

def read_png_gray(path: Path) -> np.ndarray:
    """
    Reads a PNG and returns a grayscale image with alpha composited onto white.
    Fixes the "looks fine in Explorer but OpenCV says all zeros" issue.
    """
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read {path}")

    # RGBA -> composite on white
    if im.ndim == 3 and im.shape[2] == 4:
        bgr = im[:, :, :3].astype(np.float32)
        a = (im[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        bgr_white = bgr * a + 255.0 * (1.0 - a)
        gray = cv2.cvtColor(bgr_white.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return gray

    # BGR -> gray
    if im.ndim == 3 and im.shape[2] == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # already gray
    return im


def svg2png_with_fallback(svg_bytes: bytes, out_png: Path, w: int, h: int) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) CairoSVG (force WHITE background so we don't get alpha-only strokes)
    try:
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=str(out_png),
            output_width=w,
            output_height=h,
            background_color="white",   # <<< CRITICAL
        )
        return "cairosvg"
    except Exception:
        pass

    # 2) Inkscape fallback (also force white background)
    if not INKSCAPE_OK:
        raise RuntimeError(
            "CairoSVG failed and Inkscape not found on PATH. Install Inkscape or add it to PATH."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
        tmp.write(svg_bytes)
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "inkscape",
            str(tmp_path),
            "--export-type=png",
            f"--export-filename={str(out_png)}",
            f"--export-width={w}",
            f"--export-height={h}",
            "--export-background=white",          # <<< CRITICAL
            "--export-background-opacity=1",      # <<< CRITICAL
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "inkscape"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass



# -----------------------------------------
# Mask post-processing
# -----------------------------------------
def raster_png_to_binary_mask(
    png_path: Path,
    out_mask_path: Path,
    bg_threshold: int = 245,
    morph_close: int = 3,
) -> None:
    """
    Produces a binary mask where dark strokes become 255 and white background becomes 0.
    bg_threshold: pixels > threshold are treated as background.
    """
    img = read_png_gray(png_path)
    if img is None:
        raise RuntimeError(f"Failed to read {png_path}")

    # background is near-white; foreground are darker strokes
    mask = np.where(img > bg_threshold, 0, 255).astype(np.uint8)

    if morph_close and morph_close > 0:
        k = np.ones((morph_close, morph_close), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cv2.imwrite(str(out_mask_path), mask)



def merge_binary_masks(mask_paths: list[Path], out_path: Path) -> None:
    acc = None
    for p in mask_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        if acc is None:
            acc = img.copy()
        else:
            acc = np.maximum(acc, img)
    if acc is None:
        raise RuntimeError("No masks to merge")
    cv2.imwrite(str(out_path), acc)


# -----------------------------------------
# Floor picking via edge match
# -----------------------------------------
def _read_gray(path: Path, max_side=1200):
    try:
        img = read_png_gray(path)
    except Exception:
        return None

    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _score_match_edges(a, b) -> float:
    e1 = cv2.Canny(a, 50, 150)
    e2 = cv2.Canny(b, 50, 150)

    e1 = cv2.resize(e1, (800, 800))
    e2 = cv2.resize(e2, (800, 800))

    e1 = e1.astype(np.float32)
    e2 = e2.astype(np.float32)

    e1 = (e1 - e1.mean()) / (e1.std() + 1e-6)
    e2 = (e2 - e2.mean()) / (e2.std() + 1e-6)

    return float((e1 * e2).mean())


def pick_best_floor_by_obstacle(
    ref_clean_rgb: Path, obstacle_floor_pngs: list[Path]
) -> tuple[int | None, list[tuple[str, float]]]:
    """
    Returns:
      best_floor_idx (int) or None
      scores list: [(filename, score), ...] sorted desc
    """
    ref = _read_gray(ref_clean_rgb)
    if ref is None:
        return None, []

    best_idx = None
    best_score = -1e18
    scores = []

    for p in obstacle_floor_pngs:
        cand = _read_gray(p)
        if cand is None:
            continue
        sc = _score_match_edges(ref, cand)
        scores.append((p.name, sc))
        if sc > best_score:
            best_score = sc
            m = re.search(r"floor(\d+)", p.name)
            best_idx = int(m.group(1)) if m else None

    scores.sort(key=lambda x: x[1], reverse=True)
    return best_idx, scores


# -----------------------------------------
# Utility: reporting / id parsing
# -----------------------------------------
def base_id_from_folder(folder_name: str) -> str | None:
    m = ID_RE.match(folder_name)
    return m.group(1) if m else None


def write_pick_report(out_dir: Path, split_folder: Path, bid: str, best_floor: int | None, scores):
    txt = out_dir / "floor_pick.txt"
    lines = []
    lines.append(f"split_folder={split_folder.name}")
    lines.append(f"base_id={bid}")
    lines.append(f"ref_clean_rgb={str((split_folder/'clean_rgb.png').resolve())}")
    lines.append(f"chosen_floor={best_floor if best_floor is not None else 'None'}")
    lines.append("scores:")
    for name, sc in scores:
        lines.append(f"  {name}\t{sc:.6f}")
    txt.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------
# Core: export masks for one split folder
# -----------------------------------------
def export_masks_for_split(split_folder: Path, svg_path: Path, out_dir: Path) -> bool:
    """
    Returns True if success, False otherwise.
    """
    ref_clean = split_folder / "clean_rgb.png"
    if not ref_clean.exists():
        print(f"[SKIP] {split_folder.name}: missing clean_rgb.png")
        return False

    svg_bytes = svg_path.read_bytes()
    floor_idxs = detect_floor_indices(svg_bytes)
    if not floor_idxs:
        floor_idxs = [1]

    # For matching floors, we rasterize an "obstacle layer" per floor:
    # Walls + windows = obstacle.
    obstacle_rasters = []
    doors_rasters = []

    for idx in floor_idxs:
        # Walls layer
        walls_svg = extract_floor_and_layers_svg(svg_bytes, keep_floor_idx=idx, match_mode="walls")
        walls_png = out_dir / f"debug_walls_floor{idx}.png"
        svg2png_with_fallback(walls_svg, walls_png, w=RASTER_W, h=RASTER_H)

        # Windows layer
        # Your SVG has "Window Regular" class and also "Glass"/"Panel" which may be part of window.
        # Start conservative: include Window + Glass + Panel.
        windows_svg = extract_floor_and_layers_svg(svg_bytes, keep_floor_idx=idx, match_mode="windows")
        windows_png = out_dir / f"debug_windows_floor{idx}.png"
        svg2png_with_fallback(windows_svg, windows_png, w=RASTER_W, h=RASTER_H)

        # Obstacle = max(walls, windows) in raster space (not binary yet)
        # We'll merge later after binarization, but for matching edges, we can merge the raw rasters too.
        # We'll make a merged debug obstacle raster now for scoring.
        wimg = read_png_gray(walls_png)
        ximg = read_png_gray(windows_png)
        if wimg is None or ximg is None:
            raise RuntimeError(f"Failed to read debug rasters for floor {idx}")

        # 1) binary masks (black bg, white fg)
        walls_b = (wimg < 245)
        wins_b  = (ximg < 245)
        obstacle_b = walls_b | wins_b

        # Debug view (optional)
        dbg = np.zeros_like(wimg, dtype=np.uint8)
        dbg[wins_b] = 180
        dbg[walls_b] = 255
        obstacle_png = out_dir / f"debug_obstacle_floor{idx}.png"
        cv2.imwrite(str(obstacle_png), dbg)

        # Save a true binary obstacle mask per floor for later selection/output
        bin_obstacle_png = out_dir / f"bin_obstacle_floor{idx}.png"
        cv2.imwrite(str(bin_obstacle_png), (obstacle_b.astype(np.uint8) * 255))

        obstacle_rasters.append(obstacle_png)  # use dbg for picking

        # Doors layer (Door + Threshold)
        doors_svg = extract_floor_and_layers_svg(svg_bytes, keep_floor_idx=idx, match_mode="doors")
        doors_png = out_dir / f"debug_doors_floor{idx}.png"
        svg2png_with_fallback(doors_svg, doors_png, w=RASTER_W, h=RASTER_H)
        doors_rasters.append(doors_png)

    # Pick best floor by obstacle match
    best_floor, scores = pick_best_floor_by_obstacle(ref_clean, obstacle_rasters)
    write_pick_report(out_dir, split_folder, base_id_from_folder(split_folder.name) or "?", best_floor, scores)

    if best_floor is None:
        print(f"[FAIL] {split_folder.name}: cannot pick best floor")
        return False

    # Export chosen masks
    chosen_doors = out_dir / f"debug_doors_floor{best_floor}.png"

    # Convert to binary masks
    # obstacle_label = walls+windows => already merged raster; just binarize it
    # Export chosen masks
    chosen_obstacle_bin = out_dir / f"bin_obstacle_floor{best_floor}.png"
    chosen_doors_png    = out_dir / f"debug_doors_floor{best_floor}.png"

    obstacle_u8 = cv2.imread(str(chosen_obstacle_bin), cv2.IMREAD_GRAYSCALE)
    if obstacle_u8 is None:
        raise RuntimeError(f"Failed to read {chosen_obstacle_bin}")

    # doors -> binary {0,255}
    dimg = read_png_gray(chosen_doors_png)
    doors_u8 = ((dimg < 245).astype(np.uint8) * 255)

    # OPTIONAL: save raw doors for inspection
    cv2.imwrite(str(out_dir / "icon_label.png"), doors_u8)

    # SUBTRACT DOORS FROM OBSTACLE (NO DILATION)
    obstacle_minus_doors = subtract_binary_masks(obstacle_u8, doors_u8)

    # OPTIONAL: a tiny close ONLY if you want to fill micro gaps in walls/windows
    # (does not re-add doors because subtraction already happened)
    k = np.ones((3, 3), np.uint8)
    obstacle_minus_doors = cv2.morphologyEx(obstacle_minus_doors, cv2.MORPH_CLOSE, k)

    cv2.imwrite(str(out_dir / "obstacle_label.png"), obstacle_minus_doors)

    # Debug: visualize what got removed (doors pixels that overlapped obstacle)
    removed = ((obstacle_u8 > 0) & (doors_u8 > 0)).astype(np.uint8) * 255
    cv2.imwrite(str(out_dir / "debug_removed_doors_from_obstacle.png"), removed)

    # Optionally clean debug files
    if not KEEP_DEBUG_FLOORS:
        for p in out_dir.glob("debug_*_floor*.png"):
            p.unlink(missing_ok=True)

    print(f"[OK] {split_folder.name}: picked Floor-{best_floor} -> obstacle_label.png + icon_label.png")
    return True


# -----------------------------------------
# Main
# -----------------------------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not INKSCAPE_OK:
        print("[WARN] Inkscape not detected. If CairoSVG fails on some SVGs, this script will stop.")

    split_folders = [p for p in CUBI_FT_ROOT.iterdir() if p.is_dir()]
    print(f"[INFO] Found {len(split_folders)} split folders in CUBI_FT_ROOT")

    ok = 0
    fail = 0
    skipped = 0
    missing_svg = 0
    skipped_bad = 0

    for split in split_folders:
        bid = base_id_from_folder(split.name)
        if bid is None:
            skipped += 1
            continue

        if bid in SKIP_IDS:
            skipped_bad += 1
            print(f"[SKIP] {split.name}: base id {bid} in SKIP_IDS list")
            continue

        svg_path = HQ_ROOT / bid / "model.svg"
        if not svg_path.exists():
            missing_svg += 1
            print(f"[SKIP] {split.name}: missing {svg_path}")
            continue

        out_dir = OUT_ROOT / split.name
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            success = export_masks_for_split(split, svg_path, out_dir)
            if success:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {split.name}: {type(e).__name__}: {e}")

    print("\n===== SUMMARY =====")
    print(f"OK: {ok}")
    print(f"FAIL: {fail}")
    print(f"SKIPPED (no numeric prefix): {skipped}")
    print(f"SKIPPED (problematic ids): {skipped_bad}")
    print(f"SKIPPED (missing SVG): {missing_svg}")
    print(f"OUT_ROOT: {OUT_ROOT}")


if __name__ == "__main__":
    main()
