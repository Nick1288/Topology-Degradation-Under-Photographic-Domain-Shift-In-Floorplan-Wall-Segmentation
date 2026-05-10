import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import cv2
from lxml import etree
import cairosvg

# ====== EDIT THESE PATHS ======
CUBI_FT_ROOT = Path(r"C:\Users\nicho\CubiCasa5k\data\cubi_ft")
HQ_ROOT = Path(r"C:\Users\nicho\CubiCasa5k\data\cubicasa5k\high_quality_architectural")
OUT_ROOT = Path(r"C:\Users\nicho\CubiCasa5k\data\cubi_ft_rasterized")  # output mirror
# ==============================

SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}

ID_RE = re.compile(r"^(\d+)")  # base id before underscore


# -----------------------------
# SVG helpers
# -----------------------------
def class_tokens(elem):
    return set((elem.get("class") or "").split())

def floor_index_from_class(g):
    for t in class_tokens(g):
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

def extract_floor_svg(svg_bytes: bytes, keep_floor_idx: int) -> bytes:
    parser = etree.XMLParser(remove_comments=True, recover=True, huge_tree=True)
    root = etree.fromstring(svg_bytes, parser=parser)

    floors = root.xpath(
        ".//svg:g[contains(concat(' ', normalize-space(@class), ' '), ' Floorplan ')]",
        namespaces=NSMAP,
    )
    floor_groups = []
    for g in floors:
        idx = floor_index_from_class(g)
        if idx is not None:
            floor_groups.append((idx, g))

    if not floor_groups:
        return etree.tostring(root, encoding="utf-8", xml_declaration=True)

    for idx, g in floor_groups:
        if idx != keep_floor_idx:
            parent = g.getparent()
            if parent is not None:
                parent.remove(g)
        else:
            force_visible(g)
            p = g.getparent()
            while p is not None:
                force_visible(p)
                p = p.getparent()

    return etree.tostring(root, encoding="utf-8", xml_declaration=True)

def detect_floor_indices(svg_bytes: bytes):
    parser = etree.XMLParser(remove_comments=True, recover=True, huge_tree=True)
    root = etree.fromstring(svg_bytes, parser=parser)
    floors = root.xpath(
        ".//svg:g[contains(concat(' ', normalize-space(@class), ' '), ' Floorplan ')]",
        namespaces=NSMAP,
    )
    idxs = sorted({floor_index_from_class(g) for g in floors if floor_index_from_class(g) is not None})
    return idxs


# -----------------------------
# Rasterization with fallback
# -----------------------------
def _inkscape_available() -> bool:
    try:
        subprocess.run(["inkscape", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

INKSCAPE_OK = _inkscape_available()

def svg2png_with_fallback(svg_bytes: bytes, out_png: Path, w: int, h: int) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) CairoSVG
    try:
        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(out_png), output_width=w, output_height=h)
        return "cairosvg"
    except Exception:
        pass

    # 2) Inkscape fallback (better SVG tolerance)
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
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "inkscape"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def rasterize_svg_all_floors(svg_path: Path, out_dir: Path, w: int, h: int):
    """
    Writes clean_rgb_floor{idx}.png for each floor detected.
    Returns list of output PNG Paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_bytes = svg_path.read_bytes()

    idxs = detect_floor_indices(svg_bytes)

    out_files = []

    if not idxs:
        out_png = out_dir / "clean_rgb_floor1.png"
        backend = svg2png_with_fallback(svg_bytes, out_png, w, h)
        out_files.append(out_png)
        return out_files

    for idx in idxs:
        floor_svg = extract_floor_svg(svg_bytes, keep_floor_idx=idx)
        out_png = out_dir / f"clean_rgb_floor{idx}.png"
        backend = svg2png_with_fallback(floor_svg, out_png, w, h)
        out_files.append(out_png)

    return out_files


# -----------------------------
# Floor picking (auto best match)
# -----------------------------
def _read_gray(path: Path, max_side=1200):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def _score_match_orb(a, b) -> float:
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        return -1.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d1, d2)
    if not m:
        return -1.0
    m = sorted(m, key=lambda x: x.distance)
    good = [x for x in m[:250] if x.distance < 64]
    return float(len(good))

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

def score_match(img_a, img_b) -> float:
    s_orb = _score_match_orb(img_a, img_b)
    if s_orb >= 0:
        return s_orb
    return _score_match_edges(img_a, img_b)

def find_reference_image(split_folder: Path) -> Path | None:
    """
    Picks the best available reference image in a cubi_ft split folder.
    Adjust names if split folders use different filenames.
    """
    preferred = [
        "f1_scaled.png",
        "f1_original.png",
        "f1.png",
        "scaled.png",
        "original.png",
    ]
    for name in preferred:
        p = split_folder / name
        if p.exists():
            return p

    # fallback: any png
    pngs = sorted(split_folder.glob("*.png"))
    return pngs[0] if pngs else None

def pick_best_floor(split_folder: Path, floor_pngs: list[Path]) -> tuple[Path | None, list[tuple[str, float]]]:
    ref_path = find_reference_image(split_folder)
    if ref_path is None:
        return None, []

    ref = _read_gray(ref_path)
    if ref is None:
        return None, []

    scores = []
    best = None
    best_score = -1e18

    for fp in floor_pngs:
        img = _read_gray(fp)
        if img is None:
            continue
        sc = score_match(ref, img)
        scores.append((fp.name, sc))
        if sc > best_score:
            best_score = sc
            best = fp

    scores.sort(key=lambda x: x[1], reverse=True)
    return best, scores


# -----------------------------
# Main
# -----------------------------
def base_id_from_folder(folder_name: str):
    m = ID_RE.match(folder_name)
    return m.group(1) if m else None

def write_pick_report(out_dir: Path, bid: str, split_name: str, scores: list[tuple[str, float]], chosen: Path | None):
    txt = out_dir / "floor_pick.txt"
    lines = []
    lines.append(f"id={bid} folder={split_name}")
    if chosen is not None:
        lines.append(f"chosen={chosen.name}")
    else:
        lines.append("chosen=None")
    lines.append("scores:")
    for name, sc in scores:
        lines.append(f"  {name}\t{sc:.6f}")
    txt.write_text("\n".join(lines), encoding="utf-8")

def main(w=1024, h=1024, keep_spares=True):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    folders = [p for p in CUBI_FT_ROOT.iterdir() if p.is_dir()]
    print(f"[INFO] Found {len(folders)} cubi_ft folders")
    if not INKSCAPE_OK:
        print("[WARN] Inkscape not detected. If CairoSVG fails on some SVGs, the script will stop.")

    missing_id = []
    missing_svg = []
    done = 0
    failed = 0
    picked = 0

    for fold in folders:
        bid = base_id_from_folder(fold.name)
        if bid is None:
            missing_id.append(fold.name)
            continue

        svg_path = HQ_ROOT / bid / "model.svg"
        if not svg_path.exists():
            missing_svg.append(bid)
            continue

        out_dir = OUT_ROOT / fold.name
        try:
            outs = rasterize_svg_all_floors(svg_path, out_dir, w=w, h=h)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fold.name} -> {svg_path} ({type(e).__name__}: {e})")
            continue

        # Auto-pick best floor and copy to clean_rgb.png
        best, scores = pick_best_floor(fold, outs)
        if best is None:
            # fallback: pick floor1 if exists else first
            fallback = None
            for fp in outs:
                if fp.name.endswith("floor1.png"):
                    fallback = fp
                    break
            if fallback is None and outs:
                fallback = outs[0]
            if fallback is not None:
                shutil.copyfile(fallback, out_dir / "clean_rgb.png")
                write_pick_report(out_dir, bid, fold.name, scores, fallback)
                print(f"[OK] {fold.name} -> {len(outs)} floor(s) | PICK fallback={fallback.name}")
                picked += 1
            else:
                write_pick_report(out_dir, bid, fold.name, scores, None)
                print(f"[OK] {fold.name} -> {len(outs)} floor(s) | PICK none")
        else:
            shutil.copyfile(best, out_dir / "clean_rgb.png")
            write_pick_report(out_dir, bid, fold.name, scores, best)
            print(f"[OK] {fold.name} -> {len(outs)} floor(s) | PICK {best.name}")
            picked += 1

        # keep_spares=True means we leave clean_rgb_floor*.png files in place.
        # If keep_spares=False, we delete non-chosen floors.
        if not keep_spares and outs:
            chosen = out_dir / "clean_rgb.png"
            chosen_name = None
            if chosen.exists():
                # find which floor file matches chosen content by name from report
                pass
            # delete all floor pngs except the picked one
            if best is not None:
                for fp in outs:
                    if fp.name != best.name:
                        fp.unlink(missing_ok=True)
            else:
                # if best None, keep as-is
                pass

        done += 1

    print("\n===== SUMMARY =====")
    print(f"Rasterized: {done}/{len(folders)} folders")
    print(f"Picked clean_rgb.png: {picked}")
    print(f"Failed: {failed}")
    if missing_id:
        print(f"Folders with no numeric prefix (skipped): {len(missing_id)}")
        for x in missing_id[:20]:
            print("  -", x)
        if len(missing_id) > 20:
            print("  ...")
    if missing_svg:
        uniq = sorted(set(missing_svg), key=lambda x: int(x))
        print(f"IDs missing model.svg in HQ_ROOT: {len(uniq)}")
        for x in uniq[:30]:
            print("  -", x)
        if len(uniq) > 30:
            print("  ...")


if __name__ == "__main__":
    main(w=1024, h=1024, keep_spares=True)
