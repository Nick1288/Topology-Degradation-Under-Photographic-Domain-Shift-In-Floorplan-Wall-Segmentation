import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


FRAG_THRESHOLD = 30
ENCLOSED_THRESHOLD = 0.25
LEAK_THRESHOLD = 0.75

SELECTED = {
    "fragmentation_only": "7595",
    "enclosure_leakage_only": "8091",
    "all_three": "7748",
}


def read_img(path: Path):
    if path is None or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255)
        img = (bgr.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return img


def read_mask(path: Path):
    if path is None or not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def find_photo(plan_id: str, photo_root: Path):
    candidates = [
        photo_root / plan_id / "photo_rect.png",
        photo_root / plan_id / "photo.png",
        photo_root / plan_id / "img.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_wall(plan_id: str, pred_root: Path):
    candidates = [
        pred_root / plan_id / "pred_walllike.png",
        pred_root / plan_id / "wall.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def overlay_wall(base, wall_mask, alpha=0.45):
    out = base.copy()
    if wall_mask is None:
        return out
    color = np.zeros_like(out)
    color[:, :, 2] = 255
    mask = wall_mask.astype(bool)
    out[mask] = (
        out[mask].astype(np.float32) * (1 - alpha)
        + color[mask].astype(np.float32) * alpha
    ).astype(np.uint8)
    return out


def compute_connected_components(mask01):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    return n, labels, stats, centroids


def color_components(mask01):
    n, labels, stats, centroids = compute_connected_components(mask01)
    out = np.full((mask01.shape[0], mask01.shape[1], 3), 255, dtype=np.uint8)

    order = list(range(1, n))
    order.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)

    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255)
    ]

    for idx, comp_id in enumerate(order):
        out[labels == comp_id] = palette[idx % len(palette)]

    for rank, comp_id in enumerate(order[:8], start=1):
        cx, cy = centroids[comp_id]
        cv2.putText(out, str(rank), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, str(rank), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def flood_outside(free01):
    h, w = free01.shape
    flood = free01.copy()

    for x in range(w):
        if flood[0, x]:
            cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (x, 0), 2)
        if flood[h - 1, x]:
            cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (x, h - 1), 2)

    for y in range(h):
        if flood[y, 0]:
            cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, y), 2)
        if flood[y, w - 1]:
            cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (w - 1, y), 2)

    outside = (flood == 2).astype(np.uint8)
    enclosed = (free01.astype(np.uint8) & (1 - outside)).astype(np.uint8)
    return outside, enclosed


def free_space_diagnostic(wall01):
    free = (1 - wall01).astype(np.uint8)
    outside, enclosed = flood_outside(free)

    vis = np.full((wall01.shape[0], wall01.shape[1], 3), 255, dtype=np.uint8)
    vis[wall01.astype(bool)] = (0, 0, 0)         # black walls
    vis[outside.astype(bool)] = (0, 165, 255)    # orange leakage
    vis[enclosed.astype(bool)] = (0, 180, 0)     # green enclosed free space
    return vis


def combined_diagnostic(wall01):
    comp_vis = color_components(wall01)
    free_vis = free_space_diagnostic(wall01)

    out = free_vis.copy()
    mask = wall01.astype(bool)
    out[mask] = comp_vis[mask]
    return out


def add_top_title(img, title):
    top_h = 60
    out = np.full((img.shape[0] + top_h, img.shape[1], 3), 255, dtype=np.uint8)
    out[top_h:, :, :] = img
    cv2.putText(out, title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def draw_legend(mode, canvas_width=420, canvas_height=896):
    leg = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    cv2.putText(leg, "Legend", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)

    y = 130

    if mode == "fragmentation_only":
        cv2.putText(leg, "colours", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "= connected", (95, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "components", (95, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)

        cv2.putText(leg, "1,2,3...", (95, y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "= largest", (95, y + 196), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "fragments", (95, y + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)

    elif mode == "enclosure_leakage_only":
        cv2.rectangle(leg, (35, y - 28), (75, y + 6), (0, 165, 255), -1)
        cv2.putText(leg, "outside-connected", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "free space", (95, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        y += 110

        cv2.rectangle(leg, (35, y - 28), (75, y + 6), (0, 180, 0), -1)
        cv2.putText(leg, "enclosed", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "free space", (95, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        y += 110

        cv2.rectangle(leg, (35, y - 28), (75, y + 6), (0, 0, 0), -1)
        cv2.putText(leg, "predicted", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "walls", (95, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)

    else:
        cv2.rectangle(leg, (35, y - 28), (75, y + 6), (0, 165, 255), -1)
        cv2.putText(leg, "outside-connected", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "free space", (95, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        y += 110

        cv2.rectangle(leg, (35, y - 28), (75, y + 6), (0, 180, 0), -1)
        cv2.putText(leg, "enclosed", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "free space", (95, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        y += 110

        cv2.putText(leg, "coloured walls", (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "= connected", (95, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(leg, "components", (95, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)

    return leg


def resize_same_height(images, h=720):
    out = []
    for img in images:
        scale = h / img.shape[0]
        w = int(round(img.shape[1] * scale))
        out.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    return out


def concat_h(images, pad=20):
    h = max(img.shape[0] for img in images)
    out = images[0]
    for img in images[1:]:
        gap = np.full((h, pad, 3), 255, dtype=np.uint8)
        out = np.concatenate([out, gap, img], axis=1)
    return out


def concat_v(images, pad=30):
    w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[1] != w:
            canvas = np.full((img.shape[0], w, 3), 255, dtype=np.uint8)
            x = (w - img.shape[1]) // 2
            canvas[:, x:x + img.shape[1]] = img
            img = canvas
        padded.append(img)

    out = padded[0]
    for img in padded[1:]:
        gap = np.full((pad, w, 3), 255, dtype=np.uint8)
        out = np.concatenate([out, gap, img], axis=0)
    return out


def build_figure(plan_id, mode, photo_root: Path, pred_root: Path):
    photo = read_img(find_photo(plan_id, photo_root))
    wall = read_mask(find_wall(plan_id, pred_root))

    if photo is None:
        photo = np.full((896, 896, 3), 255, dtype=np.uint8)
    if wall is None:
        wall = np.zeros((photo.shape[0], photo.shape[1]), dtype=np.uint8)

    if wall.shape[:2] != photo.shape[:2]:
        wall = cv2.resize(wall, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST)

    left = overlay_wall(photo, wall)

    if mode == "fragmentation_only":
        middle = color_components(wall)
        title = "Failure mode A: Fragmentation only"
    elif mode == "enclosure_leakage_only":
        middle = free_space_diagnostic(wall)
        title = "Failure mode B: Enclosure loss + leakage"
    else:
        middle = combined_diagnostic(wall)
        title = "Failure mode C: All three criteria violated"

    right = draw_legend(mode, canvas_width=420, canvas_height=photo.shape[0])

    left, middle, right = resize_same_height([left, middle, right], h=720)
    row = concat_h([left, middle, right], pad=24)
    row = add_top_title(row, title)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo-root", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_figs = []

    for mode, plan_id in SELECTED.items():
        fig = build_figure(plan_id, mode, Path(args.photo_root), Path(args.pred_root))
        out_path = out_dir / f"{mode}.png"
        cv2.imwrite(str(out_path), fig)
        all_figs.append(fig)
        print(f"[OK] Saved {mode}: {out_path}")

    combined = concat_v(all_figs, pad=36)
    combined_path = out_dir / "selected_failure_modes_combined.png"
    cv2.imwrite(str(combined_path), combined)
    print(f"[OK] Saved combined figure: {combined_path}")


if __name__ == "__main__":
    main()