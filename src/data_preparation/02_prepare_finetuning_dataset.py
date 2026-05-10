import os
import json
import re
import copy
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw

# ==========================================
# CONFIGURATION
# ==========================================

CLASS_MAP = {
    "Background": 0,
    "Room": 1,
    "Wall": 2,
    "Door": 3,
    "Stairs": 4,
    "Window": 5
}

VIS_COLORS = {
    "Room": (50, 50, 50),     # Dark Gray
    "Wall": (0, 0, 255),      # Red
    "Door": (0, 255, 0),      # Green
    "Stairs": (255, 0, 0),    # Blue
    "Window": (255, 255, 0)   # Cyan/Yellow
}

# ==========================================
# PART 1: ROBUST DECODER
# ==========================================
def decode_segmentation(seg, H, W):
    if isinstance(seg, list):
        img = Image.new("L", (W, H), 0)
        drw = ImageDraw.Draw(img)
        polys = seg if (len(seg) > 0 and isinstance(seg[0], list)) else [seg]
        for poly in polys:
            if not poly or len(poly) < 6: continue
            pts = []
            for v in poly:
                try: pts.append(float(v))
                except Exception: pass
            if len(pts) < 6 or (len(pts) % 2) != 0: continue
            xy = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
            drw.polygon(xy, outline=1, fill=1)
        return np.array(img, dtype=np.uint8)

    if isinstance(seg, dict) and "counts" in seg:
        counts = seg["counts"]
        size = seg.get("size", [H, W])
        h, w = int(size[0]), int(size[1])
        if isinstance(counts, str): counts = _coco_rle_string_to_counts(counts)
        if not isinstance(counts, (list, tuple)): return np.zeros((H, W), dtype=np.uint8)
        flat = np.zeros(h * w, dtype=np.uint8)
        idx, val = 0, 0
        for run in counts:
            run = int(run)
            if run < 0: return np.zeros((H, W), dtype=np.uint8)
            if run > 0:
                if val == 1:
                    end = min(idx + run, flat.size)
                    flat[idx:end] = 1
                idx += run
                if idx >= flat.size: break
            val ^= 1
        mask = flat.reshape((w, h)).T
        if (h, w) != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8)

    return np.zeros((H, W), dtype=np.uint8)

def _coco_rle_string_to_counts(s):
    counts = []
    p, m = 0, 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            k += 1
            if p >= len(s) and more: more = 0
        if c & 0x10: x |= -1 << (5 * k)
        if m > 1: x += counts[m - 2]
        counts.append(x)
        m += 1
    return counts

# ==========================================
# PART 2: MERGING & NORMALIZING
# ==========================================
def merge_jsons(json_path_1, json_path_2, output_path):
    print(f"[1/5] Merging {json_path_1.name} and {json_path_2.name}...")
    with open(json_path_1, "r", encoding="utf-8") as f: old = json.load(f)
    with open(json_path_2, "r", encoding="utf-8") as f: new = json.load(f)

    name_to_id = {c["name"]: c["id"] for c in old.get("categories", [])}
    max_cat_id = max(name_to_id.values()) if name_to_id else 0
    cat_map = {}

    for c in new.get("categories", []):
        name = c["name"]
        if name in name_to_id:
            cat_map[c["id"]] = name_to_id[name]
        else:
            max_cat_id += 1
            name_to_id[name] = max_cat_id
            cat_map[c["id"]] = max_cat_id
            old["categories"].append({"id": max_cat_id, "name": name, "supercategory": c.get("supercategory", "")})

    max_img_id = max(im["id"] for im in old.get("images", [])) if old.get("images") else 0
    max_ann_id = max(a["id"] for a in old.get("annotations", [])) if old.get("annotations") else 0

    for im in new.get("images", []):
        im2 = copy.deepcopy(im)
        im2["id"] = im["id"] + max_img_id
        old["images"].append(im2)

    for a in new.get("annotations", []):
        a2 = copy.deepcopy(a)
        a2["id"] = a["id"] + max_ann_id
        a2["image_id"] = a["image_id"] + max_img_id
        a2["category_id"] = cat_map[a["category_id"]]
        old["annotations"].append(a2)
    
    return old

def fix_and_normalize_labels(data):
    """
    Consolidates variant labels:
    - walls, wall_polygon -> Wall
    - windows, glass -> Window
    """
    print("[2/5] Normalizing Wall and Window labels...")

    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = s.replace(" ", "_").replace("-", "_")
        while "__" in s: s = s.replace("__", "_")
        return s

    # 1. Define Merge Groups
    merge_groups = {
        "Wall": ["wall", "walls", "wall_polygon", "wallpoly", "wall_poly"],
        "Window": ["window", "windows", "glass", "fenestration"],
        # Add others if needed: "Door": ["door", "doors", "entrance"]
    }

    # 2. Find or Create Canonical Categories
    canonical_ids = {} # {"Wall": 12, "Window": 15}
    existing_cat_ids = [c["id"] for c in data.get("categories", [])]
    next_id = max(existing_cat_ids, default=0) + 1

    for target_name, variants in merge_groups.items():
        # Look for existing exact match first
        found_id = None
        for c in data.get("categories", []):
            if norm(c.get("name")) == norm(target_name):
                found_id = c["id"]
                break
        
        # If not found, create it
        if found_id is None:
            found_id = next_id
            next_id += 1
            data.setdefault("categories", []).append({
                "id": found_id, "name": target_name, "supercategory": "structure"
            })
        
        canonical_ids[target_name] = found_id

    # 3. Build Remapping Table
    # Map old_id -> new_canonical_id
    id_remap = {}
    for c in data.get("categories", []):
        n = norm(c.get("name"))
        # Check against merge groups
        for target_name, variants in merge_groups.items():
            # If name matches any variant (e.g. "wall_polygon"), map it to "Wall" ID
            if n in variants or any(v in n for v in variants):
                id_remap[c["id"]] = canonical_ids[target_name]
                break

    # 4. Remap Annotations
    remapped_count = 0
    for ann in data.get("annotations", []):
        old_id = ann.get("category_id")
        if old_id in id_remap:
            ann["category_id"] = id_remap[old_id]
            remapped_count += 1

    # 5. Clean Categories (Remove merged duplicates)
    final_cats = []
    seen_ids = set()
    # Keep canonicals and anything NOT in the remap source list
    # (i.e. keep Rooms, Stairs, Doors)
    remapped_sources = set(id_remap.keys())
    
    for c in data.get("categories", []):
        # If it's one of our canonical targets, keep it
        if c["id"] in canonical_ids.values():
            if c["id"] not in seen_ids:
                final_cats.append(c)
                seen_ids.add(c["id"])
        # If it was remapped (merged into something else), skip it
        elif c["id"] in remapped_sources:
            continue
        # Otherwise (Room, Door, Stairs), keep it
        else:
            if c["id"] not in seen_ids:
                final_cats.append(c)
                seen_ids.add(c["id"])

    data["categories"] = final_cats

    print(f"      Remapped {remapped_count} annotations.")
    print(f"      Canonical IDs: {canonical_ids}")
    return data

# ==========================================
# PART 3: PROCESSOR
# ==========================================
def extract_numeric_id(text):
    m = re.findall(r"\d+", text)
    return m[-1].lstrip("0") or "0" if m else None

def build_image_index(root_dir):
    print(f"[3/5] Indexing images in {root_dir}...")
    index = {}
    for p in Path(root_dir).rglob("*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "photo_rect" in p.name:
             id_str = extract_numeric_id(p.parent.name) or extract_numeric_id(p.stem)
             if id_str: index[id_str] = p
    return index

def process_node(id_str, img, vis_img, room_mask, door_mask, window_mask, class_npy, out_root, overrides):
    H, W = img.shape[:2]
    
    # 1. Check for Manual Split
    split_action = None
    if id_str in overrides:
        rule = overrides[id_str]
        if rule["type"] == "split":
            split_action = (rule["axis"], rule["val"])
            print(f"      [MANUAL] Splitting {id_str} at {rule['axis']}={rule['val']}")

    # 2. Execute Split (Recursion)
    if split_action:
        axis, val = split_action
        if val < 5 or (axis=='y' and val > H-5) or (axis=='x' and val > W-5):
            print(f"      [WARN] Split val {val} out of bounds for {id_str}. Saving as Single.")
        else:
            if axis == 'y':
                process_node(id_str+"_T", img[:val], vis_img[:val], room_mask[:val], door_mask[:val], window_mask[:val], class_npy[:val], out_root, overrides)
                process_node(id_str+"_B", img[val:], vis_img[val:], room_mask[val:], door_mask[val:], window_mask[val:], class_npy[val:], out_root, overrides)
            else:
                process_node(id_str+"_L", img[:, :val], vis_img[:, :val], room_mask[:, :val], door_mask[:, :val], window_mask[:, :val], class_npy[:, :val], out_root, overrides)
                process_node(id_str+"_R", img[:, val:], vis_img[:, val:], room_mask[:, val:], door_mask[:, val:], window_mask[:, val:], class_npy[:, val:], out_root, overrides)
            return

    # 3. Save Leaf Node
    out_dir = out_root / id_str
    os.makedirs(out_dir, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "photo_rect.png"), img)
    cv2.imwrite(str(out_dir / "photo_room.png"), room_mask)
    cv2.imwrite(str(out_dir / "photo_icon.png"), door_mask)
    
    # Optional: Save Window mask for debug (not strictly used by CubiCasa code but useful)
    # cv2.imwrite(str(out_dir / "photo_window.png"), window_mask) 
    
    np.save(str(out_dir / "room_labels.npy"), class_npy)
    cv2.imwrite(str(out_dir / "finetune_label.png"), vis_img)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    BASE = Path(r"C:\Users\nicho\CubiCasa5k")
    JSON_1 = BASE / "instances_default_3.json"
    JSON_2 = BASE / "instances_default_4.json"
    RAW_DIR = BASE / "photos_eval_train"
    OUT_DIR = BASE / "data" / "cubi_ft"
    OVERRIDE_FILE = BASE / "split_overrides.json"

    # Load Overrides
    overrides = {}
    if OVERRIDE_FILE.exists():
        with open(OVERRIDE_FILE) as f: overrides = json.load(f)
        print(f"Loaded {len(overrides)} manual decisions.")

    # Merge & Fix
    merged = merge_jsons(JSON_1, JSON_2, BASE / "merged.json")
    final_data = fix_and_normalize_labels(merged)
    
    img_index = build_image_index(RAW_DIR)
    cat_to_name = {c["id"]: c["name"] for c in final_data["categories"]}
    anns_by_img = {}
    for a in final_data["annotations"]: anns_by_img.setdefault(a["image_id"], []).append(a)

    print("[4/5] Generating Data...")
    
    for im in final_data["images"]:
        id_str = extract_numeric_id(Path(im["file_name"]).name)
        if not id_str: id_str = extract_numeric_id(Path(im["file_name"]).parent.name)
        
        if not id_str or id_str not in img_index: 
            continue

        path = img_index[id_str]
        bgr = cv2.imread(str(path))
        if bgr is None: continue
        
        H, W = bgr.shape[:2]

        vis_img = bgr.copy()
        room_mask = np.zeros((H,W), np.uint8)
        door_mask = np.zeros((H,W), np.uint8)
        window_mask = np.zeros((H,W), np.uint8)
        class_npy = np.zeros((H,W), dtype=np.int32)

        anns = anns_by_img.get(im["id"], [])
        
        # Sort so Rooms are drawn first, then items on top
        sorted_anns = sorted(anns, key=lambda x: 0 if "room" in cat_to_name.get(x["category_id"], "").lower() else 1)

        for a in sorted_anns:
            cname = cat_to_name.get(a["category_id"], "Undefined")
            mask = decode_segmentation(a["segmentation"], im["height"], im["width"])
            
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask > 0
            name_lower = cname.lower()

            # --- CASE-INSENSITIVE MAPPING ---
            if "room" in name_lower:
                class_npy[mask_bool] = CLASS_MAP["Room"]
                room_mask = cv2.bitwise_or(room_mask, mask)
                overlay = vis_img.copy()
                overlay[mask_bool] = VIS_COLORS["Room"]
                vis_img = cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0)

            elif "wall" in name_lower:
                class_npy[mask_bool] = CLASS_MAP["Wall"]
                vis_img[mask_bool] = VIS_COLORS["Wall"]

            elif "door" in name_lower:
                class_npy[mask_bool] = CLASS_MAP["Door"]
                door_mask = cv2.bitwise_or(door_mask, mask)
                vis_img[mask_bool] = VIS_COLORS["Door"]

            elif "stairs" in name_lower:
                class_npy[mask_bool] = CLASS_MAP["Stairs"]
                vis_img[mask_bool] = VIS_COLORS["Stairs"]

            # Extract window masks when annotations are available.
            elif "window" in name_lower:
                class_npy[mask_bool] = CLASS_MAP["Window"]
                window_mask = cv2.bitwise_or(window_mask, mask)
                vis_img[mask_bool] = VIS_COLORS["Window"]

        # Recursion
        process_node(id_str, bgr, vis_img, room_mask, door_mask, window_mask, class_npy, OUT_DIR, overrides)

    print("✅ DONE! Data preparation complete.")
