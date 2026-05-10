import cv2
import json
import os
import numpy as np
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_DIR = r"C:\Users\nicho\CubiCasa5k\photos_eval_train"
OUTPUT_JSON = r"C:\Users\nicho\CubiCasa5k\split_overrides.json"

MAX_VIEW_W = 1280
MAX_VIEW_H = 800

# FILES TO IGNORE (Junk filter)
IGNORE_TERMS = ["debug", "mask", "label", "crop", "segmentation", "out", "vis"]
# ==========================================

def extract_id(filename):
    m = re.findall(r"\d+", filename)
    return m[-1].lstrip("0") or "0" if m else None

# --- SAFE RECTIFICATION ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def rectify_image(image):
    h_orig, w_orig = image.shape[:2]
    total_area = h_orig * w_orig
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > (0.25 * total_area): 
                screenCnt = approx; break

    if screenCnt is None: return image

    pts = screenCnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

def get_best_image_for_id(image_list):
    # Priority: photo_rect > original > shortest filename
    for p in image_list:
        if "photo_rect" in p.name: return p
    for p in image_list:
        if "original" in p.name: return p
    return sorted(image_list, key=lambda x: len(x.name))[0]

def main():
    overrides = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f: overrides = json.load(f)
        print(f"Loaded {len(overrides)} existing decisions.")

    print("Scanning images...")
    candidates = {}
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in Path(IMAGE_DIR).rglob(ext):
            if any(term in p.name.lower() for term in IGNORE_TERMS): continue
            pid = extract_id(p.stem) or extract_id(p.parent.name)
            if pid: candidates.setdefault(pid, []).append(p)

    raw_files = {}
    for pid, paths in candidates.items():
        raw_files[pid] = get_best_image_for_id(paths)

    # Queue: (Virtual_ID, Image_Array, Original_ID)
    queue = []
    sorted_ids = sorted(raw_files.keys())
    for pid in sorted_ids:
        # Only add if root ID is NOT in overrides
        if pid not in overrides:
            queue.append( (pid, None, pid) )

    print(f"Queue size: {len(queue)} items.")
    print("\nCONTROLS:")
    print("  [Click]   : Set Split Line")
    print("  [R-Click] : Rotate Axis")
    print("  [SPACE]   : Confirm Split (Creates Sub-Tasks)")
    print("  [S]       : Single (Done)")
    print("  [ESC]     : Quit")

    cv2.namedWindow("Splitter", cv2.WINDOW_NORMAL) 

    current_idx = 0
    while current_idx < len(queue):
        virtual_id, img_data, root_id = queue[current_idx]
        
        # Skip if done
        if virtual_id in overrides:
            current_idx += 1; continue

        # Load & Rectify (Lazy Load)
        if img_data is None:
            path = raw_files[root_id]
            print(f"Loading Root: {path.name}")
            raw = cv2.imread(str(path))
            if raw is None: current_idx += 1; continue
            
            if "photo_rect" in path.name:
                img_data = raw
            else:
                img_data = rectify_image(raw)

        # --- VIEW SCALING ---
        H, W = img_data.shape[:2]
        scale_w = MAX_VIEW_W / W
        scale_h = MAX_VIEW_H / H
        scale = min(scale_w, scale_h)
        if scale > 1.0: scale = 1.0 # Don't upscale small crops
            
        disp_w = int(W * scale)
        disp_h = int(H * scale)
        
        # Default Split State
        split_axis = 'y' if H > W else 'x' 
        real_split_pos = H // 2 if split_axis == 'y' else W // 2
        
        def mouse_cb(event, x, y, flags, param):
            nonlocal real_split_pos, split_axis
            real_x = int(x / scale)
            real_y = int(y / scale)
            if event == cv2.EVENT_LBUTTONDOWN:
                real_split_pos = real_x if split_axis == 'x' else real_y
            elif event == cv2.EVENT_RBUTTONDOWN:
                split_axis = 'x' if split_axis == 'y' else 'y'
                real_split_pos = W // 2 if split_axis == 'x' else H // 2

        cv2.setMouseCallback("Splitter", mouse_cb)

        print(f"[{current_idx+1}/{len(queue)}] Reviewing: {virtual_id}")

        done_with_item = False
        while not done_with_item:
            # Render
            display = cv2.resize(img_data, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            
            disp_pos = int(real_split_pos * scale)
            if split_axis == 'x':
                cv2.line(display, (disp_pos, 0), (disp_pos, disp_h), (0, 255, 0), 2)
            else:
                cv2.line(display, (0, disp_pos), (disp_w, disp_pos), (0, 255, 0), 2)

            cv2.rectangle(display, (0, 0), (disp_w, 35), (0, 0, 0), -1)
            info = f"ID: {virtual_id} | [S]=Single | [SPACE]=Split"
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            cv2.imshow("Splitter", display)
            key = cv2.waitKey(20) & 0xFF

            if key == 27: # ESC
                with open(OUTPUT_JSON, "w") as f: json.dump(overrides, f, indent=2)
                return

            if key == ord('s'): # Single
                overrides[virtual_id] = {"type": "single"}
                print(f"  -> SAVED: {virtual_id} = Single")
                done_with_item = True
            
            if key == 32: # SPACE -> RECURSION TRIGGER
                overrides[virtual_id] = {"type": "split", "axis": split_axis, "val": real_split_pos}
                print(f"  -> SAVED: {virtual_id} = Split {split_axis}={real_split_pos}")
                
                # --- CREATE CHILDREN ---
                if split_axis == 'y':
                    # Crop logic
                    top = img_data[:real_split_pos, :]
                    bot = img_data[real_split_pos:, :]
                    
                    # Insert at FRONT of queue so you see them NEXT
                    queue.insert(current_idx + 1, (virtual_id + "_T", top, root_id))
                    queue.insert(current_idx + 2, (virtual_id + "_B", bot, root_id))
                else:
                    left = img_data[:, :real_split_pos]
                    right = img_data[:, real_split_pos:]
                    
                    queue.insert(current_idx + 1, (virtual_id + "_L", left, root_id))
                    queue.insert(current_idx + 2, (virtual_id + "_R", right, root_id))
                
                done_with_item = True

        if len(overrides) % 5 == 0:
            with open(OUTPUT_JSON, "w") as f: json.dump(overrides, f, indent=2)

        current_idx += 1

    with open(OUTPUT_JSON, "w") as f: json.dump(overrides, f, indent=2)
    print("Done! All items processed.")

if __name__ == "__main__":
    main()