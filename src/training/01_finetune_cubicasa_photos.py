import os
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from floortrans.models import get_model

# ==========================================
# CONFIGURATION
# ==========================================
NUM_CLASSES = 6 
IMG_SIZE = 1024

CLASS_NAMES = ["Background", "Room", "Wall", "Door", "Stair", "Window"]
WW3_CLASS_NAMES = ["Background", "Wall", "Window"]
W2_CLASS_NAMES = ["Background", "WallLike"]
W3D_CLASS_NAMES = ["Background", "WallLike", "Door"]

COLORS = {
    0: (0,0,0),       # Bg
    1: (100,100,100), # Room (Gray)
    2: (0,0,255),     # Wall (Red)
    3: (0,255,0),     # Door (Green)
    4: (255,0,0),     # Stair (Blue)
    5: (0,255,255)    # Window (Yellow)
}

# ==========================================
# 1. HELPER FUNCTIONS (Image Proc & Metrics)
# ==========================================
def compute_ink_mask(img_bgr, thresh=200):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < thresh).astype(np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    edges = (edges > 0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges_d = cv2.dilate(edges, ker, iterations=1)
    ink = (ink & edges_d).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker, iterations=1)
    return ink

def build_ignore_mask(img_bgr, walllike_gt, max_cc_area=3000):
    ink = compute_ink_mask(img_bgr)
    ignore = ((ink == 1) & (walllike_gt == 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ignore, connectivity=8)
    out = np.zeros_like(ignore)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_cc_area:
            out[lab == i] = 1
    return out.astype(np.uint8)

def flood_outside(non_wall_bin):
    h, w = non_wall_bin.shape
    ff = non_wall_bin.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    seeds = []
    for x in range(w):
        if ff[0, x] == 1: seeds.append((x, 0))
        if ff[h-1, x] == 1: seeds.append((x, h-1))
    for y in range(h):
        if ff[y, 0] == 1: seeds.append((0, y))
        if ff[y, w-1] == 1: seeds.append((w-1, y))

    outside = np.zeros_like(ff, dtype=np.uint8)
    for (sx, sy) in seeds:
        if outside[sy, sx] == 1: continue
        tmp = ff.copy()
        cv2.floodFill(tmp, mask, (sx, sy), 2)
        newly = (tmp == 2).astype(np.uint8)
        outside = np.maximum(outside, newly)
        ff[newly == 1] = 0
    return outside

def boundary_map(bin_mask):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    er = cv2.erode(bin_mask, ker)
    return (bin_mask - er).clip(0,1)

def boundary_f1(pred, gt, tol=2):
    pb = boundary_map(pred)
    gb = boundary_map(gt)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
    gb_d = cv2.dilate(gb, ker)
    pb_d = cv2.dilate(pb, ker)
    tp = (pb & gb_d).sum()
    fp = (pb & ~gb_d).sum()
    fn = (gb & ~pb_d).sum()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)

def derive_windows_from_wall(img_bgr, walllike_bin, canny1=60, canny2=150, thin_pct=30, min_area=60):
    H, W = walllike_bin.shape
    wall = (walllike_bin > 0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    wall_er = cv2.erode(wall, ker, iterations=1)
    wall_boundary = (wall - wall_er).clip(0,1)

    non_wall = (1 - wall).astype(np.uint8)
    outside = flood_outside(non_wall)

    dist = cv2.distanceTransform(wall, cv2.DIST_L2, 5)
    vals = dist[wall == 1]
    if vals.size == 0: return np.zeros((H,W), np.uint8)

    tau = np.percentile(vals, thin_pct)
    thin = ((dist > 0) & (dist <= max(1.0, tau))).astype(np.uint8)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = (cv2.Canny(gray, canny1, canny2) > 0).astype(np.uint8)
    cand = (thin & cv2.dilate(edges, ker, iterations=1)).astype(np.uint8)

    near_boundary = cv2.dilate(wall_boundary, ker, iterations=2)
    near_outside = cv2.dilate(outside, ker, iterations=2)
    cand = (cand & near_boundary & (near_outside > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, ker, iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    win = np.zeros_like(cand)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            win[lab == i] = 1
    return win

def skeletonize_cv(bin_mask):
    img = (bin_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0: break
    return (skel > 0).astype(np.uint8)

def skeleton_iou(pred, gt):
    sp = skeletonize_cv(pred)
    sg = skeletonize_cv(gt)
    inter = np.logical_and(sp, sg).sum()
    union = np.logical_or(sp, sg).sum()
    return inter / (union + 1e-6)

def bin_iou(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-6)

def comp_count(bin_mask):
    bin_mask = (bin_mask > 0).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    return int(max(0, n - 1))

def postprocess_walllike(pred_bin, min_cc_area=80, close_ks=3):
    m = (pred_bin > 0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ks, close_ks))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_cc_area:
            out[lab == i] = 1
    return out

# ==========================================
# 2. DATASET & SAMPLER
# ==========================================
def focus_crop_multi(img, lbl, out_size, *masks, crop_size=512, focus_prob=0.8, focus_classes=(2,5)):
    H, W = lbl.shape[:2]
    do_focus = (random.random() < focus_prob)
    if do_focus:
        coords = []
        for c in focus_classes:
            ys, xs = np.where(lbl == c)
            if len(xs) > 0: coords.append((ys, xs))
        if coords:
            ys, xs = random.choice(coords)
            j = random.randrange(len(xs))
            cy, cx = int(ys[j]), int(xs[j])
        else:
            cy, cx = random.randrange(H), random.randrange(W)
    else:
        cy, cx = random.randrange(H), random.randrange(W)

    half = crop_size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    pad_top, pad_left = max(0, -y0), max(0, -x0)
    pad_bot, pad_right = max(0, y1 - H), max(0, x1 - W)

    if pad_top or pad_left or pad_bot or pad_right:
        img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255,255,255))
        lbl = cv2.copyMakeBorder(lbl, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        masks = [cv2.copyMakeBorder(m, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0) for m in masks]
        y0 += pad_top; y1 += pad_top; x0 += pad_left; x1 += pad_left

    img_c = cv2.resize(img[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    lbl_c = cv2.resize(lbl[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    masks_c = [cv2.resize(m[y0:y1, x0:x1], (out_size, out_size), interpolation=cv2.INTER_NEAREST) for m in masks]
    return (img_c, lbl_c, *masks_c)


def enforce_gt_priority(label_np, wall_id=2, door_id=3, window_id=5):
    out = label_np.copy()
    out[out == door_id] = door_id
    out[out == window_id] = window_id
    return out

class WindowAwareSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, window_class_id=5, window_ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.window_ratio = window_ratio
        self.window_indices = []
        self.other_indices = []
        print("[SAMPLER] Scanning dataset for Windows (Class 5)...")
        for i, sample in enumerate(dataset.samples):
            lbl_path = sample / "room_labels.npy"
            if lbl_path.exists():
                lbl = np.load(str(lbl_path))
                if window_class_id in lbl: self.window_indices.append(i)
                else: self.other_indices.append(i)
            else: self.other_indices.append(i)
        self.num_batches = len(dataset) // batch_size
        print(f"[SAMPLER] Found {len(self.window_indices)} images with Windows, {len(self.other_indices)} without.")

    def __iter__(self):
        n_win = int(self.batch_size * self.window_ratio)
        n_other = self.batch_size - n_win
        win_idx = self.window_indices[:]
        other_idx = self.other_indices[:]
        random.shuffle(win_idx)
        random.shuffle(other_idx)
        import itertools
        win_iter = itertools.cycle(win_idx)
        other_iter = itertools.cycle(other_idx)
        for _ in range(self.num_batches):
            batch = []
            for _ in range(n_win): batch.append(next(win_iter))
            for _ in range(n_other): batch.append(next(other_iter))
            random.shuffle(batch)
            yield batch

    def __len__(self): return self.num_batches
    
class ForensicsDataset(Dataset):
    def __init__(self, data_root, split="train", img_size=1024):
        self.root = Path(data_root)
        self.img_size = img_size
        self.split = split
        self.samples = sorted([d for d in self.root.iterdir() if d.is_dir()])
        split_idx = int(0.9 * len(self.samples))
        if split == "train": self.samples = self.samples[:split_idx]
        else: self.samples = self.samples[split_idx:]
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        img_path = folder / "photo_rect.png"
        lbl_path = folder / "room_labels.npy"

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Missing image: {img_path}")

        if not lbl_path.exists():
            raise RuntimeError(f"Missing label: {lbl_path}")
        label = np.load(str(lbl_path)).astype(np.int32)

        # load masks at native resolution
        door = cv2.imread(str(folder / "photo_icon.png"), cv2.IMREAD_GRAYSCALE)
        door = (door > 0).astype(np.uint8) if door is not None else np.zeros(label.shape, np.uint8)

        win = None
        win_path = folder / "photo_window.png"
        if win_path.exists():
            win0 = cv2.imread(str(win_path), cv2.IMREAD_GRAYSCALE)
            if win0 is not None:
                win = (win0 > 0).astype(np.uint8)

        # resize FULL FRAME for both train and val (remove distribution shift)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        door = cv2.resize(door, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if win is not None:
            win = cv2.resize(win, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # apply overrides AFTER resize so everything lines up
        label = apply_icon_overrides(label, door, win)

        # Ignore small non-structural ink regions during loss calculation.
        structure = ((label == 2) | (label == 3) | (label == 5)).astype(np.uint8)
        ignore = build_ignore_mask(img, structure)

        img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
        return {"image": img_t, "label": torch.from_numpy(label).long(), "ignore": torch.from_numpy(ignore).bool()}


# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def remap_w2(lbl_t):
    out = torch.zeros_like(lbl_t)
    out[(lbl_t == 2) | (lbl_t == 5)] = 1
    return out

def remap_w3d(lbl_t):
    out = torch.zeros_like(lbl_t)
    out[(lbl_t == 2) | (lbl_t == 5)] = 1  # wall + window => walllike
    out[lbl_t == 3] = 2                  # door
    return out

class DoorAwareSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, min_door_px=50, door_ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.door_ratio = door_ratio
        self.door_indices = []
        self.other_indices = []

        print("[SAMPLER] Scanning dataset for Doors (photo_icon.png)...")
        for i, folder in enumerate(dataset.samples):
            p = Path(folder)
            dpath = p / "photo_icon.png"
            if dpath.exists():
                m = cv2.imread(str(dpath), cv2.IMREAD_GRAYSCALE)
                if m is not None and int((m > 0).sum()) >= min_door_px:
                    self.door_indices.append(i)
                else:
                    self.other_indices.append(i)
            else:
                self.other_indices.append(i)

        self.num_batches = len(dataset) // batch_size
        print(f"[SAMPLER] Found {len(self.door_indices)} images with Doors, {len(self.other_indices)} without.")

    def __iter__(self):
        n_door = int(self.batch_size * self.door_ratio)
        n_other = self.batch_size - n_door

        door_idx = self.door_indices[:]
        other_idx = self.other_indices[:]
        random.shuffle(door_idx)
        random.shuffle(other_idx)

        import itertools
        door_iter = itertools.cycle(door_idx if door_idx else other_idx)
        other_iter = itertools.cycle(other_idx if other_idx else door_idx)

        for _ in range(self.num_batches):
            batch = []
            for _ in range(n_door):
                batch.append(next(door_iter))
            for _ in range(n_other):
                batch.append(next(other_iter))
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def remap_ww3(lbl_t):
    out = torch.zeros_like(lbl_t)
    out[lbl_t == 2] = 1
    out[lbl_t == 5] = 2
    return out

class FeatureExtractor2(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.body = original_model
        self.feat_deep = None
        self.feat_shallow = None
        deep = None
        shallow = None
        convs = [(n, m) for n, m in self.body.named_modules() if isinstance(m, nn.Conv2d)]
        for n, m in convs:
            if m.out_channels == 256:
                shallow = (n, m)
                break
        for n, m in reversed(convs):
            if m.out_channels == 256:
                deep = (n, m)
                break
        if shallow is None or deep is None: raise RuntimeError("Could not find shallow/deep 256-ch convs.")
        print("[HOOK] shallow =", shallow[0])
        print("[HOOK] deep    =", deep[0])
        shallow[1].register_forward_hook(self._hook_shallow)
        deep[1].register_forward_hook(self._hook_deep)

    def _hook_shallow(self, m, i, o): self.feat_shallow = o
    def _hook_deep(self, m, i, o): self.feat_deep = o

    def forward(self, x):
        self.feat_shallow = None
        self.feat_deep = None
        _ = self.body(x)
        if self.feat_shallow is None or self.feat_deep is None: raise RuntimeError("Hook failed.")
        return self.feat_shallow, self.feat_deep

class SegmentationHead(nn.Module):
    def __init__(self, base_model, num_classes=6):
        super().__init__()
        raw_net = get_model("hg_furukawa_original", 51)
        self.backbone = FeatureExtractor2(raw_net)
        self.b1 = nn.Conv2d(256, 64, 1, bias=False)
        self.b2 = nn.Conv2d(256, 64, 3, padding=6, dilation=6, bias=False)
        self.b3 = nn.Conv2d(256, 64, 3, padding=12, dilation=12, bias=False)
        self.bn = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Conv2d(192, 128, 1, bias=False)
        self.bn_proj = nn.BatchNorm2d(128)
        self.shallow_proj = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(128 + 64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=4), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        feat_shallow, feat_deep = self.backbone(x)
        x1 = self.b1(feat_deep); x2 = self.b2(feat_deep); x3 = self.b3(feat_deep)
        deep = torch.cat([x1, x2, x3], dim=1)
        deep = self.relu(self.bn(deep))
        deep = self.relu(self.bn_proj(self.project(deep)))
        shallow = self.shallow_proj(feat_shallow)
        fused = torch.cat([deep, shallow], dim=1)
        fused = self.fuse(fused)
        return self.up(fused)

# ==========================================
# 4. LOSS & IO
# ==========================================
def masked_focal_ce(logits, targets, ignore_mask, weight, gamma=2.0):
    loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    loss = ((1 - torch.exp(-loss)) ** gamma) * loss
    loss = loss[~ignore_mask]
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=logits.device)

def apply_icon_overrides(label, door_mask, window_mask=None, wall_id=2, door_id=3, window_id=5):
    out = label.copy()
    if window_mask is not None:
        out[window_mask > 0] = window_id
    out[door_mask > 0] = door_id
    return out

def masked_tversky(logits, targets, ignore_mask, alpha=0.3, beta=0.7, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    C = probs.shape[1]
    onehot = F.one_hot(targets, C).permute(0,3,1,2).float()
    valid = (~ignore_mask).unsqueeze(1)
    probs = probs * valid
    onehot = onehot * valid
    dims = (0,2,3)
    TP = (probs * onehot).sum(dims)
    FP = (probs * (1 - onehot)).sum(dims)
    FN = ((1 - probs) * onehot).sum(dims)
    tversky = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    return 1 - tversky.mean()

class MetricTracker:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    def reset(self):
        self.inter = {i: 0 for i in range(self.num_classes)}
        self.union = {i: 0 for i in range(self.num_classes)}
        self.total_pix = 0
        self.correct_pix = 0
    def update(self, pred_logits, target):
        preds = torch.argmax(pred_logits, dim=1).cpu().numpy()
        targ = target.cpu().numpy()
        self.total_pix += targ.size
        self.correct_pix += (preds == targ).sum()
        for c in range(self.num_classes):
            p = (preds == c); t = (targ == c)
            self.inter[c] += np.logical_and(p, t).sum()
            self.union[c] += np.logical_or(p, t).sum()
    def summary(self):
        acc = self.correct_pix / max(1, self.total_pix)
        ious = {self.class_names[c]: self.inter[c]/max(1, self.union[c]) for c in range(self.num_classes)}
        vals = [v for k, v in ious.items() if k.lower() != "background"]
        mIoU = float(np.mean(vals)) if len(vals) else 0.0
        return acc, mIoU, ious

def save_vis(img_t, pred_t, gt_t, out_path, mode="full6"):
    img = (img_t.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    if mode == "ww3": palette = {0:(0,0,0), 1:(0,0,255), 2:(0,255,255)}
    elif mode == "w2": palette = {0:(0,0,0), 1:(0,0,255)}
    else: palette = COLORS
    def colorize(m):
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        for c, col in palette.items():
            if c == 0: continue
            vis[m == c] = col
        return vis
    p_vis = colorize(pred_t.cpu().numpy())
    g_vis = colorize(gt_t.cpu().numpy())
    combo = np.hstack([cv2.addWeighted(img, 0.7, p_vis, 0.3, 0), cv2.addWeighted(img, 0.7, g_vis, 0.3, 0)])
    cv2.putText(combo, f"Pred vs GT ({mode})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imwrite(out_path, combo)

def save_vis_windows(img_bgr, walllike_bin, win_bin, out_path):
    img = img_bgr.copy()
    wall_col = np.zeros_like(img); wall_col[walllike_bin > 0] = (0,0,255)
    win_col = np.zeros_like(img); win_col[win_bin > 0] = (0,255,255)
    over = cv2.addWeighted(img, 0.75, wall_col, 0.25, 0)
    over = cv2.addWeighted(over, 0.85, win_col, 0.35, 0)
    cv2.putText(over, "WallLike (red) + Window candidates (yellow)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imwrite(out_path, over)

def save_ckpt(path, model, opt, epoch, best=None):
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(), "best": best}, path)

def load_ckpt(path, model, opt=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[CKPT] Loaded full model. missing={len(msg.missing_keys)}")
    if opt is not None and "opt_state" in ckpt:
        opt.load_state_dict(ckpt["opt_state"])
        print("[CKPT] Loaded optimizer state.")
    return ckpt.get("epoch", 0)

def load_backbone_only(model, path, device="cpu"):
    """
    Robustly load ONLY the Hourglass backbone weights into:
        model.backbone.body

    Works for checkpoints saved as:
    - full SegmentationHead state_dict (keys like 'backbone.body.*')
    - DataParallel ('module.backbone.body.*')
    - raw hourglass state_dict (keys like '*', already matching body)
    - wrapper checkpoints with {"model_state": ...}
    """
    print(f"[LOADER] Loading backbone weights from {path}")
    ckpt = torch.load(path, map_location=device)

    # unwrap common formats
    sd = ckpt.get("model_state", ckpt)
    if not isinstance(sd, dict):
        raise RuntimeError(f"[LOADER] Unexpected checkpoint format at {path}")

    # strip DataParallel prefix
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    body_sd = model.backbone.body.state_dict()

    # helper: try mapping from checkpoint key -> body key
    def maybe_strip_prefix(k: str) -> str:
        # most common cases
        prefixes = [
            "backbone.body.",
            "body.",
            "model.backbone.body.",
            "net.backbone.body.",
        ]
        for p in prefixes:
            if k.startswith(p):
                return k[len(p):]
        return k

    filtered = {}
    for k, v in sd.items():
        bk = maybe_strip_prefix(k)
        if bk in body_sd and body_sd[bk].shape == v.shape:
            filtered[bk] = v

    msg = model.backbone.body.load_state_dict(filtered, strict=False)

    print(f"[LOADER] Backbone loaded: {len(filtered)} tensors. Missing={len(msg.missing_keys)}")
    if len(filtered) == 0:
        # Print representative keys for checkpoint-format debugging.
        some_keys = list(sd.keys())[:10]
        some_body = list(body_sd.keys())[:10]
        print("[LOADER][DEBUG] Example ckpt keys:", some_keys)
        print("[LOADER][DEBUG] Example body keys:", some_body)
        raise RuntimeError(
            "[LOADER] Loaded 0 tensors into backbone. "
            "Your checkpoint key prefixes do not match any supported pattern above."
        )


# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = ForensicsDataset(args.data_root, "train", args.img_size)
    val_ds   = ForensicsDataset(args.data_root, "val", args.img_size)
    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty.")

    # loaders
    if args.mode == "w2":
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
    elif args.mode == "w3d":
        sampler = DoorAwareSampler(train_ds, batch_size=args.batch, min_door_px=50, door_ratio=0.5)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        sampler = WindowAwareSampler(train_ds, batch_size=args.batch, window_ratio=0.5)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # modes
    if args.mode == "w2":
        num_classes, class_names = 2, W2_CLASS_NAMES
    elif args.mode == "w3d":
        num_classes, class_names = 3, W3D_CLASS_NAMES
    elif args.mode == "ww3":
        num_classes, class_names = 3, WW3_CLASS_NAMES
    else:
        num_classes, class_names = 6, CLASS_NAMES

    model = SegmentationHead(None, num_classes=num_classes).to(device)
    tracker = MetricTracker(num_classes=num_classes, class_names=class_names)

    # load backbone init
    if args.pretrained_backbone:
        load_backbone_only(model, args.pretrained_backbone, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accum = args.accum
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)

    # optional full checkpoint resume
    start_epoch = 0
    if args.pretrained:
        start_epoch = load_ckpt(args.pretrained, model, opt, device=device)

    # BEST tracking
    best_score = -1e9
    best_epoch = -1

    def metric_for_best(ious_dict):
        # Choose what "best" means per mode
        if args.mode == "w2":
            return float(ious_dict.get("WallLike", 0.0))
        if args.mode == "w3d":
            return float(ious_dict.get("Door", 0.0))
        if args.mode == "ww3":
            return float(ious_dict.get("Window", 0.0))
        # full6: prioritize door quality when selecting the best checkpoint.
        return float(ious_dict.get("Door", 0.0))

    print(f"--- STARTING ({args.mode}) ---")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        opt.zero_grad(set_to_none=True)

        # class weights
        if args.mode == "w2":
            cw = torch.tensor([1.0, 2.0]).to(device)
        elif args.mode == "w3d":
            cw = torch.tensor([1.0, 2.0, 6.0]).to(device)  # door heavier
        elif args.mode == "ww3":
            cw = torch.tensor([0.1, 10.0, 15.0]).to(device)
        else:
            cw = torch.tensor([0.05, 0.5, 20.0, 6.0, 2.0, 12.0]).to(device)

        for step, batch in enumerate(pbar, start=1):
            img = batch["image"].to(device)
            label_raw = batch["label"].clamp(0, 5).to(device)

            if args.mode == "w2":
                label = remap_w2(label_raw)
            elif args.mode == "w3d":
                label = remap_w3d(label_raw)
            elif args.mode == "ww3":
                label = remap_ww3(label_raw)
            else:
                label = label_raw

            logits = model(img)
            ignore = batch["ignore"].to(device)

            loss_map = F.cross_entropy(logits, label, weight=cw, reduction="none")
            if (~ignore).any():
                loss = loss_map[~ignore].mean()
            else:
                loss = loss_map.mean()

            (loss / accum).backward()

            if (step % accum == 0) or (step == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------------- EVAL ----------------
        model.eval()
        tracker.reset()

        # save more samples every epoch
        save_left = int(getattr(args, "save_vis_n", 8))  # default 8 if not provided
        saved = 0

        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                img = batch["image"].to(device)
                lbl_raw = batch["label"].clamp(0, 5).to(device)

                if args.mode == "w2":
                    lbl = remap_w2(lbl_raw)
                elif args.mode == "w3d":
                    lbl = remap_w3d(lbl_raw)
                elif args.mode == "ww3":
                    lbl = remap_ww3(lbl_raw)
                else:
                    lbl = lbl_raw

                logits = model(img)
                tracker.update(logits, lbl)

                # save N images from val (more than just first batch)
                if saved < save_left:
                    preds = torch.argmax(logits, 1)
                    bs = img.shape[0]
                    for k in range(bs):
                        if saved >= save_left:
                            break
                        outp = f"{args.out_dir}/ep{epoch:02d}_val_{saved:02d}.jpg"
                        save_vis(img[k], preds[k], lbl[k], outp, mode=args.mode)
                        saved += 1

        acc, mIoU, ious = tracker.summary()

        if args.mode == "w2":
            print(f"Ep {epoch}: mIoU={mIoU:.3f} | WallLike={ious['WallLike']:.3f}")
            sched.step(ious["WallLike"])
        elif args.mode == "w3d":
            print(f"Ep {epoch}: mIoU={mIoU:.3f} | WallLike={ious['WallLike']:.3f} | Door={ious['Door']:.3f}")
            sched.step(ious["Door"])
        elif args.mode == "ww3":
            print(f"Ep {epoch}: mIoU={mIoU:.3f} | Wall={ious['Wall']:.3f} | Win={ious['Window']:.3f}")
            sched.step(ious["Window"])
        else:
            # full6 prints only if keys exist
            w = ious.get("Wall", 0.0)
            d = ious.get("Door", 0.0)
            win = ious.get("Window", 0.0)
            print(f"Ep {epoch}: mIoU={mIoU:.3f} | Wall={w:.3f} | Win={win:.3f} | Door={d:.3f}")
            sched.step(d)

        print(f"[LR] {opt.param_groups[0]['lr']:.2e}")

        # ---------------- SAVE CKPTS ----------------
        ckpt_name = (
            "stage0_w2_walllike.pth" if args.mode == "w2"
            else ("stage1_w3d_walllike_door.pth" if args.mode == "w3d"
                  else ("stageA_ww3.pth" if args.mode == "ww3"
                        else "stageB_full6.pth"))
        )

        # always save last
        save_ckpt(f"{args.out_dir}/{ckpt_name}", model, opt, epoch)
        save_ckpt(f"{args.out_dir}/last.pth", model, opt, epoch)

        # save best
        score = metric_for_best(ious)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_ckpt(f"{args.out_dir}/best.pth", model, opt, epoch, best={"score": best_score, "epoch": best_epoch})
            print(f"[BEST] Updated best.pth at epoch {best_epoch} with score={best_score:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument("--out-dir", default="./runs_forensics")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--mode", choices=["w2", "w3d", "ww3", "full6"], default="full6")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-vis-n", type=int, default=8)
    args = parser.parse_args()
    train_loop(args)
