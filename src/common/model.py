from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from floortrans.models import get_model


class FeatureExtractor2(nn.Module):
    """Expose shallow and deep 256-channel features from the CubiCasa backbone."""

    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.body = original_model
        self.feat_deep = None
        self.feat_shallow = None

        convs = [(name, module) for name, module in self.body.named_modules() if isinstance(module, nn.Conv2d)]
        shallow = next(((name, module) for name, module in convs if module.out_channels == 256), None)
        deep = next(((name, module) for name, module in reversed(convs) if module.out_channels == 256), None)
        if shallow is None or deep is None:
            raise RuntimeError("Could not find shallow/deep 256-channel convolution layers.")

        shallow[1].register_forward_hook(self._hook_shallow)
        deep[1].register_forward_hook(self._hook_deep)

    def _hook_shallow(self, _module, _inputs, output):
        self.feat_shallow = output

    def _hook_deep(self, _module, _inputs, output):
        self.feat_deep = output

    def forward(self, x: torch.Tensor):
        self.feat_shallow = None
        self.feat_deep = None
        _ = self.body(x)
        if self.feat_shallow is None or self.feat_deep is None:
            raise RuntimeError("Feature hooks did not capture backbone outputs.")
        return self.feat_shallow, self.feat_deep


class SegmentationHead(nn.Module):
    """Fine-tuning head used by finetune_cubicasa_photos.py and thesis experiments."""

    def __init__(self, num_classes: int = 3):
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

        self.shallow_proj = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor):
        feat_shallow, feat_deep = self.backbone(x)
        x1 = self.b1(feat_deep)
        x2 = self.b2(feat_deep)
        x3 = self.b3(feat_deep)
        deep = torch.cat([x1, x2, x3], dim=1)
        deep = self.relu(self.bn(deep))
        deep = self.relu(self.bn_proj(self.project(deep)))
        shallow = self.shallow_proj(feat_shallow)
        fused = torch.cat([deep, shallow], dim=1)
        fused = self.fuse(fused)
        return self.up(fused)


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    message = model.load_state_dict(state_dict, strict=strict)
    if message.missing_keys:
        print(f"[WARN] Missing checkpoint keys: {len(message.missing_keys)}")
    if message.unexpected_keys:
        print(f"[WARN] Unexpected checkpoint keys: {len(message.unexpected_keys)}")
    print(f"[OK] Loaded checkpoint: {ckpt_path}")
    return message


def load_backbone_only(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

    prefixes = ("backbone.body.", "body.", "model.backbone.body.", "net.backbone.body.")
    body_state = model.backbone.body.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        body_key = next((key[len(prefix):] for prefix in prefixes if key.startswith(prefix)), key)
        if body_key in body_state and body_state[body_key].shape == value.shape:
            filtered[body_key] = value

    message = model.backbone.body.load_state_dict(filtered, strict=False)
    print(f"[OK] Loaded {len(filtered)} backbone tensors from {ckpt_path}")
    if not filtered:
        raise RuntimeError(f"Loaded 0 tensors into backbone from {ckpt_path}")
    return message


def bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0)


def run_segmentation(img_bgr: np.ndarray, model: nn.Module, device: torch.device, img_size: int):
    img_resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = bgr_to_tensor(img_resized).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    wall = (pred == 1).astype(np.uint8)
    door = (pred == 2).astype(np.uint8)
    return wall, door

