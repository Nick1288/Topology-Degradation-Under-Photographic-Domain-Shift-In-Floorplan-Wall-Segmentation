from __future__ import annotations

import cv2
import numpy as np


def preprocess_contrast(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_deskew(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None or len(lines) == 0:
        return img

    angles = [line[0][1] for line in lines[:20]]
    median_angle = np.median(angles) - np.pi / 2
    if abs(median_angle) >= 0.3:
        return img

    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), np.degrees(median_angle), 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
