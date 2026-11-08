"""Geometric helpers: corner ordering and perspective (homography) warping."""

from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    - top-left has the smallest (x + y), bottom-right the largest.
    - top-right has the smallest (y - x), bottom-left the largest.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    d = np.diff(pts, axis=1).ravel()  # y - x
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral `pts` in `image` to a flat, top-down rectangle.

    The output size is derived from the longest opposite edges so that the
    document keeps its real aspect ratio.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    # 3x3 homography matrix M such that dst ~ M * rect
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))
