"""Generate synthetic 'phone photos' of documents for demos and tests.

A flat document with text is drawn, then warped with a random perspective
onto a cluttered background - mimicking a skewed mobile capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_document(w: int = 850, h: int = 1100) -> np.ndarray:
    doc = np.full((h, w, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(doc, "SMART DOCUMENT SCANNER", (60, 110), font, 1.4, (20, 20, 20), 3)
    cv2.line(doc, (60, 140), (w - 60, 140), (40, 40, 40), 2)
    lines = [
        "Image Processing - Course Project",
        "Pipeline: Grayscale -> Gaussian -> Canny",
        "          -> Contours -> Perspective Warp",
        "",
        "Invoice No: 2026-0417",
        "Item                 Qty        Price",
        "Notebook              3         45.00",
        "Pencil set            1         30.00",
        "Total                           75.00",
    ]
    for i, text in enumerate(lines):
        cv2.putText(doc, text, (60, 220 + i * 60), font, 0.9, (30, 30, 30), 2)
    for i in range(6):
        y = 800 + i * 40
        cv2.line(doc, (60, y), (w - 60 - (i % 3) * 90, y), (90, 90, 90), 3)
    return doc


def make_background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array(rng.integers(40, 110, size=3), dtype=np.uint8)
    bg = np.tile(base, (h, w, 1)).astype(np.int16)
    # wood-like stripes + noise
    stripes = (np.sin(np.linspace(0, 40, h))[:, None] * 12).astype(np.int16)
    bg += stripes[..., None]
    bg += rng.integers(-10, 10, size=bg.shape, dtype=np.int16)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    # clutter: a "mug" and a "pen"
    cv2.circle(bg, (int(w * 0.88), int(h * 0.15)), int(min(w, h) * 0.07), (30, 30, 160), -1)
    cv2.line(bg, (int(w * 0.05), int(h * 0.9)), (int(w * 0.2), int(h * 0.97)), (200, 180, 20), 8)
    return bg


def make_photo(seed: int, out_w: int = 1200, out_h: int = 1600):
    rng = np.random.default_rng(seed)
    doc = make_document()
    dh, dw = doc.shape[:2]
    bg = make_background(out_w, out_h, rng)

    cx, cy = out_w / 2, out_h / 2
    hw, hh = out_w * 0.32, out_h * 0.33
    jitter = lambda: rng.uniform(-0.12, 0.12) * out_w  # noqa: E731
    dst = np.array([
        [cx - hw + jitter(), cy - hh + jitter()],
        [cx + hw + jitter(), cy - hh + jitter()],
        [cx + hw + jitter(), cy + hh + jitter()],
        [cx - hw + jitter(), cy + hh + jitter()],
    ], dtype=np.float32)
    src = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, M, (out_w, out_h))
    mask = cv2.warpPerspective(np.full((dh, dw), 255, np.uint8), M, (out_w, out_h))
    photo = np.where(mask[..., None] > 0, warped, bg)

    # uneven lighting
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    shade = 0.75 + 0.25 * (xx / out_w)
    photo = np.clip(photo * shade[..., None], 0, 255).astype(np.uint8)
    return photo, dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", type=Path, default=Path(__file__).parent / "input")
    p.add_argument("-n", "--count", type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        photo, _ = make_photo(seed=i)
        path = args.output / f"sample_{i + 1}.jpg"
        cv2.imwrite(str(path), photo)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
