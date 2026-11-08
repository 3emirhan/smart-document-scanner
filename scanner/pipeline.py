"""The 5-stage document scanning pipeline.

1. Grayscale conversion
2. Gaussian filtering
3. Canny edge detection
4. Contour extraction
5. Perspective warping
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import four_point_transform, order_points


class DocumentNotFoundError(RuntimeError):
    """Raised when no document-like quadrilateral can be found."""


@dataclass
class ScanResult:
    warped: np.ndarray            # perspective-corrected color image
    scanned: np.ndarray           # final output (color or "scanned" B/W)
    corners: np.ndarray           # 4x2 corners in original image coordinates
    steps: dict = field(default_factory=dict)  # intermediate images for debug


class DocumentScanner:
    def __init__(
        self,
        process_height: int = 500,
        blur_kernel: int = 5,
        canny_low: int = 75,
        canny_high: int = 200,
        approx_epsilon: float = 0.02,
        min_area_ratio: float = 0.10,
        enhance: bool = True,
    ) -> None:
        if blur_kernel % 2 == 0 or blur_kernel < 1:
            raise ValueError("blur_kernel must be a positive odd integer")
        self.process_height = process_height
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.approx_epsilon = approx_epsilon
        self.min_area_ratio = min_area_ratio
        self.enhance = enhance

    # ---- Stage 1 ---------------------------------------------------------
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- Stage 2 ---------------------------------------------------------
    def gaussian_filter(self, gray: np.ndarray) -> np.ndarray:
        k = self.blur_kernel
        return cv2.GaussianBlur(gray, (k, k), 0)

    # ---- Stage 3 ---------------------------------------------------------
    def detect_edges(self, blurred: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        # Close small gaps so the document outline forms a single contour.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---- Stage 4 ---------------------------------------------------------
    def find_document_contour(self, edges: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise DocumentNotFoundError("No contours found in the image.")

        min_area = self.min_area_ratio * edges.shape[0] * edges.shape[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_epsilon * peri, True)
            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
                and cv2.contourArea(approx) >= min_area
            ):
                return approx.reshape(4, 2).astype(np.float32)

        # Fallback: minimum-area rectangle around the largest contour.
        largest = contours[0]
        if cv2.contourArea(cv2.convexHull(largest)) < min_area:
            raise DocumentNotFoundError(
                "Could not find a document outline. Try a photo with better "
                "contrast between the document and the background."
            )
        box = cv2.boxPoints(cv2.minAreaRect(largest))
        return box.astype(np.float32)

    # ---- Stage 5 ---------------------------------------------------------
    @staticmethod
    def warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        return four_point_transform(image, corners)

    # ---- Optional post-processing ---------------------------------------
    @staticmethod
    def scan_effect(warped: np.ndarray) -> np.ndarray:
        """Make the result look like a hardware scanner output (clean B/W)."""
        gray = DocumentScanner.to_grayscale(warped)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )

    # ---- Full pipeline --------------------------------------------------
    def scan(self, image: np.ndarray) -> ScanResult:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        # Work on a downscaled copy for speed & stable thresholds.
        ratio = image.shape[0] / float(self.process_height)
        width = max(1, int(round(image.shape[1] / ratio)))
        small = cv2.resize(image, (width, self.process_height),
                           interpolation=cv2.INTER_AREA)

        gray = self.to_grayscale(small)
        blurred = self.gaussian_filter(gray)
        edges = self.detect_edges(blurred)
        corners_small = self.find_document_contour(edges)

        corners = order_points(corners_small * ratio)
        warped = self.warp(image, corners)
        scanned = self.scan_effect(warped) if self.enhance else warped

        outline = small.copy() if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        cv2.polylines(outline, [corners_small.astype(np.int32)], True, (0, 255, 0), 2)
        for x, y in corners_small.astype(int):
            cv2.circle(outline, (x, y), 6, (0, 0, 255), -1)

        steps = {
            "1_grayscale": gray,
            "2_gaussian": blurred,
            "3_canny": edges,
            "4_contour": outline,
            "5_warped": warped,
        }
        return ScanResult(warped=warped, scanned=scanned, corners=corners, steps=steps)
