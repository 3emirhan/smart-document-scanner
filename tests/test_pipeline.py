import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from generate_samples import make_photo  # noqa: E402

from scanner import DocumentNotFoundError, DocumentScanner, order_points  # noqa: E402


@pytest.mark.parametrize("seed", range(5))
def test_detects_corners_on_synthetic_photo(seed):
    photo, true_corners = make_photo(seed)
    result = DocumentScanner().scan(photo)
    err = np.linalg.norm(result.corners - order_points(true_corners), axis=1)
    # every detected corner within 2% of the image height of the truth
    assert err.max() < 0.02 * photo.shape[0], err


def test_output_is_upright_document():
    photo, _ = make_photo(0)
    result = DocumentScanner().scan(photo)
    h, w = result.warped.shape[:2]
    assert h > w  # portrait page, like the source document
    # the flattened page should be mostly bright paper
    assert result.warped.mean() > 150


def test_steps_are_recorded():
    photo, _ = make_photo(1)
    result = DocumentScanner().scan(photo)
    assert list(result.steps) == [
        "1_grayscale", "2_gaussian", "3_canny", "4_contour", "5_warped",
    ]


def test_blank_image_raises():
    with pytest.raises(DocumentNotFoundError):
        DocumentScanner().scan(np.zeros((600, 400, 3), dtype=np.uint8))


def test_invalid_kernel():
    with pytest.raises(ValueError):
        DocumentScanner(blur_kernel=4)
