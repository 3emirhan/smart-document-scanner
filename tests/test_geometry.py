import numpy as np

from scanner.geometry import four_point_transform, order_points


def test_order_points_shuffled():
    pts = np.array([[100, 10], [10, 90], [10, 10], [110, 100]], dtype=np.float32)
    rect = order_points(pts)
    np.testing.assert_array_equal(rect, [[10, 10], [100, 10], [110, 100], [10, 90]])


def test_four_point_transform_output_size():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    pts = np.array([[50, 50], [250, 50], [250, 150], [50, 150]], dtype=np.float32)
    out = four_point_transform(img, pts)
    assert out.shape[:2] == (100, 200)


def test_four_point_transform_identity_content():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 255
    pts = np.array([[20, 20], [79, 20], [79, 79], [20, 79]], dtype=np.float32)
    out = four_point_transform(img, pts)
    assert out.mean() > 250
