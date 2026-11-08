"""Smart Document Scanner - automated perspective correction & edge detection."""

from .geometry import order_points, four_point_transform
from .pipeline import DocumentScanner, ScanResult, DocumentNotFoundError

__all__ = [
    "DocumentScanner",
    "ScanResult",
    "DocumentNotFoundError",
    "order_points",
    "four_point_transform",
]
__version__ = "1.0.0"
