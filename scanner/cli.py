"""Command-line interface: python -m scanner <image> [options]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .pipeline import DocumentNotFoundError, DocumentScanner

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scanner",
        description="Turn skewed phone photos of documents into flat, scanner-like images.",
    )
    p.add_argument("input", type=Path, help="Image file or folder of images")
    p.add_argument("-o", "--output", type=Path, default=Path("output"),
                   help="Output folder (default: ./output)")
    p.add_argument("--color", action="store_true",
                   help="Keep color output instead of black & white scan effect")
    p.add_argument("--debug", action="store_true",
                   help="Also save every intermediate pipeline step")
    p.add_argument("--show", action="store_true",
                   help="Display results in a window (needs a GUI)")
    p.add_argument("--canny", nargs=2, type=int, metavar=("LOW", "HIGH"),
                   default=(75, 200), help="Canny thresholds (default: 75 200)")
    p.add_argument("--blur", type=int, default=5,
                   help="Gaussian kernel size, odd number (default: 5)")
    return p


def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [path]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scanner = DocumentScanner(
        blur_kernel=args.blur,
        canny_low=args.canny[0],
        canny_high=args.canny[1],
        enhance=not args.color,
    )

    files = collect_images(args.input)
    if not files:
        print(f"No images found at {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for f in files:
        image = cv2.imread(str(f))
        if image is None:
            print(f"[skip] could not read {f}", file=sys.stderr)
            failures += 1
            continue
        try:
            result = scanner.scan(image)
        except DocumentNotFoundError as e:
            print(f"[fail] {f.name}: {e}", file=sys.stderr)
            failures += 1
            continue

        out_path = args.output / f"{f.stem}_scanned.png"
        cv2.imwrite(str(out_path), result.scanned)
        print(f"[ok]   {f.name} -> {out_path}")

        if args.debug:
            debug_dir = args.output / f"{f.stem}_steps"
            debug_dir.mkdir(exist_ok=True)
            for name, img in result.steps.items():
                cv2.imwrite(str(debug_dir / f"{name}.png"), img)

        if args.show:
            cv2.imshow("Detected", result.steps["4_contour"])
            cv2.imshow("Scanned", result.scanned)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 1 if failures == len(files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
