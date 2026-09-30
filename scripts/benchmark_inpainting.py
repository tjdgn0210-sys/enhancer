"""Run the production LaMa inpainter on an image and binary mask."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.services.inpainter import InpainterError, inpaint_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the production LaMa inpainting path.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("mask", type=Path, help="Binary removal mask path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/run"),
        help="Directory for original, mask, LaMa, and comparison PNGs",
    )
    parser.add_argument("--cleanup-level", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()

    try:
        with Image.open(args.image) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(args.mask) as source_mask:
            mask = source_mask.convert("L")
    except (OSError, ValueError) as error:
        parser.error(f"Could not read input image or mask: {error}")

    if mask.size != original.size:
        parser.error(f"Image and mask dimensions must match ({original.width}x{original.height}).")

    binary_mask = mask.point(lambda value: 255 if value >= 128 else 0)
    masked_pixels = sum(binary_mask.histogram()[1:])
    masked_ratio = masked_pixels / (original.width * original.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original.save(args.output_dir / "original.png")
    binary_mask.save(args.output_dir / "mask.png")
    start = perf_counter()
    try:
        result = inpaint_image(original, binary_mask, args.cleanup_level)
    except InpainterError as error:
        parser.error(str(error))
    elapsed = perf_counter() - start

    if result.size != original.size:
        parser.error("LaMa output dimensions do not match the original image.")
    result.save(args.output_dir / "lama.png")

    comparison = Image.new("RGB", (original.width * 2, original.height), "white")
    comparison.paste(original, (0, 0))
    comparison.paste(result, (original.width, 0))
    comparison.save(args.output_dir / "comparison.png")

    print(f"Image size: {original.width}x{original.height}")
    print(f"Cleanup level: {args.cleanup_level}")
    print(f"Masked area: {masked_ratio:.2%}")
    print(f"Processing time: {elapsed:.3f} seconds")
    print(f"Output directory: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
