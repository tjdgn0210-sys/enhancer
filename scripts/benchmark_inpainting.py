"""Run the production LaMa inpainter on an image and binary mask."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.services.inpainter import InpainterError, inpaint_image  # noqa: E402


AREA_THRESHOLDS = (0.05, 0.075, 0.10, 0.125, 0.15)


def mask_features(image: Image.Image, binary_mask: Image.Image) -> tuple[float, float, float]:
    """Return masked-area ratio, mask bounding-box aspect ratio, and ring brightness variation."""
    mask_array = np.asarray(binary_mask, dtype=np.uint8)
    mask_pixels = mask_array > 0
    height, width = mask_array.shape
    area_ratio = float(mask_pixels.mean())
    x, y, box_width, box_height = cv2.boundingRect(mask_array)
    aspect_ratio = box_width / box_height if box_height else 0.0

    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded = cv2.dilate(mask_array, ring_kernel) > 0
    ring_pixels = expanded & ~mask_pixels
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    ring_variation = float(np.std(gray[ring_pixels])) if np.any(ring_pixels) else 0.0
    return area_ratio, aspect_ratio, ring_variation


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
    parser.add_argument("--opencv-radius", type=float, choices=(2, 3, 5), default=3)
    parser.add_argument(
        "--manual-preference",
        choices=("lama", "telea", "mixed"),
        help="Observed better result for this case; mixed cases are excluded from agreement counts",
    )
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
    masked_ratio, mask_aspect_ratio, ring_variation = mask_features(original, binary_mask)
    selected_method = "telea" if masked_ratio <= 0.10 else "lama"
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

    original_bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    mask_array = np.asarray(binary_mask, dtype=np.uint8)
    opencv_outputs = {}
    for method_name, method in (
        ("telea", cv2.INPAINT_TELEA),
        ("navier_stokes", cv2.INPAINT_NS),
    ):
        start = perf_counter()
        output_bgr = cv2.inpaint(original_bgr, mask_array, args.opencv_radius, method)
        opencv_elapsed = perf_counter() - start
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(output_rgb)
        output.save(args.output_dir / f"{method_name}.png")
        opencv_outputs[method_name] = (output, opencv_elapsed)

    panels = (("Original", original), ("LaMa", result)) + tuple(
        (name.replace("_", " ").title(), output)
        for name, (output, _) in opencv_outputs.items()
    )
    label_height = 28
    comparison = Image.new(
        "RGB", (original.width * len(panels), original.height + label_height), "white"
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(comparison)
    for index, (label, panel) in enumerate(panels):
        x = index * original.width
        draw.text((x + 8, 6), label, fill="black")
        comparison.paste(panel, (x, label_height))
    comparison.save(args.output_dir / "comparison.png")

    print(f"Image size: {original.width}x{original.height}")
    print(f"Cleanup level: {args.cleanup_level}")
    print(f"Masked area: {masked_ratio:.2%}")
    print(f"Mask bounding-box aspect ratio (width/height): {mask_aspect_ratio:.3f}")
    print(f"Local ring brightness variation (grayscale std dev): {ring_variation:.3f}")
    print("Benchmark-only candidate: area <= 10% selects Telea; otherwise LaMa")
    print(f"Candidate-selected method: {selected_method}")
    for threshold in AREA_THRESHOLDS:
        threshold_method = "telea" if masked_ratio <= threshold else "lama"
        if args.manual_preference is None:
            agreement = "manual preference not provided"
        elif args.manual_preference == "mixed":
            agreement = "unclear (excluded from agreement count)"
        else:
            agreement = "matches" if threshold_method == args.manual_preference else "does not match"
        print(
            f"Area threshold {threshold:.1%}: {threshold_method} -> {agreement}"
        )
    print(f"LaMa time: {elapsed:.3f} seconds")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV radius: {args.opencv_radius:g} px")
    for method_name, (_, opencv_elapsed) in opencv_outputs.items():
        print(f"{method_name} time: {opencv_elapsed:.3f} seconds")
    print(f"Output directory: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
