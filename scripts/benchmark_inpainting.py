"""Run the production LaMa inpainter on an image and binary mask."""

import argparse
import csv
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
SUMMARY_COLUMNS = (
    "case_name",
    "width",
    "height",
    "mask_area_percent",
    "lama_seconds",
    "telea_seconds",
    "navier_stokes_seconds",
)


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


def run_case(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    cleanup_level: int,
    opencv_radius: float,
    manual_preference: str | None = None,
) -> dict[str, int | float | str]:
    try:
        with Image.open(image_path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read image or mask: {error}") from error

    if mask.size != original.size:
        raise ValueError(
            f"Image/mask size mismatch: image is {original.width}x{original.height}, "
            f"mask is {mask.width}x{mask.height}."
        )

    binary_mask = mask.point(lambda value: 255 if value >= 128 else 0)
    masked_ratio, mask_aspect_ratio, ring_variation = mask_features(original, binary_mask)
    selected_method = "telea" if masked_ratio <= 0.10 else "lama"
    output_dir.mkdir(parents=True, exist_ok=True)

    original.save(output_dir / "original.png")
    binary_mask.save(output_dir / "mask.png")
    start = perf_counter()
    result = inpaint_image(original, binary_mask, cleanup_level)
    lama_elapsed = perf_counter() - start

    if result.size != original.size:
        raise ValueError("LaMa output dimensions do not match the original image.")
    result.save(output_dir / "lama.png")

    original_bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    mask_array = np.asarray(binary_mask, dtype=np.uint8)
    opencv_outputs = {}
    opencv_times = {}
    for method_name, method in (
        ("telea", cv2.INPAINT_TELEA),
        ("navier_stokes", cv2.INPAINT_NS),
    ):
        start = perf_counter()
        output_bgr = cv2.inpaint(original_bgr, mask_array, opencv_radius, method)
        opencv_times[method_name] = perf_counter() - start
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        output = Image.fromarray(output_rgb)
        output.save(output_dir / f"{method_name}.png")
        opencv_outputs[method_name] = output

    panels = (("Original", original), ("LaMa", result)) + tuple(
        (name.replace("_", " ").title(), output)
        for name, output in opencv_outputs.items()
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
    comparison.save(output_dir / "comparison.png")

    print(f"Image size: {original.width}x{original.height}")
    print(f"Cleanup level: {cleanup_level}")
    print(f"Masked area: {masked_ratio:.2%}")
    print(f"Mask bounding-box aspect ratio (width/height): {mask_aspect_ratio:.3f}")
    print(f"Local ring brightness variation (grayscale std dev): {ring_variation:.3f}")
    print("Benchmark-only candidate: area <= 10% selects Telea; otherwise LaMa")
    print(f"Candidate-selected method: {selected_method}")
    for threshold in AREA_THRESHOLDS:
        threshold_method = "telea" if masked_ratio <= threshold else "lama"
        if manual_preference is None:
            agreement = "manual preference not provided"
        elif manual_preference == "mixed":
            agreement = "unclear (excluded from agreement count)"
        else:
            agreement = "matches" if threshold_method == manual_preference else "does not match"
        print(f"Area threshold {threshold:.1%}: {threshold_method} -> {agreement}")
    print(f"LaMa time: {lama_elapsed:.3f} seconds")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV radius: {opencv_radius:g} px")
    for method_name, elapsed in opencv_times.items():
        print(f"{method_name} time: {elapsed:.3f} seconds")
    print(f"Output directory: {output_dir.resolve()}")

    return {
        "case_name": image_path.parent.name,
        "width": original.width,
        "height": original.height,
        "mask_area_percent": masked_ratio * 100,
        "lama_seconds": lama_elapsed,
        "telea_seconds": opencv_times["telea"],
        "navier_stokes_seconds": opencv_times["navier_stokes"],
    }


def print_summary(rows: list[dict[str, int | float | str]]) -> None:
    headings = ("Case", "Dimensions", "Mask %", "LaMa s", "Telea s", "NS s")
    if not rows:
        print("\nBenchmark summary")
        print(" | ".join(headings))
        print("No valid cases to summarize.")
        return
    formatted_rows = [
        (
            str(row["case_name"]),
            f"{row['width']}x{row['height']}",
            f"{row['mask_area_percent']:.2f}",
            f"{row['lama_seconds']:.3f}",
            f"{row['telea_seconds']:.3f}",
            f"{row['navier_stokes_seconds']:.3f}",
        )
        for row in rows
    ]
    widths = [max(len(headings[i]), *(len(row[i]) for row in formatted_rows)) for i in range(len(headings))]
    print("\nBenchmark summary")
    print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(headings)))
    print("-+-".join("-" * width for width in widths))
    for row in formatted_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark LaMa and OpenCV inpainting.")
    parser.add_argument("image", nargs="?", type=Path, help="Input image path")
    parser.add_argument("mask", nargs="?", type=Path, help="Binary removal mask path")
    parser.add_argument("--dataset", type=Path, help="Directory containing case subdirectories")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/run"),
        help="Output directory (batch mode creates one subdirectory per case)",
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
        if args.dataset:
            if args.image or args.mask:
                parser.error("Use either --dataset or positional image and mask paths, not both.")
            if not args.dataset.is_dir():
                parser.error(f"Dataset directory does not exist: {args.dataset}")

            rows = []
            failed_cases = 0
            for case_dir in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
                image_path = case_dir / "image.png"
                mask_path = case_dir / "mask.png"
                if not image_path.is_file() or not mask_path.is_file():
                    missing = [name for name, path in (("image.png", image_path), ("mask.png", mask_path)) if not path.is_file()]
                    print(f"ERROR [{case_dir.name}]: missing {', '.join(missing)}")
                    failed_cases += 1
                    continue
                try:
                    rows.append(
                        run_case(
                            image_path,
                            mask_path,
                            args.output_dir / case_dir.name,
                            args.cleanup_level,
                            args.opencv_radius,
                        )
                    )
                except (OSError, ValueError, InpainterError, cv2.error) as error:
                    print(f"ERROR [{case_dir.name}]: {error}")
                    failed_cases += 1

            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
                writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            print_summary(rows)
            print(f"Summary CSV: {(args.output_dir / 'summary.csv').resolve()}")
            return 1 if not rows and failed_cases else 0

        if args.image is None or args.mask is None:
            parser.error("Provide image and mask paths, or use --dataset.")
        run_case(
            args.image,
            args.mask,
            args.output_dir,
            args.cleanup_level,
            args.opencv_radius,
            args.manual_preference,
        )
        return 0
    except (OSError, ValueError, InpainterError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
