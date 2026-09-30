import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parents[2]
ENGINE_DIR = PROJECT_DIR / "tools" / "realesrgan"
EXECUTABLE = ENGINE_DIR / "realesrgan-ncnn-vulkan.exe"
MODEL_DIR = ENGINE_DIR / "models"
MODEL_NAME = "realesrgan-x4plus"
TIMEOUT_SECONDS = 180

# realesrgan-x4plus preserves composition at its native 4x output. Smaller
# levels are produced by downsampling that full-frame result with Pillow.
LEVEL_SCALES = {1: 2, 2: 3, 3: 4, 4: 4}


class EnhancementError(Exception):
    pass


def enhance_image(contents: bytes, level: int) -> tuple[bytes, int, int]:
    if not EXECUTABLE.is_file():
        raise EnhancementError(
            "Real-ESRGAN is not configured. Place realesrgan-ncnn-vulkan.exe in tools/realesrgan/."
        )
    if not (MODEL_DIR / f"{MODEL_NAME}.param").is_file() or not (MODEL_DIR / f"{MODEL_NAME}.bin").is_file():
        raise EnhancementError(
            f"Real-ESRGAN model is missing. Place {MODEL_NAME}.param and {MODEL_NAME}.bin in tools/realesrgan/models/."
        )

    scale = LEVEL_SCALES[level]
    try:
        with tempfile.TemporaryDirectory(prefix="enhancer-") as temporary_dir:
            temp_path = Path(temporary_dir)
            input_path = temp_path / "input.png"
            output_path = temp_path / "output.png"
            with Image.open(BytesIO(contents)) as source:
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                original_width, original_height = normalized.size
                normalized.save(input_path, format="PNG")

            command = [
                str(EXECUTABLE), "-i", str(input_path), "-o", str(output_path),
                "-n", MODEL_NAME, "-m", str(MODEL_DIR), "-s", "4", "-f", "png",
            ]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False
                )
            except subprocess.TimeoutExpired as error:
                raise EnhancementError("Real-ESRGAN timed out. Try a smaller image or lower enhancement level.") from error
            except OSError as error:
                raise EnhancementError("Could not start the Real-ESRGAN executable.") from error

            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise EnhancementError(f"Real-ESRGAN failed: {detail[-500:] or 'unknown error'}")
            if not output_path.is_file():
                raise EnhancementError("Real-ESRGAN did not produce an output image.")

            output = output_path.read_bytes()
            try:
                with Image.open(BytesIO(output)) as result_image:
                    result_image.verify()
                with Image.open(BytesIO(output)) as result_image:
                    width, height = result_image.size
                    if (width, height) != (original_width * 4, original_height * 4):
                        raise EnhancementError("Real-ESRGAN produced unexpected output dimensions.")
                    if scale != 4:
                        resized = result_image.resize(
                            (original_width * scale, original_height * scale),
                            Image.Resampling.LANCZOS,
                        )
                        resized_output = BytesIO()
                        resized.save(resized_output, format="PNG")
                        output = resized_output.getvalue()
                        width, height = resized.size
            except (UnidentifiedImageError, OSError, ValueError) as error:
                raise EnhancementError("Real-ESRGAN produced an invalid image.") from error
            return output, width, height
    except EnhancementError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise EnhancementError("Could not prepare the image for Real-ESRGAN.") from error
