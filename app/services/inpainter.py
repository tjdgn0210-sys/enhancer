from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps


MODEL_PATH = Path(__file__).resolve().parents[2] / "tools" / "lama" / "inpainting_lama_2025jan.onnx"
MODEL_SIZE = 512
# (long-edge scale, minimum px, maximum px) for the final blend only.
BLEND_FEATHER = {1: (0.003, 2, 3), 2: (0.005, 4, 6), 3: (0.008, 6, 10)}


class InpainterError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_session() -> ort.InferenceSession:
    if not MODEL_PATH.is_file():
        raise InpainterError("LaMa model is missing from tools/lama.")
    try:
        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        input_names = {item.name for item in session.get_inputs()}
        output_names = {item.name for item in session.get_outputs()}
        if input_names != {"image", "mask"} or "output" not in output_names:
            raise ValueError("Unexpected LaMa model inputs or outputs.")
        return session
    except Exception as error:
        raise InpainterError("The local LaMa model could not be loaded.") from error


def inpaint_image(image: Image.Image, mask: Image.Image | bytes, cleanup_level: int = 2) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(mask, bytes):
        try:
            with Image.open(BytesIO(mask)) as mask_image:
                removal_mask = mask_image.convert("L")
        except (OSError, ValueError) as error:
            raise InpainterError("The removal mask is invalid.") from error
    else:
        removal_mask = mask.convert("L")
    if removal_mask.size != source.size:
        raise InpainterError("Image and removal mask dimensions do not match.")

    try:
        width, height = source.size
        scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized_image = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_mask = removal_mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
        left = (MODEL_SIZE - resized_width) // 2
        top = (MODEL_SIZE - resized_height) // 2
        right = MODEL_SIZE - resized_width - left
        bottom = MODEL_SIZE - resized_height - top

        image_array = np.asarray(resized_image, dtype=np.uint8)[:, :, ::-1]
        padded_image = np.pad(image_array, ((top, bottom), (left, right), (0, 0)), mode="edge")
        mask_array = (np.asarray(resized_mask, dtype=np.uint8) > 0).astype(np.float32)
        padded_mask = np.pad(mask_array, ((top, bottom), (left, right)), mode="constant")
        image_tensor = np.transpose(padded_image.astype(np.float32) / 255.0, (2, 0, 1))[None]
        mask_tensor = padded_mask[None, None]

        output = _get_session().run(["output"], {"image": image_tensor, "mask": mask_tensor})[0]
        if output.shape != (1, 3, MODEL_SIZE, MODEL_SIZE) or not np.isfinite(output).all():
            raise ValueError("LaMa returned an invalid output tensor.")
        output_bgr = np.clip(output[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        output_rgb = output_bgr[:, :, ::-1]
        restored = Image.fromarray(output_rgb).crop((left, top, left + resized_width, top + resized_height))
        restored = restored.resize(source.size, Image.Resampling.BILINEAR)
        binary_blend_mask = removal_mask.point(lambda value: 255 if value > 0 else 0)
        ratio, minimum, maximum = BLEND_FEATHER[cleanup_level]
        feather_radius = max(minimum, min(maximum, round(max(width, height) * ratio)))
        blend_mask = binary_blend_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        return Image.composite(restored, source, blend_mask)
    except InpainterError:
        raise
    except Exception as error:
        raise InpainterError("LaMa could not process this image.") from error
