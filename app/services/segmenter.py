from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


MODEL_DIR = Path(__file__).resolve().parents[2] / "tools" / "segmentation"
ENCODER_PATH = MODEL_DIR / "edge_sam_3x_encoder.onnx"
DECODER_PATH = MODEL_DIR / "edge_sam_3x_decoder.onnx"
INPUT_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[None, :, None, None]
MAX_POINT_MASK_AREA = 0.40
MAX_COMBINED_MASK_AREA = 0.50


class SegmenterError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_sessions() -> tuple[ort.InferenceSession, ort.InferenceSession]:
    if not ENCODER_PATH.is_file() or not DECODER_PATH.is_file():
        raise SegmenterError("EdgeSAM ONNX models are missing. Run scripts/setup_dev.ps1.")
    try:
        encoder = ort.InferenceSession(str(ENCODER_PATH), providers=["CPUExecutionProvider"])
        decoder = ort.InferenceSession(str(DECODER_PATH), providers=["CPUExecutionProvider"])
        if [item.name for item in encoder.get_inputs()] != ["image"]:
            raise ValueError("Unexpected EdgeSAM encoder inputs.")
        decoder_inputs = {item.name for item in decoder.get_inputs()}
        if decoder_inputs != {"image_embeddings", "point_coords", "point_labels"}:
            raise ValueError("Unexpected EdgeSAM decoder inputs.")
        return encoder, decoder
    except Exception as error:
        raise SegmenterError("The local EdgeSAM ONNX models could not be loaded.") from error


def _resize_shape(width: int, height: int) -> tuple[int, int]:
    scale = INPUT_SIZE / max(width, height)
    return int(width * scale + 0.5), int(height * scale + 0.5)


def _to_original_mask(logits: np.ndarray, input_size: tuple[int, int], original_size: tuple[int, int]) -> np.ndarray:
    resized_width, resized_height = input_size
    width, height = original_size
    mask = Image.fromarray(logits.astype(np.float32), mode="F")
    mask = mask.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    mask = mask.crop((0, 0, resized_width, resized_height))
    mask = mask.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(mask, dtype=np.float32) > 0.0


def segment_points(contents: bytes, points: list[dict[str, int]]) -> dict:
    try:
        with Image.open(BytesIO(contents)) as uploaded:
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
        width, height = image.size
        total_pixels = width * height
        combined_mask = np.zeros((height, width), dtype=bool)
        point_results = []

        if points:
            encoder, decoder = _get_sessions()
            resized_width, resized_height = _resize_shape(width, height)
            resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None]
            normalized = (pixels - PIXEL_MEAN) / PIXEL_STD
            padded = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            padded[:, :, :resized_height, :resized_width] = normalized
            embedding = encoder.run([encoder.get_outputs()[0].name], {"image": padded})[0]

            for point in points:
                coords = np.array([[[
                    point["x"] * resized_width / width,
                    point["y"] * resized_height / height,
                ]]], dtype=np.float32)
                labels = np.array([[1.0]], dtype=np.float32)
                scores, masks = decoder.run(None, {
                    "image_embeddings": embedding,
                    "point_coords": coords,
                    "point_labels": labels,
                })
                scores = np.asarray(scores).reshape(-1)
                candidates = np.asarray(masks).squeeze(0)
                if candidates.ndim == 2:
                    candidates = candidates[None, :, :]
                if len(candidates) != len(scores) or not len(scores):
                    raise ValueError("EdgeSAM returned unexpected mask candidates.")
                selected = candidates[int(np.argmax(scores))]
                candidate_mask = _to_original_mask(
                    selected, (resized_width, resized_height), (width, height)
                )
                mask_pixels = int(np.count_nonzero(candidate_mask))
                area_ratio = mask_pixels / total_pixels
                accepted = area_ratio <= MAX_POINT_MASK_AREA
                point_result = {
                    "x": point["x"],
                    "y": point["y"],
                    "accepted": accepted,
                    "mask_pixel_count": mask_pixels,
                    "total_pixels": total_pixels,
                    "area_ratio": area_ratio,
                    "rejection_reason": None if accepted else "mask_area_over_40_percent",
                }
                point_results.append(point_result)
                if accepted:
                    combined_mask |= candidate_mask

        combined_pixels = int(np.count_nonzero(combined_mask))
        combined_ratio = combined_pixels / total_pixels
        output = BytesIO()
        Image.fromarray(combined_mask.astype(np.uint8) * 255, mode="L").save(output, format="PNG")
        return {
            "points": point_results,
            "mask": output.getvalue(),
            "mask_pixel_count": combined_pixels,
            "total_pixels": total_pixels,
            "area_ratio": combined_ratio,
            "safe": combined_ratio <= MAX_COMBINED_MASK_AREA,
        }
    except SegmenterError:
        raise
    except Exception as error:
        raise SegmenterError("EdgeSAM could not segment the selected image points.") from error
