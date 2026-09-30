from pathlib import Path
from typing import Annotated
import base64
import json
import math
from io import BytesIO
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from app.services.enhancer import EnhancementError, enhance_image
from app.services.inpainter import InpainterError, inpaint_image
from app.services.ocr import OcrEngineError, detect_text_regions, generate_removal_mask
from app.services.segmenter import SegmenterError, segment_points


BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Enhancer")
logger = logging.getLogger(__name__)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


class TextRegion(BaseModel):
    text: str
    confidence: float
    polygon: list[list[float]]


class ProcessResponse(BaseModel):
    success: bool
    filename: str
    width: int
    height: int
    mode: str
    cleanup_level: int | None
    enhancement_level: int
    message: str
    text_regions: list[TextRegion] | None = None
    text_region_count: int | None = None
    mask_width: int | None = None
    mask_height: int | None = None
    mask_preview: str | None = None
    cleaned_image: str | None = None
    enhanced_image: str | None = None
    enhanced_width: int | None = None
    enhanced_height: int | None = None
    extra_points: list[dict[str, int]] | None = None
    segmentation_points: list[dict] | None = None
    segmentation_mask: str | None = None
    segmentation_area_ratio: float | None = None
    segmentation_mask_safe: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    combined_removal_mask: str | None = None
    combined_removal_area_ratio: float | None = None


@app.post("/api/process", response_model=None)
async def validate_process_request(
    image: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    enhancement_level: Annotated[int, Form()],
    cleanup_level: Annotated[int | None, Form()] = None,
    extra_points: Annotated[str | None, Form()] = None,
) -> ProcessResponse | Response:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Choose a JPG, PNG, or WebP image.")
    if mode not in {"clean_enhance", "enhance_only"}:
        raise HTTPException(status_code=422, detail="Invalid processing mode.")
    if mode == "clean_enhance" and cleanup_level not in {1, 2, 3}:
        raise HTTPException(status_code=422, detail="Cleanup level must be 1, 2, or 3.")
    if enhancement_level not in {1, 2, 3}:
        raise HTTPException(status_code=422, detail="Enhancement level must be 1, 2, or 3.")

    contents = await image.read(MAX_FILE_SIZE + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Choose an image file.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum file size is 20 MB.")
    try:
        with Image.open(BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(BytesIO(contents)) as uploaded_image:
            normalized_image = ImageOps.exif_transpose(uploaded_image)
            width, height = normalized_image.size
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from None
    finally:
        await image.close()

    validated_points: list[dict[str, int]] = []
    if mode == "clean_enhance":
        try:
            points = json.loads(extra_points) if extra_points is not None else []
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=400, detail="Extra points must be valid JSON.") from None
        if not isinstance(points, list):
            raise HTTPException(status_code=400, detail="Extra points must be a list.")
        if len(points) > 20:
            raise HTTPException(status_code=400, detail="A maximum of 20 extra points is allowed.")
        for point in points:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise HTTPException(status_code=400, detail="Each extra point must contain numeric x and y coordinates.")
            x, y = point["x"], point["y"]
            if (isinstance(x, bool) or isinstance(y, bool)
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                    or (isinstance(x, float) and not math.isfinite(x))
                    or (isinstance(y, float) and not math.isfinite(y))):
                raise HTTPException(status_code=400, detail="Each extra point must contain numeric x and y coordinates.")
            x, y = int(x), int(y)
            if x < 0 or y < 0 or x >= width or y >= height:
                raise HTTPException(status_code=400, detail="Extra point coordinates must be inside the uploaded image.")
            validated_points.append({"x": x, "y": y})

    if mode == "clean_enhance":
        try:
            segmentation = segment_points(contents, validated_points)
        except SegmenterError as error:
            logger.error("EdgeSAM request failed: %s", error)
            raise HTTPException(status_code=503, detail=str(error)) from None
        try:
            text_regions = detect_text_regions(contents, cleanup_level)
            mask_png = generate_removal_mask(width, height, text_regions, cleanup_level)
        except OcrEngineError as error:
            logger.error("OCR request failed: %s", error)
            raise HTTPException(status_code=503, detail=str(error)) from None

        warnings = []
        accepted_segmentation_mask = Image.new("L", (width, height), 0)
        segmentation_area_ratio = segmentation["area_ratio"]
        if not segmentation["safe"]:
            warnings.append("EdgeSAM selections were excluded because their combined mask exceeded 50% of the image.")
        else:
            with Image.open(BytesIO(segmentation["mask"])) as edge_mask:
                accepted_segmentation_mask = edge_mask.convert("L")
        if any(not point["accepted"] for point in segmentation["points"]):
            warnings.append("One or more UI selections were excluded because their masks exceeded 40% of the image.")

        with Image.open(BytesIO(mask_png)) as ocr_mask:
            final_removal_mask = ImageChops.lighter(ocr_mask.convert("L"), accepted_segmentation_mask)
        combined_pixels = sum(final_removal_mask.histogram()[1:])
        combined_removal_area_ratio = combined_pixels / (width * height)
        combined_mask_png = BytesIO()
        final_removal_mask.save(combined_mask_png, format="PNG")
        final_removal_mask_bytes = combined_mask_png.getvalue()

        with Image.open(BytesIO(contents)) as uploaded_image:
            clean_source = ImageOps.exif_transpose(uploaded_image).convert("RGB")
        has_safe_ui_mask = segmentation["safe"] and segmentation_area_ratio > 0
        if text_regions or has_safe_ui_mask:
            try:
                cleaned = inpaint_image(clean_source, final_removal_mask_bytes, cleanup_level)
            except InpainterError as error:
                logger.error("LaMa request failed: %s", error)
                raise HTTPException(status_code=503, detail=str(error)) from None
            if text_regions and has_safe_ui_mask:
                message = f"Cleaned {len(text_regions)} text regions and selected UI regions."
            elif text_regions:
                message = f"Cleaned {len(text_regions)} text regions."
            else:
                message = "Removed selected UI regions."
        else:
            cleaned = clean_source
            message = "No text or safe UI selection found. Original image returned."
        cleaned_png = BytesIO()
        cleaned.save(cleaned_png, format="PNG")
        try:
            enhanced_png, enhanced_width, enhanced_height = enhance_image(
                cleaned_png.getvalue(), enhancement_level
            )
        except EnhancementError as error:
            logger.error("Real-ESRGAN request failed after cleanup: %s", error)
            raise HTTPException(status_code=503, detail=str(error)) from None

        return ProcessResponse(
            success=True,
            filename=image.filename or "image",
            width=width,
            height=height,
            mode=mode,
            cleanup_level=cleanup_level,
            enhancement_level=enhancement_level,
            message=f"{message} Enhanced image ready.",
            text_regions=text_regions,
            text_region_count=len(text_regions),
            mask_width=width,
            mask_height=height,
            mask_preview=base64.b64encode(mask_png).decode("ascii"),
            cleaned_image=base64.b64encode(cleaned_png.getvalue()).decode("ascii"),
            enhanced_image=base64.b64encode(enhanced_png).decode("ascii"),
            enhanced_width=enhanced_width,
            enhanced_height=enhanced_height,
            extra_points=validated_points,
            segmentation_points=segmentation["points"],
            segmentation_mask=base64.b64encode(segmentation["mask"]).decode("ascii"),
            segmentation_area_ratio=segmentation["area_ratio"],
            segmentation_mask_safe=segmentation["safe"],
            warnings=warnings,
            combined_removal_mask=base64.b64encode(final_removal_mask_bytes).decode("ascii"),
            combined_removal_area_ratio=combined_removal_area_ratio,
        )

    try:
        output, output_width, output_height = enhance_image(contents, enhancement_level)
    except EnhancementError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None

    return Response(
        content=output,
        media_type="image/png",
        headers={
            "X-Image-Width": str(output_width),
            "X-Image-Height": str(output_height),
            "X-Image-Filename": f"{Path(image.filename or 'image').stem}-enhanced.png",
        },
    )
