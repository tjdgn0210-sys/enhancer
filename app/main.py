from pathlib import Path
from typing import Annotated
import base64
from io import BytesIO
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from app.services.enhancer import EnhancementError, enhance_image
from app.services.inpainter import InpainterError, inpaint_image
from app.services.ocr import OcrEngineError, detect_text_regions, generate_removal_mask


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


@app.post("/api/process", response_model=None)
async def validate_process_request(
    image: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    enhancement_level: Annotated[int, Form()],
    cleanup_level: Annotated[int | None, Form()] = None,
) -> ProcessResponse | Response:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Choose a JPG, PNG, or WebP image.")
    if mode not in {"clean_enhance", "enhance_only"}:
        raise HTTPException(status_code=422, detail="Invalid processing mode.")
    if mode == "clean_enhance" and cleanup_level not in {1, 2, 3}:
        raise HTTPException(status_code=422, detail="Cleanup level must be 1, 2, or 3.")
    if enhancement_level not in {1, 2, 3, 4}:
        raise HTTPException(status_code=422, detail="Enhancement level must be 1, 2, 3, or 4.")

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

    if mode == "clean_enhance":
        try:
            text_regions = detect_text_regions(contents, cleanup_level)
            mask_png = generate_removal_mask(width, height, text_regions, cleanup_level)
        except OcrEngineError as error:
            logger.error("OCR request failed: %s", error)
            raise HTTPException(status_code=503, detail=str(error)) from None
        with Image.open(BytesIO(contents)) as uploaded_image:
            clean_source = ImageOps.exif_transpose(uploaded_image).convert("RGB")
        if text_regions:
            try:
                cleaned = inpaint_image(clean_source, mask_png, cleanup_level)
            except InpainterError as error:
                logger.error("LaMa request failed: %s", error)
                raise HTTPException(status_code=503, detail=str(error)) from None
            message = f"Cleaned {len(text_regions)} text regions."
        else:
            cleaned = clean_source
            message = "No text found. Original image returned."
        cleaned_png = BytesIO()
        cleaned.save(cleaned_png, format="PNG")
        return ProcessResponse(
            success=True,
            filename=image.filename or "image",
            width=width,
            height=height,
            mode=mode,
            cleanup_level=cleanup_level,
            enhancement_level=enhancement_level,
            message=message,
            text_regions=text_regions,
            text_region_count=len(text_regions),
            mask_width=width,
            mask_height=height,
            mask_preview=base64.b64encode(mask_png).decode("ascii"),
            cleaned_image=base64.b64encode(cleaned_png.getvalue()).decode("ascii"),
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
