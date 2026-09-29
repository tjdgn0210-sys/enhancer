from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from io import BytesIO
from app.services.enhancer import EnhancementError, enhance_image


BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Enhancer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


class ProcessResponse(BaseModel):
    success: bool
    filename: str
    width: int
    height: int
    mode: str
    cleanup_level: int | None
    enhancement_level: int
    message: str


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
        return ProcessResponse(
            success=False,
            filename=image.filename or "image",
            width=width,
            height=height,
            mode=mode,
            cleanup_level=cleanup_level,
            enhancement_level=enhancement_level,
            message="Clean + Enhance is not implemented yet.",
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
