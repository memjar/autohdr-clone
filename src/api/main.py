"""
AutoHDR Clone - FastAPI Backend
================================

REST API for real estate photo editing with RAW file support.

Run locally:
    cd /path/to/autohdr-clone
    uvicorn src.api.main:app --reload --port 8000

Deploy to Railway:
    Set start command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT

Endpoints:
    POST /process            - Main processing endpoint (matches Vercel frontend)
    POST /hdr/merge          - Merge HDR brackets
    POST /effects/day-to-dusk - Convert to twilight
    GET  /health             - Health check
"""

import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAW file support
try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("Warning: rawpy not installed. RAW file support disabled.")

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.processor import AutoHDRProcessor, ProcessingSettings

# Try to import HDR merger (may not exist yet)
try:
    from src.core.hdr_merge import HDRMerger, HDRConfig
    HAS_HDR_MERGER = True
except ImportError:
    HAS_HDR_MERGER = False

# Try to import twilight (may not exist yet)
try:
    from src.models.twilight import TwilightConverter, TwilightConfig
    HAS_TWILIGHT = True
except ImportError:
    HAS_TWILIGHT = False

app = FastAPI(
    title="AutoHDR Clone API",
    description="Open-source AI real estate photo editing",
    version="0.1.0"
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://autohdr-clone.vercel.app",
        "https://*.vercel.app",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# RAW FILE EXTENSIONS
# ============================================

RAW_EXTENSIONS = {
    '.arw', '.srf', '.sr2',  # Sony
    '.cr2', '.cr3', '.crw',  # Canon
    '.nef', '.nrw',          # Nikon
    '.dng',                  # Adobe
    '.orf',                  # Olympus
    '.rw2',                  # Panasonic
    '.pef', '.ptx',          # Pentax
    '.raf',                  # Fujifilm
    '.erf',                  # Epson
    '.mrw',                  # Minolta
    '.3fr', '.fff',          # Hasselblad
    '.iiq',                  # Phase One
    '.rwl',                  # Leica
    '.srw',                  # Samsung
    '.x3f',                  # Sigma
    '.raw',                  # Generic
}


# ============================================
# MODELS
# ============================================

class HDRMergeRequest(BaseModel):
    method: str = "mertens"
    tone_map: str = "reinhard"
    align: bool = True


class TwilightRequest(BaseModel):
    sky_intensity: float = 0.9
    window_glow: float = 0.8


class ProcessingResponse(BaseModel):
    success: bool
    job_id: str
    message: str


# ============================================
# HELPERS
# ============================================

def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Read uploaded file into OpenCV image, including RAW formats."""
    contents = file.file.read()
    filename = file.filename or "image.jpg"
    ext = Path(filename).suffix.lower()

    # Handle RAW files
    if ext in RAW_EXTENSIONS:
        if not HAS_RAWPY:
            raise HTTPException(400, f"RAW file support not available. Install rawpy.")
        try:
            with rawpy.imread(io.BytesIO(contents)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=8
                )
                # Convert RGB to BGR for OpenCV
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(400, f"Could not decode RAW file {filename}: {str(e)}")

    # Standard image formats
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Could not decode image: {filename}")
    return image


async def read_image_async(file: UploadFile) -> np.ndarray:
    """Async version of read_image_from_upload."""
    contents = await file.read()
    filename = file.filename or "image.jpg"
    ext = Path(filename).suffix.lower()

    # Handle RAW files
    if ext in RAW_EXTENSIONS:
        if not HAS_RAWPY:
            raise HTTPException(400, f"RAW file support not available. Install rawpy.")
        try:
            with rawpy.imread(io.BytesIO(contents)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=8
                )
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise HTTPException(400, f"Could not decode RAW file {filename}: {str(e)}")

    # Standard image formats
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Could not decode image: {filename}")
    return image


def image_to_bytes(image: np.ndarray, format: str = ".jpg", quality: int = 90) -> bytes:
    """Convert OpenCV image to bytes."""
    if format == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        encode_params = []
    success, encoded = cv2.imencode(format, image, encode_params)
    if not success:
        raise HTTPException(500, "Failed to encode result image")
    return encoded.tobytes()


def merge_brackets_mertens(images: List[np.ndarray]) -> np.ndarray:
    """Merge bracketed exposures using Mertens exposure fusion."""
    if len(images) == 1:
        return images[0]

    # Ensure all images same size
    base_shape = images[0].shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] != base_shape:
            img = cv2.resize(img, (base_shape[1], base_shape[0]))
        resized.append(img)

    # Mertens exposure fusion
    merge_mertens = cv2.createMergeMertens()
    images_float = [img.astype(np.float32) / 255.0 for img in resized]
    fusion = merge_mertens.process(images_float)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    return fusion


# ============================================
# ROUTES
# ============================================

@app.get("/")
async def root():
    return {
        "name": "AutoHDR Clone API",
        "version": "1.0.0",
        "endpoints": {
            "process": "POST /process - Main endpoint (HDR merge or twilight)",
            "hdr_merge": "POST /hdr/merge - Legacy HDR merge",
            "day_to_dusk": "POST /effects/day-to-dusk - Legacy twilight",
            "health": "GET /health - Status check"
        },
        "features": {
            "raw_support": HAS_RAWPY,
            "formats": list(RAW_EXTENSIONS) if HAS_RAWPY else ["jpg", "png", "tiff"]
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rawpy": HAS_RAWPY,
        "hdr_merger": HAS_HDR_MERGER,
        "twilight": HAS_TWILIGHT
    }


# ============================================
# MAIN PROCESSING ENDPOINT (matches Vercel frontend)
# ============================================

@app.post("/process")
async def process_images(
    images: List[UploadFile] = File(..., description="Images to process"),
    mode: str = Query("hdr", description="Processing mode: hdr or twilight"),
    brightness: float = Query(0, ge=-2, le=2),
    contrast: float = Query(0, ge=-2, le=2),
    vibrance: float = Query(0, ge=-2, le=2),
    whiteBalance: float = Query(0, ge=-2, le=2),
):
    """
    Main processing endpoint - matches Vercel frontend API.

    - **HDR mode**: Upload 2-9 bracketed exposures, merges with Mertens fusion
    - **Twilight mode**: Upload 1 daytime photo, converts to dusk

    Supports all RAW formats: ARW, CR2, NEF, DNG, etc.
    """
    if not images:
        raise HTTPException(400, "No images provided")

    try:
        # Read all images (including RAW)
        image_arrays = []
        for upload in images:
            img = await read_image_async(upload)
            image_arrays.append(img)

        # Configure processor
        settings = ProcessingSettings(
            brightness=brightness,
            contrast=contrast,
            vibrance=vibrance,
            white_balance=whiteBalance,
        )

        if mode == "twilight":
            # Twilight processing
            settings.twilight_style = "pink"
            processor = AutoHDRProcessor(settings)
            result = processor.process(image_arrays[0])
        else:
            # HDR bracket merging
            merged = merge_brackets_mertens(image_arrays)
            processor = AutoHDRProcessor(settings)
            result = processor.process(merged)

        # Return as high-quality JPEG
        result_bytes = image_to_bytes(result, ".jpg", quality=90)

        return Response(
            content=result_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="autohdr_{mode}.jpg"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ============================================
# LEGACY ENDPOINTS (for direct API use)
# ============================================

@app.post("/hdr/merge")
async def hdr_merge(
    images: List[UploadFile] = File(...),
    method: str = Form("mertens"),
    tone_map: str = Form("reinhard"),
    align: bool = Form(True)
):
    """
    Merge multiple exposure brackets into HDR.

    Upload 2-9 images taken at different exposures.
    Returns the merged, tone-mapped result.
    """
    if len(images) < 2:
        raise HTTPException(400, "Need at least 2 images for HDR merge")
    if len(images) > 9:
        raise HTTPException(400, "Maximum 9 images supported")

    # Read all images (with RAW support)
    loaded_images = []
    for img_file in images:
        img = await read_image_async(img_file)
        loaded_images.append(img)

    # Use Mertens fusion (built-in, always works)
    result = merge_brackets_mertens(loaded_images)

    # Apply HDR processing
    processor = AutoHDRProcessor(ProcessingSettings())
    result = processor.process(result)

    # Return as JPEG
    result_bytes = image_to_bytes(result, ".jpg", quality=90)

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=hdr_result.jpg"}
    )


@app.post("/effects/day-to-dusk")
async def day_to_dusk(
    image: UploadFile = File(...),
    style: str = Form("pink"),  # pink, blue, orange
    brightness: float = Form(-0.5),  # -2 to 2
):
    """
    Convert daytime exterior photo to twilight/dusk.

    Parameters:
    - style: Twilight color (pink, blue, orange)
    - brightness: Adjustment (-2 to 2, negative = darker)
    """
    # Read image (with RAW support)
    img = await read_image_async(image)

    # Configure for twilight
    settings = ProcessingSettings(
        brightness=brightness,
        twilight_style=style if style in ["pink", "blue", "orange"] else "pink"
    )

    # Process
    processor = AutoHDRProcessor(settings)
    result = processor.process(img)

    # Return as JPEG
    result_bytes = image_to_bytes(result, ".jpg")

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=twilight_result.jpg"}
    )


@app.post("/edit/remove")
async def remove_object(
    image: UploadFile = File(...),
    prompt: str = Form(None),
    mask: UploadFile = File(None)
):
    """
    Remove objects from image using AI inpainting.

    Provide either:
    - prompt: Text description of what to remove (e.g., "remove the car")
    - mask: Binary mask image where white = area to remove

    Coming soon: Requires LaMa or IOPaint integration.
    """
    # TODO: Implement with LaMa / Grounded-SAM
    raise HTTPException(501, "Object removal coming soon. Use IOPaint for now: https://github.com/Sanster/IOPaint")


@app.post("/edit/sky-replace")
async def sky_replace(
    image: UploadFile = File(...),
    sky_type: str = Form("twilight")  # twilight, blue, dramatic, cloudy
):
    """
    Replace sky in image.

    Coming soon: Requires sky segmentation model.
    """
    # TODO: Implement with sky segmentation
    raise HTTPException(501, "Sky replacement coming soon. Requires segmentation model.")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
