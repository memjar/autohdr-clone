"""
AutoHDR Clone - FastAPI Backend
================================

REST API for real estate photo editing.

Run:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    POST /hdr/merge          - Merge HDR brackets
    POST /effects/day-to-dusk - Convert to twilight
    POST /edit/remove        - Remove objects (coming soon)
    GET  /health             - Health check
"""

import io
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.hdr_merge import HDRMerger, HDRConfig
from src.models.twilight import TwilightConverter, TwilightConfig

app = FastAPI(
    title="AutoHDR Clone API",
    description="Open-source AI real estate photo editing",
    version="0.1.0"
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Read uploaded file into OpenCV image."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Could not decode image: {file.filename}")
    return image


def image_to_bytes(image: np.ndarray, format: str = ".jpg") -> bytes:
    """Convert OpenCV image to bytes."""
    success, encoded = cv2.imencode(format, image)
    if not success:
        raise HTTPException(500, "Failed to encode result image")
    return encoded.tobytes()


# ============================================
# ROUTES
# ============================================

@app.get("/")
async def root():
    return {
        "name": "AutoHDR Clone API",
        "version": "0.1.0",
        "endpoints": {
            "hdr_merge": "/hdr/merge",
            "day_to_dusk": "/effects/day-to-dusk",
            "remove_object": "/edit/remove (coming soon)",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    Methods:
    - mertens: Exposure fusion (no exposure data needed) - recommended
    - debevec: Traditional HDR (estimates exposures)
    - robertson: Iterative HDR (estimates exposures)

    Tone mapping (for debevec/robertson):
    - reinhard: Natural look
    - drago: More contrast
    - mantiuk: Local contrast
    """
    if len(images) < 2:
        raise HTTPException(400, "Need at least 2 images for HDR merge")
    if len(images) > 9:
        raise HTTPException(400, "Maximum 9 images supported")

    # Read all images
    loaded_images = []
    for img_file in images:
        img = read_image_from_upload(img_file)
        loaded_images.append(img)

    # Configure merger
    config = HDRConfig(
        align_images=align,
        merge_method=method,
        tone_map_method=tone_map
    )

    # Merge
    merger = HDRMerger(config)

    # Estimate exposures if needed
    exposures = None
    if method in ["debevec", "robertson"]:
        from src.core.hdr_merge import estimate_exposures
        exposures = estimate_exposures(loaded_images)

    result = merger.merge_brackets(loaded_images, exposures)

    # Return as JPEG
    result_bytes = image_to_bytes(result, ".jpg")

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=hdr_result.jpg"}
    )


@app.post("/effects/day-to-dusk")
async def day_to_dusk(
    image: UploadFile = File(...),
    sky_intensity: float = Form(0.9),
    window_glow: float = Form(0.8),
    brightness: float = Form(0.7)
):
    """
    Convert daytime exterior photo to twilight/dusk.

    Parameters:
    - sky_intensity: How much to blend the twilight sky (0-1)
    - window_glow: Intensity of window lighting effect (0-1)
    - brightness: Overall brightness reduction (0-1, lower = darker)
    """
    # Read image
    img = read_image_from_upload(image)

    # Configure
    config = TwilightConfig(
        sky_blend_strength=sky_intensity,
        window_glow_intensity=window_glow,
        brightness_reduction=brightness
    )

    # Convert
    converter = TwilightConverter(config)
    result = converter.convert(img)

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

    Sky types:
    - twilight: Dusk gradient (blue to orange)
    - blue: Clear blue sky
    - dramatic: Dark stormy clouds
    - cloudy: Overcast white/grey

    Coming soon: Custom sky upload.
    """
    # Read image
    img = read_image_from_upload(image)

    # Use twilight converter's sky replacement
    converter = TwilightConverter()
    sky_mask = converter.segment_sky(img)

    # Create sky based on type
    if sky_type == "twilight":
        new_sky = converter.create_twilight_sky(img.shape[:2])
    elif sky_type == "blue":
        # Create blue gradient
        h, w = img.shape[:2]
        new_sky = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            ratio = y / h
            new_sky[y, :] = [255 - int(100 * ratio), 200 - int(50 * ratio), 50 + int(50 * ratio)]
    else:
        raise HTTPException(400, f"Unknown sky type: {sky_type}")

    # Replace
    result = converter.replace_sky(img, sky_mask, new_sky)

    # Return
    result_bytes = image_to_bytes(result, ".jpg")
    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=sky_replaced.jpg"}
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
