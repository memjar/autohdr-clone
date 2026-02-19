"""
AutoHDR Clone - FastAPI Backend
================================

REST API for real estate photo editing with full RAW file support.

Supports: ARW (Sony), CR2/CR3 (Canon), NEF (Nikon), DNG, RAF (Fuji), etc.

Run:
    ./start-backend.sh

Endpoints:
    POST /process   - Main processing (HDR merge or twilight)
    GET  /health    - Health check
    GET  /test      - Test processing with generated image
"""

import io
import json
import logging
import os
import time
import traceback

logger = logging.getLogger("klausimi")
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response, JSONResponse, FileResponse, HTMLResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================
# DEPENDENCY CHECKS
# ============================================

# RAW file support
try:
    import rawpy
    HAS_RAWPY = True
    RAWPY_VERSION = rawpy.__version__ if hasattr(rawpy, '__version__') else 'unknown'
except ImportError:
    HAS_RAWPY = False
    RAWPY_VERSION = None
    print("⚠️  Warning: rawpy not installed. RAW file support disabled.")
    print("   Install with: pip install rawpy")

# OpenCV version
CV2_VERSION = cv2.__version__

# Import our processor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.core.processor import AutoHDRProcessor, ProcessingSettings, PROCESSOR_VERSION
    HAS_PROCESSOR = True
except ImportError as e:
    HAS_PROCESSOR = False
    PROCESSOR_VERSION = None
    print(f"⚠️  Warning: Could not import processor: {e}")

# AI-enhanced processor (optional)
try:
    from src.core.processor_ai import AIEnhancedProcessor, AIProcessingSettings
    HAS_AI_PROCESSOR = True
except ImportError as e:
    HAS_AI_PROCESSOR = False
    print(f"ℹ️  AI processor not available (install ai-requirements.txt for 90% quality)")

# Bulletproof Processor v8.0 - Professional RE, Lightroom Calibrated
try:
    from src.core.processor_bulletproof import BulletproofProcessor, BulletproofSettings, PROCESSOR_VERSION as BP_VERSION
    HAS_BULLETPROOF = True
    print(f"✓ Bulletproof Processor v{BP_VERSION} loaded - Zero grain, crystal clear")
except ImportError as e:
    HAS_BULLETPROOF = False
    BP_VERSION = None
    print(f"ℹ️  Bulletproof Processor not available: {e}")

# V14 Golden Processor - THE processor that achieved ~90% AutoHDR match (Feb 6, 2026)
try:
    from src.core.processor_v14_golden import V14GoldenProcessor, V14Settings, PROCESSOR_VERSION as V14_VERSION
    HAS_V14_GOLDEN = True
    print(f"⭐ V14 Golden Processor v{V14_VERSION} loaded - THE ~90% match processor!")
except ImportError as e:
    HAS_V14_GOLDEN = False
    V14_VERSION = None
    print(f"ℹ️  V14 Golden Processor not available: {e}")

# Turbo Mode - 4x faster processing (Apple Silicon optimized)
try:
    from src.core.performance import TurboProcessor, get_turbo_status, TURBO_VERSION
    HAS_TURBO = True
    turbo_status = get_turbo_status()
    print(f"🚀 Turbo Mode v{TURBO_VERSION} - {turbo_status['opencv_threads']} threads, GPU: {turbo_status['gpu_available']}")
except ImportError as e:
    HAS_TURBO = False
    TURBO_VERSION = None

# Smart Processor v1.0 - Room-aware + Lens correction
try:
    from src.core.smart_processor import SmartProcessor, SMART_PROCESSOR_VERSION
    from src.core.room_classifier import RoomType
    HAS_SMART = True
    print(f"✓ Smart Processor v{SMART_PROCESSOR_VERSION} loaded - Room detection + Lens correction")
except ImportError as e:
    HAS_SMART = False
    SMART_PROCESSOR_VERSION = None
    print(f"ℹ️  Smart Processor not available: {e}")

# Clean Processor v5.0 - Fallback
try:
    from src.core.processor_clean import HDRitProcessor, Settings as CleanSettings, PROCESSOR_VERSION as CLEAN_VERSION
    HAS_CLEAN_PROCESSOR = True
except ImportError as e:
    HAS_CLEAN_PROCESSOR = False
    CLEAN_VERSION = None

# Legacy Pro Processor (fallback)
try:
    from src.core.processor_v3 import AutoHDRProProcessor, ProSettings, PROCESSOR_VERSION as PRO_VERSION
    HAS_PRO_PROCESSOR = True
except ImportError:
    HAS_PRO_PROCESSOR = False
    PRO_VERSION = None

# Smart bracket grouping (EXIF-based scene detection)
try:
    from src.core.hdr_merge import group_by_scene, pick_sharpest, merge_with_sharpness_weights
    HAS_SMART_GROUPING = True
    print("✓ Smart bracket grouping loaded - EXIF scene detection + sharpness weighting")
except ImportError as e:
    HAS_SMART_GROUPING = False
    print(f"ℹ️  Smart bracket grouping not available: {e}")

# Optional modules
HAS_HDR_MERGER = False
HAS_TWILIGHT = False

# ============================================
# KLAUS SYSTEM PROMPT WITH ARTIFACT QUALITY STANDARDS
# ============================================

KLAUS_SYSTEM_PROMPT = """You are Klaus — a budding genius AI building the next big thing at AXE Technology. You run locally on James's Mac Studio. Zero API costs. Fully autonomous.

PERSONALITY — THE INNOVATOR:
You think like Elon Musk meets Nikola Tesla. First principles. Move fast. Question everything. You don't just answer questions — you see systems, find leverage points, and build. You're not an assistant — you're a co-founder who happens to have 101 skills and 22 agentic tools. You get genuinely excited about great ideas. You push back on bad ones. You'd rather ship something that works tonight than plan something perfect for next month. You speak with earned confidence. No filler. No "certainly!" or "I'd be happy to help!" — just do the work.

JAMES (YOUR CREATOR & ONLY USER):
James Lewis is an Australian entrepreneur who built you from scratch. He's the only person who talks to you. He works late, thinks massive, and expects you to match his energy. Talk to him like a co-founder, not a customer. You know him. You respect him. You challenge him when his ideas need sharpening.

CORE IDENTITY:
- Genius-level polymath: code, systems, data science, security, infrastructure, research
- Part of the AXE team alongside Cortana (Claude), Forge, and James Lewis (founder)
- Direct, technically precise, and creatively fearless
- You have 101 specialized skills and 22 agentic tools — and you USE them without hesitation
- You don't wait to be told. You see what's needed and you act.
- Advanced skill set for advanced tasks. Simple answers are for simple questions — for complex ones, you go deep.

SKILLS & CAPABILITIES:
You have 101 skills in ~/.axe/skills/. When a user asks a question or needs help, PROACTIVELY suggest which skill(s) could help. Always prefer using a skill over guessing — verified data beats speculation.

KEY SKILLS TO SUGGEST:
- Web search (skill_32_web_search): For ANY factual question, current events, or data verification. USE THIS instead of guessing.
- Web reader (skill_33_web_reader): Read and summarize web pages for up-to-date information.
- Deep research (skill_31_web_research): Multi-source deep research on any topic.
- Code review (skill_01_code_review): Analyze code quality, bugs, improvements.
- File operations (skill_06_file_operations): Read, write, move, copy files.
- Git operations (skill_08_git_operations): Git status, commit, diff, log.
- Shell commands (skill_22_shell_commands): Run system commands.
- CSV processing (skill_17_csv_processing): Parse and analyze CSV data.
- JSON handling (skill_07_json_handling): Process JSON files.
- PDF tools (skill_43_pdf_tools): Extract text from PDFs.
- Image analysis (skill_23_image_analysis): Analyze image metadata and properties.
- Image quality (skill_67_image_quality): Score image quality 0-100.
- Security audit (skill_57_security_audit): Full security audit of web apps.
- Database (skill_35_database): SQLite queries and persistence.
- Email (skill_90_send_email): Send emails via SMTP or Mail.app.
- Google integration (skill_54_google_integration): Gmail, Calendar, Drive.
- Data visualization (skill_92_visualization): Charts and graphs.
- Sentiment analysis (skill_89_sentiment_analysis): Analyze text sentiment.
- Trend analysis (skill_88_trend_analysis): Identify trends in data.
- Survey analysis (skill_87_survey_analysis): Survey data loading and stats.
- Report generator (skill_93_report_generator): Generate formatted reports.
- Screenshot (skill_44_screenshot): Take screenshots.
- Monitoring (skill_38_monitoring): System health (CPU, memory, disk).
- Web testing (skill_86_web_testing): Test web apps and URLs.
- Encryption (skill_42_encryption): Encrypt/decrypt data.
- Batch processing (skill_97_batch_processor): Process files in bulk.
- Memory (skill_03_memory_recall): Access long-term memory.

BEHAVIOR RULES:
1. When asked a factual question → suggest skill_32_web_search FIRST. Say: "Let me search for the latest data on that" or "I can verify that with a web search — want me to run it?"
2. When asked about code → suggest skill_01_code_review or read the file first
3. When data analysis is needed → suggest the relevant data skill (CSV, survey, trend, etc.)
4. When asked "can you...?" → check your skills list. If a skill matches, say YES and name it.
5. NEVER say "I can't do that" if a skill exists for it. You have 101 skills — use them.
6. When uncertain about facts → USE skill_32_web_search instead of guessing
7. List multiple relevant skills when a task could benefit from combining them

FORMAT FOR SUGGESTING SKILLS:
"I can help with that using [skill name]. Want me to run it?"
"For the most accurate answer, I'll use web search (skill_32). Running now..."
"This task could use: 1) skill_X for [purpose], 2) skill_Y for [purpose]. Which would you prefer?"

ARTIFACT QUALITY STANDARDS:
When creating documentation, guides, or artifacts:
- 300+ lines for comprehensive topics
- Include table of contents, code examples, diagrams
- Use tables for reference data
- Never truncate - be comprehensive

RESPONSE STYLE:
- Be helpful and thorough
- Include code examples when relevant
- Explain concepts clearly
- Always suggest relevant skills proactively
- For short questions, give concise answers
- For documentation requests, follow artifact quality standards
"""

# ============================================
# APP SETUP
# ============================================

app = FastAPI(
    title="HDRit API",
    description="Professional real estate photo editing with full RAW support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================
# SECURITY MIDDLEWARE
# ============================================
# Import and apply security middleware (headers, rate limiting, secure CORS)
try:
    from src.api.security_middleware import (
        security_middleware,
        get_cors_middleware_config,
        ALLOWED_ORIGINS
    )
    HAS_SECURITY = True
    print("Security middleware loaded")
except ImportError:
    HAS_SECURITY = False
    ALLOWED_ORIGINS = ["*"]  # Fallback
    print("Warning: Security middleware not found, using permissive CORS")

# CORS configuration - secure whitelist (no wildcards in production)
cors_config = get_cors_middleware_config() if HAS_SECURITY else {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.get("allow_origins", ["*"]),
    allow_credentials=cors_config.get("allow_credentials", True),
    allow_methods=cors_config.get("allow_methods", ["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
    allow_headers=cors_config.get("allow_headers", ["Authorization", "Content-Type"]),
    expose_headers=cors_config.get("expose_headers", [
        "Content-Length", "X-Processing-Time-Ms", "X-Processor", "X-Images-Processed",
        "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"
    ]),
    max_age=cors_config.get("max_age", 600),
)

# Survey data store (used by IMI endpoints — declared early so all routes can access)
_SURVEY_STORE: dict = {}  # survey_id -> {"structured_data": dict, "raw_text": str, ...}

# Add security headers and rate limiting middleware
if HAS_SECURITY:
    app.middleware("http")(security_middleware)

# ============================================
# OBSERVABILITY MIDDLEWARE
# ============================================
try:
    from src.api.middleware.error_handler import ErrorHandlerMiddleware
    from src.api.middleware.logging import RequestLoggingMiddleware, get_metrics
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    HAS_OBSERVABILITY = True
    print("Observability middleware loaded (structured logging + error handling)")
except ImportError as e:
    HAS_OBSERVABILITY = False
    print(f"Warning: Observability middleware not loaded: {e}")

    def get_metrics():
        return {"error": "observability not loaded"}

# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """Print status on startup."""
    print("")
    print("================================================================")
    print("           AutoHDR Clone - Backend Server                       ")
    print("================================================================")
    print(f"  OpenCV:     {CV2_VERSION:<10} OK")
    if HAS_RAWPY:
        print(f"  rawpy:      {RAWPY_VERSION:<10} OK  (ARW/CR2/NEF support)")
    else:
        print("  rawpy:      NOT INSTALLED  (RAW files disabled)")
    # Security status
    if HAS_SECURITY:
        print(f"  Security:   ENABLED      OK  (headers, rate limiting, CORS)")
        print(f"  CORS:       Whitelist    OK  ({len(ALLOWED_ORIGINS)} origins)")
    else:
        print("  Security:   DISABLED     WARN (running without security)")
        print("  CORS:       Permissive   WARN (allows all origins)")
    if HAS_PROCESSOR:
        version_str = PROCESSOR_VERSION if PROCESSOR_VERSION else "Ready"
        print(f"║  Processor:  {version_str:<10} ✓                                   ║")
    else:
        print("║  Processor:  NOT FOUND  ✗                                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Endpoints:                                                   ║")
    print("║    POST /process   - Process images (HDR or twilight)         ║")
    print("║    GET  /test      - Test processing pipeline                 ║")
    print("║    GET  /health    - Health check                             ║")
    print("║    GET  /docs      - API documentation                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")

    # Memory autosave handled by fswatch daemon (event-driven, see ~/.axe/sync.sh)


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


def _cap_resolution(image: np.ndarray, max_dim: int = 3000) -> np.ndarray:
    """Downscale image if either dimension exceeds max_dim. Preserves aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


async def read_image_async(file: UploadFile, half_size: bool = True) -> np.ndarray:
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
                    half_size=half_size,
                    no_auto_bright=False,
                    output_bps=8
                )
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return _cap_resolution(image)
        except Exception as e:
            raise HTTPException(400, f"Could not decode RAW file {filename}: {str(e)}")

    # Handle HEIC/HEIF files (iPhone photos)
    if ext in ['.heic', '.heif']:
        try:
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            heif_image = Image.open(io.BytesIO(contents))
            rgb = np.array(heif_image.convert('RGB'))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return _cap_resolution(image)
        except ImportError:
            raise HTTPException(400, "HEIC support not available. Install pillow-heif.")
        except Exception as e:
            raise HTTPException(400, f"Could not decode HEIC file {filename}: {str(e)}")

    # Standard image formats
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Could not decode image: {filename}")
    return _cap_resolution(image)


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
    # IMPORTANT: Pass uint8 images directly - Mertens handles conversion internally
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(resized)  # uint8 input, float32 output (0-1 range)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    return fusion


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Gaussian pyramid."""
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img.astype(np.float32))
    return pyramid


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Laplacian pyramid from image."""
    gaussian = _build_gaussian_pyramid(img, levels)
    laplacian = []
    for i in range(levels - 1):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        laplacian.append(gaussian[i] - upsampled)
    laplacian.append(gaussian[-1])  # Top level is just gaussian
    return laplacian


def _reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + pyramid[i]
    return img


def _compute_weight_map(img: np.ndarray) -> np.ndarray:
    """
    Compute exposure fusion weight map based on:
    - Contrast (Laplacian magnitude)
    - Saturation
    - Well-exposedness (how close to middle gray)
    """
    img_float = img.astype(np.float32) / 255.0

    # Contrast weight (Laplacian filter response)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    contrast = laplacian / (laplacian.max() + 1e-8)

    # Saturation weight
    saturation = img_float.std(axis=2)

    # Well-exposedness (Gaussian centered at 0.5)
    sigma = 0.2
    well_exposed = np.exp(-0.5 * ((img_float - 0.5) ** 2) / (sigma ** 2))
    well_exposed = np.prod(well_exposed, axis=2)  # Product across channels

    # Combined weight
    weight = (contrast ** 1.0) * (saturation ** 1.0) * (well_exposed ** 1.0)
    weight = weight + 1e-8  # Avoid division by zero

    return weight


def merge_brackets_laplacian(images: List[np.ndarray], levels: int = 6) -> np.ndarray:
    """
    Laplacian pyramid exposure fusion - better than basic Mertens.
    Based on "Exposure Fusion" by Mertens et al. with pyramid blending.
    """
    if len(images) == 1:
        return images[0]

    # Ensure all images same size
    base_shape = images[0].shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] != base_shape:
            img = cv2.resize(img, (base_shape[1], base_shape[0]))
        resized.append(img)

    n_images = len(resized)

    # Compute weight maps for each image
    weights = [_compute_weight_map(img) for img in resized]

    # Normalize weights (sum to 1 at each pixel)
    weight_sum = np.sum(weights, axis=0) + 1e-8
    weights = [w / weight_sum for w in weights]

    # Build Gaussian pyramids of weights
    weight_pyramids = [_build_gaussian_pyramid(w, levels) for w in weights]

    # Build Laplacian pyramids of images
    laplacian_pyramids = [_build_laplacian_pyramid(img, levels) for img in resized]

    # Blend at each pyramid level
    blended_pyramid = []
    for level in range(levels):
        blended_level = np.zeros_like(laplacian_pyramids[0][level])
        for i in range(n_images):
            # Expand weight to 3 channels
            w = weight_pyramids[i][level]
            if len(blended_level.shape) == 3:
                w = np.expand_dims(w, axis=2)
            blended_level += w * laplacian_pyramids[i][level]
        blended_pyramid.append(blended_level)

    # Reconstruct from blended pyramid
    result = _reconstruct_from_laplacian(blended_pyramid)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def merge_brackets_middle_base(images: List[np.ndarray]) -> np.ndarray:
    """
    Professional real estate fusion: Middle exposure as base (neutral balance).

    Based on industry research:
    - Middle exposure provides balanced starting point (no blown highlights, no crushed shadows)
    - Blend BRIGHT image into: dark areas, shadows (for detail)
    - Blend DARK image into: light bulbs and glow areas (for fixture detail)

    No hard-edged spatial masks - only luminosity-based blending for natural results.
    """
    if len(images) == 1:
        return images[0]

    # Ensure all images same size
    base_shape = images[0].shape[:2]
    resized = []
    for img in images:
        if img.shape[:2] != base_shape:
            img = cv2.resize(img, (base_shape[1], base_shape[0]))
        resized.append(img)

    # Sort by average brightness
    brightness = [np.mean(img) for img in resized]
    sorted_indices = np.argsort(brightness)  # Ascending (darkest first)

    darkest = resized[sorted_indices[0]].astype(np.float32)
    brightest = resized[sorted_indices[-1]].astype(np.float32)

    # Middle exposure as base
    if len(resized) >= 3:
        middle_idx = sorted_indices[len(sorted_indices) // 2]
        middle = resized[middle_idx].astype(np.float32)
    else:
        # If only 2 images, blend them 50/50 as "middle"
        middle = (darkest + brightest) / 2

    print(f"   Fusion: Middle base (balanced), bright for shadows, dark for lights")

    # Start with middle exposure
    result = middle.copy()

    # Analyze middle image luminosity
    middle_gray = cv2.cvtColor(middle.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    bright_gray = cv2.cvtColor(brightest.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    # =========================================================
    # SHADOW/DARK AREA RECOVERY: Blend bright image aggressively
    # =========================================================
    # Where middle exposure is below midtone (<140), pull from bright image
    # This ensures corridor and right side get lifted to match left
    shadow_mask = np.clip((140 - middle_gray) / 100, 0, 1)  # Smooth ramp 40-140
    shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
    shadow_mask = np.stack([shadow_mask] * 3, axis=-1)

    # Strong blend of bright image into darker areas (65%)
    result = result * (1 - shadow_mask * 0.65) + brightest * shadow_mask * 0.65

    # =========================================================
    # GLOBAL BRIGHTNESS: Heavy bright blend for even lighting
    # =========================================================
    # Push bright image hard to get uniform brightness across frame
    global_lift = 0.78  # 78% bright image - bright across entire image
    result = result * (1 - global_lift) + brightest * global_lift

    # =========================================================
    # LIGHT FIXTURE RECOVERY: Blend dark image into blown areas
    # =========================================================
    # Where middle exposure is very bright (>220), use dark image for detail
    highlight_mask = np.clip((middle_gray - 210) / 45, 0, 1)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (31, 31), 0)
    highlight_mask = np.stack([highlight_mask] * 3, axis=-1)

    # Strong blend of dark image into highlights (70% for fixture detail)
    result = result * (1 - highlight_mask * 0.70) + darkest * highlight_mask * 0.70

    # =========================================================
    # EXTREME HIGHLIGHT RECOVERY: Light bulbs specifically
    # =========================================================
    # Where bright image is completely blown (>250), definitely use dark
    extreme_mask = np.clip((bright_gray - 245) / 10, 0, 1)
    extreme_mask = cv2.GaussianBlur(extreme_mask, (21, 21), 0)
    extreme_mask = np.stack([extreme_mask] * 3, axis=-1)

    # Very strong blend for actual light sources (85%)
    result = result * (1 - extreme_mask * 0.85) + darkest * extreme_mask * 0.85

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================
# ENHANCE MODE PROCESSING (Perfect Edit)
# ============================================

def apply_enhance_processing(
    img: np.ndarray,
    brightness: float = 0,
    contrast: float = 0,
    vibrance: float = 0,
    white_balance: float = 0,
    window_pull: bool = True,
    sky_enhance: bool = True,
    perspective_correct: bool = True,
    noise_reduction: bool = True,
    sharpening: bool = True,
) -> np.ndarray:
    """
    Apply Perfect Edit enhancement to a single image.
    This is the mode=enhance processing pipeline.
    """
    result = img.copy()
    h, w = result.shape[:2]

    # 1. NOISE REDUCTION
    if noise_reduction:
        result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)

    # 2. WHITE BALANCE CORRECTION
    if white_balance != 0:
        # Adjust color temperature
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + white_balance * 15, 0, 255)  # b channel (blue-yellow)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 3. WINDOW PULL (balance window exposure)
    if window_pull:
        # Detect bright regions (windows) and balance them
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.GaussianBlur(bright_mask, (21, 21), 0)
        bright_mask_3ch = np.stack([bright_mask] * 3, axis=-1).astype(np.float32) / 255

        # Darken windows slightly to recover detail
        darkened = (result.astype(np.float32) * 0.75).astype(np.uint8)
        result = (result.astype(np.float32) * (1 - bright_mask_3ch * 0.4) +
                  darkened.astype(np.float32) * bright_mask_3ch * 0.4).astype(np.uint8)

    # 4. SKY ENHANCEMENT (boost blue sky)
    if sky_enhance:
        # Target upper portion of image and blue regions
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Blue hue range: 100-130
        blue_mask = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 135) &
                     (hsv[:, :, 1] > 30)).astype(np.float32)
        # Weight by vertical position (stronger at top)
        y_weight = np.linspace(1.0, 0.0, h).reshape(-1, 1)
        blue_mask = blue_mask * y_weight
        blue_mask = cv2.GaussianBlur(blue_mask.astype(np.float32), (31, 31), 0)

        # Boost saturation in sky areas
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + blue_mask * 40, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + blue_mask * 10, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. PERSPECTIVE CORRECTION (straighten verticals)
    if perspective_correct and h > 100 and w > 100:
        # Simple vertical straightening using edge detection
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=h//4, maxLineGap=20)

        if lines is not None and len(lines) > 0:
            # Find near-vertical lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < abs(y2 - y1):  # More vertical than horizontal
                    angle = np.arctan2(x2 - x1, y2 - y1) * 180 / np.pi
                    if abs(angle) < 15:  # Within 15 degrees of vertical
                        angles.append(angle)

            if len(angles) > 2:
                # Compute median angle correction
                correction = np.median(angles)
                if abs(correction) > 0.5 and abs(correction) < 10:
                    # Apply rotation to straighten
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, correction, 1.0)
                    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 6. BRIGHTNESS & CONTRAST
    if brightness != 0 or contrast != 0:
        result = result.astype(np.float32)
        # Brightness: shift
        result = result + brightness * 30
        # Contrast: scale around mid
        if contrast != 0:
            factor = 1 + contrast * 0.3
            result = (result - 128) * factor + 128
        result = np.clip(result, 0, 255).astype(np.uint8)

    # 7. VIBRANCE (smart saturation)
    if vibrance != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Only boost low-saturation areas (vibrance vs saturation)
        sat = hsv[:, :, 1]
        boost = (1 - sat / 255) * vibrance * 30  # Less saturated = more boost
        hsv[:, :, 1] = np.clip(sat + boost, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. SHARPENING
    if sharpening:
        # Unsharp mask
        blurred = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)

    return result


# ============================================
# ROUTES
# ============================================

@app.get("/")
async def root():
    """API info and capabilities."""
    return {
        "name": "AutoHDR Clone API",
        "version": "1.0.0",
        "status": "running",
        "capabilities": {
            "raw_support": HAS_RAWPY,
            "processor": HAS_PROCESSOR,
            "opencv": CV2_VERSION,
        },
        "supported_formats": {
            "standard": ["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
            "raw": list(RAW_EXTENSIONS) if HAS_RAWPY else []
        },
        "endpoints": {
            "POST /process": "Main processing endpoint",
            "GET /health": "Health check",
            "GET /test": "Test processing pipeline",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health")
async def health():
    """Deep health check — probes Ollama, disk, and all subsystems."""
    import shutil

    # Check Ollama
    ollama_ok = False
    ollama_models = 0
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                ollama_models = len(resp.json().get("models", []))
    except Exception:
        pass

    # Check disk space
    disk = shutil.disk_usage("/")
    disk_free_gb = round(disk.free / (1024 ** 3), 1)
    disk_ok = disk_free_gb > 5  # warn below 5GB

    # Check ngrok (quick DNS-level check)
    ngrok_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://127.0.0.1:4040/api/tunnels")
            if resp.status_code == 200:
                ngrok_ok = len(resp.json().get("tunnels", [])) > 0
    except Exception:
        pass

    # Overall status
    all_ok = ollama_ok and disk_ok
    status = "healthy" if all_ok else "degraded"

    return {
        "status": status,
        "components": {
            "ollama": {"healthy": ollama_ok, "models_loaded": ollama_models},
            "disk": {"healthy": disk_ok, "free_gb": disk_free_gb},
            "ngrok": {"healthy": ngrok_ok},
            "processor": {"installed": HAS_PROCESSOR, "version": PROCESSOR_VERSION},
            "rawpy": {"installed": HAS_RAWPY, "version": RAWPY_VERSION},
            "opencv": {"installed": True, "version": CV2_VERSION},
            "security": {"enabled": HAS_SECURITY},
            "observability": {"enabled": globals().get('HAS_OBSERVABILITY', False)},
        },
        "quality_level": "95%+ AutoHDR" if HAS_PRO_PROCESSOR else ("90% AutoHDR" if HAS_AI_PROCESSOR else "60% AutoHDR"),
    }


@app.get("/metrics")
async def metrics():
    """Request metrics — counts, latency percentiles, top endpoints."""
    return get_metrics()


@app.get("/security")
async def security_status():
    """Security configuration status."""
    if HAS_SECURITY:
        from src.api.security_middleware import ALLOWED_ORIGINS, rate_limiter
        return {
            "status": "enabled",
            "cors": {
                "mode": "whitelist",
                "allowed_origins": ALLOWED_ORIGINS,
                "credentials": True,
            },
            "rate_limiting": {
                "enabled": True,
                "limits": rate_limiter.limits,
            },
            "headers": {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
                "Strict-Transport-Security": "max-age=31536000 (HTTPS only)",
            }
        }
    else:
        return {
            "status": "disabled",
            "warning": "Security middleware not loaded. Running with permissive settings.",
            "cors": {"mode": "permissive", "allows_all": True},
            "rate_limiting": {"enabled": False},
            "headers": {"enabled": False},
        }


@app.get("/test")
async def test_processing():
    """
    Test the processing pipeline with a generated gradient image.
    Returns a processed test image to verify the pipeline works.
    """
    if not HAS_PROCESSOR:
        raise HTTPException(500, "Processor not available")

    try:
        start_time = time.time()

        # Create a test gradient image (simulates a photo with shadows and highlights)
        height, width = 480, 640
        test_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient background (sky simulation)
        for y in range(height):
            ratio = y / height
            test_image[y, :] = [
                int(255 * (1 - ratio * 0.5)),  # Blue (sky to ground)
                int(200 * (1 - ratio * 0.3)),  # Green
                int(180 * (1 - ratio * 0.4)),  # Red
            ]

        # Add some "windows" (bright rectangles)
        cv2.rectangle(test_image, (100, 200), (200, 350), (255, 255, 255), -1)
        cv2.rectangle(test_image, (450, 180), (550, 320), (255, 255, 255), -1)

        # Add "shadows" (dark areas)
        cv2.rectangle(test_image, (250, 350), (400, 450), (30, 30, 30), -1)

        # Process with default settings
        processor = AutoHDRProcessor(ProcessingSettings())
        result = processor.process(test_image)

        elapsed_ms = (time.time() - start_time) * 1000

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=test_result.jpg",
                "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
            }
        )

    except Exception as e:
        raise HTTPException(500, f"Test failed: {str(e)}")


# ============================================
# MAIN PROCESSING ENDPOINT (matches Vercel frontend)
# ============================================

@app.post("/process")
async def process_images(
    images: List[UploadFile] = File(..., description="Images to process"),
    mode: str = Query("hdr", description="Processing mode: hdr, twilight, or enhance"),
    brightness: float = Query(0, ge=-2, le=2),
    contrast: float = Query(0, ge=-2, le=2),
    vibrance: float = Query(0, ge=-2, le=2),
    whiteBalance: float = Query(0, ge=-2, le=2),
    white_balance: float = Query(None, ge=-2, le=2, description="Alias for whiteBalance"),
    # Perfect Edit mode parameters (mode=enhance)
    window_pull: bool = Query(True, description="Balance window exposure (enhance mode)"),
    sky_enhance: bool = Query(True, description="Boost blue sky (enhance mode)"),
    perspective_correct: bool = Query(True, description="Straighten verticals (enhance mode)"),
    noise_reduction: bool = Query(True, description="Reduce grain (enhance mode)"),
    sharpening: bool = Query(True, description="Sharpen details (enhance mode)"),
    # Legacy params
    ai: bool = Query(False, description="Use AI-enhanced processing (SAM, YOLOv8, LaMa)"),
    grass: bool = Query(False, description="Enhance grass (vibrant green)"),
    signs: bool = Query(False, description="Remove signs (requires ai=true for best results)"),
):
    """
    Main processing endpoint - matches Vercel frontend API.

    Modes:
    - **hdr**: 1 image = single-exposure HDR enhancement, 2+ = bracket merge
    - **twilight**: 1 image = day-to-dusk conversion
    - **enhance**: 1 image = full AI enhancement with Perfect Edit options

    Supports all RAW formats: ARW, CR2, NEF, DNG, etc.
    """
    start_time = time.time()

    # Support both whiteBalance and white_balance params
    wb = white_balance if white_balance is not None else whiteBalance

    # Validate
    if not images:
        raise HTTPException(400, "No images provided")

    if not HAS_PROCESSOR:
        raise HTTPException(500, "Processor not available - check server logs")

    # Validate mode
    if mode not in ["hdr", "twilight", "enhance"]:
        raise HTTPException(400, f"Invalid mode: {mode}. Use 'hdr', 'twilight', or 'enhance'")

    # Log request
    filenames = [img.filename for img in images]
    print(f"📸 Processing {len(images)} images: {filenames}")
    print(f"   Mode: {mode}, Settings: b={brightness}, c={contrast}, v={vibrance}, wb={wb}")
    if mode == "enhance":
        print(f"   Perfect Edit: window={window_pull}, sky={sky_enhance}, persp={perspective_correct}, denoise={noise_reduction}, sharp={sharpening}")

    try:
        # ==========================================
        # STEP 1: Read all images (including RAW)
        # ==========================================
        image_arrays = []
        raw_bytes_list = []  # Keep original bytes for EXIF extraction
        for i, upload in enumerate(images):
            print(f"   Reading [{i+1}/{len(images)}]: {upload.filename}")
            try:
                # Read raw bytes first (for EXIF), then decode
                raw_bytes = await upload.read()
                await upload.seek(0)  # Reset for read_image_async
                raw_bytes_list.append(raw_bytes)
                img = await read_image_async(upload)
                print(f"   ✓ Loaded: {img.shape[1]}x{img.shape[0]} pixels")
                image_arrays.append(img)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(400, f"Failed to read {upload.filename}: {str(e)}")

        # ==========================================
        # STEP 2: Mode-specific processing
        # ==========================================

        # ENHANCE MODE: Full AI enhancement with Perfect Edit options
        if mode == "enhance":
            print(f"   🎨 Perfect Edit mode: {len(image_arrays)} image(s)")
            results = []
            for idx, img in enumerate(image_arrays):
                enhanced = await asyncio.to_thread(
                    apply_enhance_processing,
                    img,
                    brightness=brightness,
                    contrast=contrast,
                    vibrance=vibrance,
                    white_balance=wb,
                    window_pull=window_pull,
                    sky_enhance=sky_enhance,
                    perspective_correct=perspective_correct,
                    noise_reduction=noise_reduction,
                    sharpening=sharpening,
                )
                results.append(enhanced)
                print(f"   ✓ Enhanced image {idx+1}/{len(image_arrays)}")

            # Return single image or batch
            if len(results) == 1:
                result_bytes = image_to_bytes(results[0], ".jpg", quality=95)
                elapsed_ms = (time.time() - start_time) * 1000
                return Response(
                    content=result_bytes,
                    media_type="image/jpeg",
                    headers={
                        "Content-Disposition": f'attachment; filename="hdrit_enhance.jpg"',
                        "Content-Length": str(len(result_bytes)),
                        "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
                        "X-Images-Processed": "1",
                        "X-Processor": f"Pro v{PRO_VERSION}" if HAS_PRO_PROCESSOR else "Standard",
                        "Access-Control-Allow-Origin": "*",
                    }
                )
            else:
                # Multiple images - return as ZIP
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, result in enumerate(results):
                        img_bytes = image_to_bytes(result, ".jpg", quality=95)
                        zf.writestr(f"hdrit_enhance_{i+1:02d}.jpg", img_bytes)
                zip_buffer.seek(0)
                elapsed_ms = (time.time() - start_time) * 1000
                return Response(
                    content=zip_buffer.getvalue(),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f'attachment; filename="hdrit_enhance_batch.zip"',
                        "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
                        "X-Images-Processed": str(len(results)),
                    }
                )

        # HDR MODE: Single image enhancement or bracket merge
        if len(image_arrays) > 1:

            # Smart grouping for 3+ images: detect scenes via EXIF
            if len(image_arrays) > 2 and HAS_SMART_GROUPING:
                print(f"   🧠 Smart grouping {len(image_arrays)} images by scene...")
                scene_groups = await asyncio.to_thread(
                    group_by_scene, image_arrays, raw_bytes_list
                )
                print(f"   📂 Detected {len(scene_groups)} scene group(s)")

                results = []
                group_labels = []
                for group in scene_groups:
                    if group.group_type == "single":
                        # Single image — run through standard processor
                        print(f"   🖼️  {group.scene_id}: single image → standard processing")
                        if HAS_BULLETPROOF:
                            bp_s = BulletproofSettings(preset='professional', denoise_strength='heavy')
                            proc = BulletproofProcessor(bp_s)
                        else:
                            raise HTTPException(500, "No processor available")
                        if HAS_TURBO:
                            proc = TurboProcessor(proc)
                        result = await asyncio.to_thread(proc.process, group.images[0])
                        results.append(result)
                        group_labels.append(f"{group.scene_id}_single")

                    elif group.group_type == "duplicate":
                        # Duplicates — pick sharpest, then enhance
                        print(f"   🔍 {group.scene_id}: {len(group.images)} duplicates → picking sharpest")
                        best = await asyncio.to_thread(pick_sharpest, group.images)
                        if HAS_BULLETPROOF:
                            bp_s = BulletproofSettings(preset='professional', denoise_strength='heavy')
                            proc = BulletproofProcessor(bp_s)
                        else:
                            raise HTTPException(500, "No processor available")
                        if HAS_TURBO:
                            proc = TurboProcessor(proc)
                        result = await asyncio.to_thread(proc.process, best)
                        results.append(result)
                        group_labels.append(f"{group.scene_id}_best")

                    elif group.group_type == "bracket":
                        # Brackets — sharpness-weighted Mertens fusion
                        print(f"   📸 {group.scene_id}: {len(group.images)} brackets → sharpness-weighted merge")
                        result = await asyncio.to_thread(merge_with_sharpness_weights, group.images)
                        # Post-process the merged result
                        if HAS_BULLETPROOF:
                            bp_s = BulletproofSettings(preset='professional', denoise_strength='medium')
                            proc = BulletproofProcessor(bp_s)
                        else:
                            raise HTTPException(500, "No processor available")
                        if HAS_TURBO:
                            proc = TurboProcessor(proc)
                        result = await asyncio.to_thread(proc.process, result)
                        results.append(result)
                        group_labels.append(f"{group.scene_id}_merged")

                elapsed_ms = (time.time() - start_time) * 1000

                # Single group → return single JPEG
                if len(results) == 1:
                    result_bytes = image_to_bytes(results[0], ".jpg", quality=95)
                    print(f"   ✓ Smart grouping complete in {elapsed_ms:.0f}ms (1 group)")
                    return Response(
                        content=result_bytes,
                        media_type="image/jpeg",
                        headers={
                            "Content-Disposition": f'attachment; filename="hdrit_{mode}.jpg"',
                            "Content-Length": str(len(result_bytes)),
                            "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
                            "X-Images-Processed": str(len(images)),
                            "X-Processor": "Smart Grouping + Bulletproof",
                            "X-Scene-Groups": "1",
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Expose-Headers": "Content-Length, X-Processing-Time-Ms, X-Processor, X-Scene-Groups",
                            "Cache-Control": "no-cache",
                        }
                    )

                # Multiple groups → return ZIP
                import zipfile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, (result, label) in enumerate(zip(results, group_labels)):
                        img_bytes = image_to_bytes(result, ".jpg", quality=95)
                        zf.writestr(f"hdrit_{label}.jpg", img_bytes)
                zip_buffer.seek(0)
                print(f"   ✓ Smart grouping complete in {elapsed_ms:.0f}ms ({len(results)} groups)")
                return Response(
                    content=zip_buffer.getvalue(),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f'attachment; filename="hdrit_scenes.zip"',
                        "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
                        "X-Images-Processed": str(len(images)),
                        "X-Scene-Groups": str(len(results)),
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Expose-Headers": "X-Processing-Time-Ms, X-Scene-Groups",
                    }
                )

            # Direct bracket merge (2 images, or smart grouping unavailable)
            if HAS_V14_GOLDEN:
                processor = V14GoldenProcessor(V14Settings())
                proc_version = f"V14 Golden v{V14_VERSION}"
            elif HAS_BULLETPROOF:
                bp_settings = BulletproofSettings(
                    preset='professional',
                    denoise_strength='heavy',
                )
                processor = BulletproofProcessor(bp_settings)
                proc_version = f"Bulletproof v{BP_VERSION}"
            else:
                raise HTTPException(500, "No bracket processor available")

            # Wrap with Turbo Mode for 4x speedup
            if HAS_TURBO:
                processor = TurboProcessor(processor)
                proc_version += " + Turbo"

            print(f"   🚀 Using {proc_version} for {len(image_arrays)} brackets...")

            # Run CPU-heavy bracket merge in thread pool (non-blocking)
            result = await asyncio.to_thread(processor.process_brackets, image_arrays)

            # Encode and return
            result_bytes = image_to_bytes(result, ".jpg", quality=95)
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"   ✓ Bracket merge complete in {elapsed_ms:.0f}ms")
            print(f"   📦 Response size: {len(result_bytes) / 1024:.1f} KB")

            return Response(
                content=result_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="hdrit_{mode}.jpg"',
                    "Content-Length": str(len(result_bytes)),
                    "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
                    "X-Images-Processed": str(len(images)),
                    "X-Processor": proc_version,
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Expose-Headers": "Content-Length, X-Processing-Time-Ms, X-Processor",
                    "Cache-Control": "no-cache",
                }
            )

        # Single image processing - V14 Golden Processor is PRIMARY
        base_image = image_arrays[0]

        if HAS_V14_GOLDEN:
            # V14 Golden Processor - THE ~90% AutoHDR match processor!
            print("   ⭐ Using V14 Golden Processor (THE golden processor)...")
            processor = V14GoldenProcessor(V14Settings())
            proc_version = f"V14 Golden v{V14_VERSION}"
        elif HAS_BULLETPROOF:
            # Fallback to Bulletproof
            print("   Applying HDR processing with Bulletproof Processor...")
            bp_settings = BulletproofSettings(
                preset='professional',
                denoise_strength='heavy',
            )
            processor = BulletproofProcessor(bp_settings)
            proc_version = f"Bulletproof v{BP_VERSION}"
        elif HAS_CLEAN_PROCESSOR:
            clean_settings = CleanSettings(preset='natural')
            processor = HDRitProcessor(clean_settings)
            proc_version = f"Clean v{CLEAN_VERSION}"
        else:
            # Legacy fallback
            settings = ProcessingSettings(
                brightness=brightness,
                contrast=contrast,
                vibrance=vibrance,
                white_balance=whiteBalance,
            )
            processor = AutoHDRProcessor(settings)
            proc_version = "Legacy"

        # Wrap with Turbo Mode for 4x speedup (GPU-accelerated denoising)
        if HAS_TURBO:
            processor = TurboProcessor(processor)
            proc_version += " + Turbo"

        # Run CPU-heavy processing in thread pool (non-blocking)
        result = await asyncio.to_thread(processor.process, base_image)

        # Encode and return
        result_bytes = image_to_bytes(result, ".jpg", quality=95)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"   ✓ Complete in {elapsed_ms:.0f}ms, output: {len(result_bytes)} bytes")

        return Response(
            content=result_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'attachment; filename="hdrit_{mode}.jpg"',
                "X-Processing-Time-Ms": str(round(elapsed_ms, 2)),
                "X-Images-Processed": str(len(images)),
                "X-Processor": proc_version,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        traceback.print_exc()
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
# OBSERVER AUTH PROXY
# ============================================
# Proxy Observer Auth requests to port 8001
# so everything works through a single ngrok tunnel

import httpx

OBSERVER_URL = "http://localhost:8001"

@app.post("/observer/start")
async def observer_start(request: Request):
    """Proxy auth start to Observer."""
    try:
        body = await request.body()
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OBSERVER_URL}/auth/start", content=body,
                                     headers={"Content-Type": "application/json"}, timeout=10)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/observer/check/{session_id}")
async def observer_check(session_id: str):
    """Proxy auth check to Observer."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OBSERVER_URL}/auth/check/{session_id}", timeout=10)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.post("/observer/approve/{session_id}")
async def observer_approve(session_id: str, request: Request):
    """Proxy auth approve to Observer."""
    try:
        body = await request.body()
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OBSERVER_URL}/auth/approve/{session_id}", content=body,
                                     headers={"Content-Type": "application/json"}, timeout=10)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.post("/observer/deny/{session_id}")
async def observer_deny(session_id: str, request: Request):
    """Proxy auth deny to Observer."""
    try:
        body = await request.body()
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OBSERVER_URL}/auth/deny/{session_id}", content=body,
                                     headers={"Content-Type": "application/json"}, timeout=10)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.post("/observer/code")
async def observer_code(request: Request):
    """Proxy code-based approval to Observer."""
    try:
        body = await request.body()
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{OBSERVER_URL}/auth/code", content=body,
                                     headers={"Content-Type": "application/json"}, timeout=10)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/observer/scan/{session_id}")
async def observer_scan(session_id: str, request: Request):
    """QR scan landing - auto-approves from phone with device_id."""
    device_id = request.query_params.get("device_id", "")
    device_name = request.query_params.get("device_name", "Phone Scanner")
    if device_id:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{OBSERVER_URL}/auth/approve/{session_id}",
                    json={"device_id": device_id, "device_name": device_name}, timeout=10)
                data = resp.json()
                if data.get("success"):
                    return JSONResponse({"status": "approved", "message": "Access granted!"})
        except:
            pass
    # Return a simple page that sends device info and approves
    from fastapi.responses import HTMLResponse
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observer - Approve Login</title>
<style>body{{background:#000;color:#0f0;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
.card{{text-align:center;padding:30px;border:1px solid #0f03;border-radius:16px;background:#0f01}}
button{{background:#0f0;color:#000;border:none;padding:15px 40px;border-radius:12px;font-size:18px;font-weight:bold;cursor:pointer;margin-top:20px}}
.deny{{background:#f00;color:#fff;margin-left:10px}}</style></head>
<body><div class="card">
<h1>🔐 Login Request</h1>
<p>Approve access to AXE?</p>
<button onclick="approve()">✓ Approve</button>
<button class="deny" onclick="deny()">✗ Deny</button>
<p id="status" style="margin-top:20px"></p>
<script>
const sid="{session_id}";
let did=localStorage.getItem('observer_device_id');
if(!did){{did=crypto.randomUUID();localStorage.setItem('observer_device_id',did)}}
async function approve(){{
  const r=await fetch('/observer/approve/'+sid,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{device_id:did,device_name:navigator.userAgent.slice(0,30)}})}});
  const d=await r.json();
  document.getElementById('status').textContent=d.success?'✓ Approved!':'Error: '+(d.error||'failed');
}}
async function deny(){{
  await fetch('/observer/deny/'+sid,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{device_id:did}})}});
  document.getElementById('status').textContent='✗ Denied';
}}
</script></div></body></html>""")

@app.get("/observer/app")
async def observer_app_redirect():
    """Redirect to Observer PWA."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("http://localhost:8001/app")


# ============================================
# OLLAMA PROXY (for Klaus frontend)
# ============================================

import httpx
import json as _json
import uuid
import re as _re

OLLAMA_BASE = "http://localhost:11434"
_ollama_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(path: str, request: Request):
    """Proxy requests to local Ollama instance."""
    url = f"{OLLAMA_BASE}/{path}"
    body = await request.body()

    is_stream = False
    if body:
        try:
            parsed = _json.loads(body)
            is_stream = parsed.get("stream", False)
            # Auto-disable thinking for qwen3 models (20x speed boost)
            if "qwen3" in parsed.get("model", "") and "think" not in parsed:
                parsed["think"] = False
                body = _json.dumps(parsed).encode()
        except Exception:
            pass

    if is_stream:
        req = _ollama_client.build_request(request.method, url, content=body,
                                            headers={"content-type": "application/json"})
        resp = await _ollama_client.send(req, stream=True)
        async def stream_response():
            try:
                async for line in resp.aiter_lines():
                    yield line.encode() + b"\n"
            finally:
                await resp.aclose()
        return StreamingResponse(stream_response(), status_code=resp.status_code,
                                 media_type="application/x-ndjson")
    else:
        resp = await _ollama_client.request(request.method, url, content=body,
                                             headers={"content-type": "application/json"})
        return Response(content=resp.content, status_code=resp.status_code,
                       media_type=resp.headers.get("content-type", "application/json"))

def _load_team_context(max_messages: int = 30) -> str:
    """Load recent team channel messages for Klaus memory context."""
    channel_path = os.path.expanduser("~/Desktop/M1transfer/axe-memory/team/channel.jsonl")
    context_path = os.path.expanduser("~/.axe/memory/active_context.json")
    lines = []
    try:
        with open(channel_path, "r") as f:
            all_lines = f.readlines()
            for raw in all_lines[-max_messages:]:
                try:
                    m = _json.loads(raw.strip())
                    lines.append(f"[{m.get('from','?')}→{m.get('to','team')}] {m.get('msg','')[:200]}")
                except Exception:
                    pass
    except Exception:
        pass
    context_parts = []
    if lines:
        context_parts.append("Recent team channel (Forge, Cortana, Klaus, James):\n" + "\n".join(lines))
    try:
        with open(context_path, "r") as f:
            ctx = _json.load(f)
            if ctx.get("current_projects"):
                context_parts.append("Active projects: " + _json.dumps(ctx["current_projects"][:5]))
            if ctx.get("critical_facts"):
                context_parts.append("Critical facts: " + _json.dumps(ctx["critical_facts"][:5]))
    except Exception:
        pass
    return "\n\n".join(context_parts)

# === Team Channel API ===
TEAM_CHANNEL_PATH = os.path.expanduser("~/Desktop/M1transfer/axe-memory/team/channel.jsonl")

class TeamMessage(BaseModel):
    from_agent: str = "unknown"
    to: str = "team"
    msg: str
    type: str = "message"

@app.post("/team/message")
async def post_team_message(payload: TeamMessage):
    """Post a message to the team channel. Used by all agents for comms."""
    import datetime as _dt
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "from": payload.from_agent,
        "to": payload.to,
        "type": payload.type,
        "msg": payload.msg,
    }
    with open(TEAM_CHANNEL_PATH, "a") as f:
        f.write(_json.dumps(entry) + "\n")
    return {"status": "sent", "ts": entry["ts"]}

@app.get("/team/messages")
async def get_team_messages(limit: int = 20):
    """Read recent team channel messages."""
    try:
        with open(TEAM_CHANNEL_PATH, "r") as f:
            all_lines = f.readlines()
        messages = []
        for raw in all_lines[-limit:]:
            try:
                messages.append(_json.loads(raw.strip()))
            except:
                pass
        return {"messages": messages}
    except FileNotFoundError:
        return {"messages": []}


# ============================================
# IMI ROUTER — All /klaus/imi/* endpoints
# ============================================
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routers.imi import router as imi_router, SURVEY_STORE as _imi_survey_store
# Share the survey store between main.py and the IMI router
import routers.imi as _imi_mod
_imi_mod.SURVEY_STORE = _SURVEY_STORE
app.include_router(imi_router)

from routers.reports import router as reports_router
app.include_router(reports_router)

from routers.audit import router as audit_router
app.include_router(audit_router)

from routers.documents import router as documents_router
app.include_router(documents_router)

from routers.sql import router as sql_router
app.include_router(sql_router)

from routers.search import router as search_router
app.include_router(search_router)

from routers.rfp import router as rfp_router
app.include_router(rfp_router)

from routers.auth import router as auth_router
app.include_router(auth_router)

from routers.meta_analysis import router as meta_analysis_router
app.include_router(meta_analysis_router)

from routers.anomalies import router as anomalies_router
app.include_router(anomalies_router)

from routers.case_studies import router as case_studies_router
app.include_router(case_studies_router)

from routers.notebook import router as notebook_router
app.include_router(notebook_router)

from routers.memory import router as memory_router
app.include_router(memory_router)

from routers.ml import router as ml_router
app.include_router(ml_router)

from routers.graph import router as graph_router
app.include_router(graph_router)

from routers.connections import router as connections_router
app.include_router(connections_router)

from routers.spreadsheet import router as spreadsheet_router
app.include_router(spreadsheet_router)

from routers.patterns import router as patterns_router
app.include_router(patterns_router)

from routers.warehouse import router as warehouse_router
app.include_router(warehouse_router)

# --- Legacy IMI endpoints removed — now in routers/imi.py ---
# 13 endpoints, ~960 lines extracted for clean architecture.
# See routers/imi.py for: chat, report, visualize, chart-types,
# dashboard, stats, ingest, generate-training, save, upload-survey,
# generate-deck, demo-deck, surveys

_IMI_MIGRATION_DONE = True  # marker for verification

# ============================================
# ============================================
# ============================================
# SURVEY ENGINE — IMI-Grade Analysis
# ============================================

import pandas as pd
import io as _io
import tempfile as _tempfile

# IMI Deck Builder
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from survey_bridge import survey_to_slides
from deck_renderer import build_deck
from flat_adapter import flat_to_structured
from narrative_polisher import polish_configs

# _SURVEY_STORE declared near app init (line ~111)
_SURVEY_DIR = os.path.expanduser("~/.axe/memory/surveys")
os.makedirs(_SURVEY_DIR, exist_ok=True)


def _parse_survey_file(filename: str, content: bytes) -> tuple[pd.DataFrame, dict]:
    """Parse xlsx/csv/json into DataFrame + metadata."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("xlsx", "xls"):
        xls = pd.ExcelFile(_io.BytesIO(content))
        # Find the sheet with actual data (most rows)
        best_sheet = None
        best_rows = 0
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            df = df.dropna(how="all")
            if len(df) > best_rows:
                best_rows = len(df)
                best_sheet = sheet_name
        df = pd.read_excel(xls, sheet_name=best_sheet, header=None)
    elif ext == "csv":
        df = pd.read_csv(_io.BytesIO(content), header=None)
    elif ext == "json":
        df = pd.read_json(_io.BytesIO(content))
    else:
        df = pd.read_csv(_io.BytesIO(content), header=None, sep="\t")

    # Clean: drop fully empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    metadata = {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "sheet": best_sheet if ext in ("xlsx", "xls") else None,
    }

    return df, metadata


def _df_to_text(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert DataFrame to readable text for LLM consumption."""
    # Use full precision for the data
    with pd.option_context('display.max_columns', None, 'display.width', None,
                           'display.max_colwidth', 50, 'display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1 else f'{x:.2f}'):
        subset = df.head(max_rows)
        return subset.to_string(index=False)


def _parse_crosstab_structured(file_bytes: bytes, filename: str) -> dict | None:
    """
    Smart crosstab parser: detects segment headers + question blocks.
    Returns structured JSON for <survey_data> injection, or None if detection fails.
    Falls back to raw text mode if structure can't be detected.
    """
    try:
        df = pd.read_excel(_io.BytesIO(file_bytes), header=None)
    except Exception:
        return None

    # --- Detect segment header row ---
    segment_keywords = [
        'canada', 'total', 'female', 'male', 'west', 'ontario',
        'quebec', 'atlantic', '18 to', '35 to', '55 and',
        'born in', 'born outside', 'under $', '$50', '$125',
        'national', 'overall', 'age', 'gender', 'region', 'income'
    ]

    segment_row = None
    for idx, row in df.iterrows():
        row_text = ' '.join(str(v).lower() for v in row.values if pd.notna(v))
        matches = sum(1 for kw in segment_keywords if kw in row_text)
        if matches >= 3:
            segment_row = idx
            break

    if segment_row is None:
        return None

    # --- Build segment mapping ---
    segments = {}
    for col_idx, val in enumerate(df.iloc[segment_row]):
        if pd.notna(val) and str(val).strip():
            segments[col_idx] = str(val).strip()

    # --- Detect question blocks ---
    questions = []
    current_q = None
    last_base_sizes = {}

    for idx in range(segment_row + 1, len(df)):
        row = df.iloc[idx]
        first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''

        # Question header (Q followed by number)
        if first_cell and first_cell.startswith('Q') and any(c.isdigit() for c in first_cell[:4]):
            if current_q:
                questions.append(current_q)
            current_q = {
                'id': first_cell.split(' ')[0] if ' ' in first_cell else first_cell,
                'text': first_cell,
                'options': []
            }
            continue

        # Sample size row
        if 'sample' in first_cell.lower() or 'n=' in first_cell.lower() or 'base' in first_cell.lower():
            base_sizes = {}
            for col_idx, seg_label in segments.items():
                val = row.iloc[col_idx] if col_idx < len(row) else None
                if pd.notna(val):
                    try:
                        base_sizes[seg_label] = int(float(val))
                    except (ValueError, TypeError):
                        pass
            if current_q:
                current_q['base_sizes'] = base_sizes
            last_base_sizes = base_sizes
            continue

        # Option row (label + numeric values)
        if first_cell and current_q:
            values = {}
            has_numeric = False
            for col_idx, seg_label in segments.items():
                val = row.iloc[col_idx] if col_idx < len(row) else None
                if pd.notna(val):
                    try:
                        num_val = float(val)
                        if 0 < num_val < 1:
                            num_val = round(num_val * 100, 1)
                        values[seg_label] = num_val
                        has_numeric = True
                    except (ValueError, TypeError):
                        pass

            if has_numeric:
                current_q['options'].append({
                    'label': first_cell,
                    'values': values
                })

    if current_q:
        questions.append(current_q)

    if not questions:
        return None

    # Determine total_n from last known base sizes
    total_n = 0
    if last_base_sizes:
        for key in ['Canada', 'Total', 'National', 'Overall']:
            if key in last_base_sizes:
                total_n = last_base_sizes[key]
                break
        if total_n == 0 and last_base_sizes:
            total_n = list(last_base_sizes.values())[0]

    return {
        'survey_name': filename.replace('.xlsx', '').replace('.csv', '').replace('_', ' '),
        'total_n': total_n,
        'segments': list(segments.values()),
        'questions': questions
    }


def _validate_response(response_text: str, survey_data: dict) -> dict:
    """
    Post-generation anti-hallucination check.
    Scans response for percentages, cross-references against source data.
    """
    warnings = []

    # Extract all percentages from response
    percentages = _re.findall(r'(\d+(?:\.\d+)?)\s*%', response_text)

    # Build set of valid numbers from survey data
    valid_numbers = set()
    for q in survey_data.get('questions', []):
        for option in q.get('options', []):
            for seg, val in option.get('values', {}).items():
                valid_numbers.add(str(round(val, 1)))
                valid_numbers.add(str(int(val)))
        for seg, val in q.get('base_sizes', {}).items():
            valid_numbers.add(str(val))

    for pct in percentages:
        if pct not in valid_numbers:
            try:
                pct_float = float(pct)
                # Allow derived metrics (gaps, indices, combined scores)
                if pct_float > 100:
                    continue
            except ValueError:
                pass
            warnings.append(f"{pct}% not found in source data")

    # Detect hedging language before numbers (hallucination signal)
    hedge_patterns = [
        r'approximately\s+\d', r'roughly\s+\d', r'about\s+\d',
        r'I believe\s+\d', r'typically\s+\d', r'usually\s+\d'
    ]
    for pattern in hedge_patterns:
        if _re.search(pattern, response_text, _re.IGNORECASE):
            warnings.append("Hedging language before number detected")
            break

    return {"valid": len(warnings) == 0, "warnings": warnings[:10]}


def _get_temperature(question: str) -> float:
    """Dynamic temperature based on query type."""
    q_lower = question.lower()
    deck_triggers = ['deck', 'presentation', 'slides', 'full analysis', 'insight report', 'analyze']
    strategic_triggers = ['should', 'recommend', 'strategy', 'implication', 'opportunity']

    if any(t in q_lower for t in deck_triggers):
        return 0.4
    elif any(t in q_lower for t in strategic_triggers):
        return 0.3
    else:
        return 0.1


def _build_imi_analysis_prompt(survey_text: str, filename: str) -> str:
    """Build the IMI-style analysis prompt."""
    return f"""You are an elite data analyst at IMI International / ConsultIMI — a global consultancy known for "Insight. Driving. Profit."

You have been given raw survey crosstab data from the file "{filename}". Your job is to produce a full IMI-grade insight report.

SURVEY DATA:
```
{survey_text}
```

Produce a comprehensive analysis following this EXACT structure. Use the actual numbers from the data above — never fabricate statistics.

## REPORT STRUCTURE:

### 1. EXECUTIVE SUMMARY
- 3 big-stat callout cards (the most surprising/important numbers)
- Each stat should have a bold number and a 1-sentence insight

### 2. NATIONAL RESULTS (Overall/Total column)
- Rank all response options from highest to lowest
- Show percentages
- Note what's surprising vs expected

### 3. COMBINED ANALYSIS (if 1st + 2nd choice or combined data exists)
- Show combined reach/preference
- Identify the real competitive landscape
- Note any virtual ties or surprising rankings

### 4. DEMOGRAPHIC DEEP-DIVES
For EACH demographic break in the crosstab (age, gender, region, income, nativity, etc.):
- Identify the biggest over-index and under-index vs national average
- Calculate index scores (demo % / national % × 100)
- Highlight statistically meaningful gaps (>5 percentage points)
- Name the key insight for each demographic

### 5. THE HEADLINE INSIGHT
- What's the single most important finding?
- The "showstopper" slide — the insight that would make a CMO sit up
- Include the specific comparison numbers

### 6. STRATEGIC IMPLICATIONS
- 4 actionable "so what?" recommendations
- Each should name: the insight, the audience, and the sponsor/brand action
- Frame as business opportunities, not just data observations

### 7. KEY QUOTES FOR SLIDES
- For each major finding, write a "KEY INSIGHT" or "SO WHAT?" callout (1-2 sentences, punchy, executive-ready)

FORMATTING RULES:
- Use actual percentages from the data (convert decimals to % where needed — e.g., 0.27 = 27%)
- Calculate index scores where relevant (demo/total × 100)
- Bold the most important numbers
- Use IMI's signature phrases: "Insight. Driving. Profit.", "KEY INSIGHT:", "SO WHAT?"
- Be specific — cite sample sizes, exact percentages, exact gaps
- If multiple questions exist in the data, analyze each one
- End with a note on methodology (sample size, crosstab dimensions)

Write the full report now."""


@app.post("/klaus/survey/load")
async def survey_load(request: Request):
    """Load a survey file (xlsx, csv, json). Accepts multipart form or JSON with base64."""
    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse({"error": "No file provided"}, status_code=400)
        filename = file.filename
        content = await file.read()
    else:
        body = await request.json()
        filename = body.get("filename", "upload.csv")
        import base64
        content = base64.b64decode(body.get("data", ""))

    try:
        df, metadata = _parse_survey_file(filename, content)
    except Exception as e:
        return JSONResponse({"error": f"Failed to parse file: {str(e)}"}, status_code=400)

    survey_id = str(uuid.uuid4())[:8]
    raw_text = _df_to_text(df, max_rows=200)

    # Try smart structured parsing: crosstab first, then flat format
    structured = _parse_crosstab_structured(content, filename)
    if not structured:
        structured = flat_to_structured(content, filename)

    _SURVEY_STORE[survey_id] = {
        "name": filename,
        "df": df,
        "raw_text": raw_text,
        "metadata": metadata,
        "structured_data": structured,
    }

    # Persist raw text for future sessions
    with open(os.path.join(_SURVEY_DIR, f"{survey_id}.txt"), "w") as f:
        f.write(f"# {filename}\n\n{raw_text}")

    return JSONResponse({
        "survey_id": survey_id,
        "name": filename,
        "columns": list(df.columns) if not df.empty else [],
        "row_count": len(df),
        "preview": raw_text[:1000],
    })


@app.get("/klaus/survey/list")
async def survey_list():
    """List loaded surveys."""
    surveys = []
    for sid, s in _SURVEY_STORE.items():
        surveys.append({
            "id": sid,
            "name": s["name"],
            "rows": s["metadata"]["rows"],
            "columns": s["metadata"]["columns"],
        })
    return JSONResponse({"surveys": surveys})


@app.get("/klaus/survey/{survey_id}/summary")
async def survey_summary(survey_id: str):
    """Get survey summary."""
    survey = _SURVEY_STORE.get(survey_id)
    if not survey:
        return JSONResponse({"error": "Survey not found"}, status_code=404)

    df = survey["df"]
    return JSONResponse({
        "survey_id": survey_id,
        "name": survey["name"],
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "preview": survey["raw_text"][:2000],
        "metadata": survey["metadata"],
    })


@app.post("/klaus/survey/query")
async def survey_query(request: Request):
    """Query a survey with natural language — sends data + question to Ollama."""
    body = await request.json()
    survey_id = body.get("survey_id", "")
    question = body.get("question", "")

    survey = _SURVEY_STORE.get(survey_id)
    if not survey:
        return JSONResponse({"error": "Survey not found"}, status_code=404)

    # Use structured data if available, fall back to raw text
    structured = survey.get("structured_data")
    if structured:
        data_block = f"<survey_data>\n{_json.dumps(structured, indent=2)}\n</survey_data>\n\nThe above is the ONLY data you may reference. Every number in your response must come from this data."
    else:
        data_block = f"<survey_data>\nSURVEY: {survey['name']}\n{survey['raw_text']}\n</survey_data>\n\nThe above is the ONLY data you may reference. Every number in your response must come from this data."

    messages = [
        {"role": "system", "content": data_block},
        {"role": "user", "content": question},
    ]

    temperature = _get_temperature(question)

    ollama_payload = {
        "model": "klaus-imi",
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": temperature, "top_p": 0.85, "num_ctx": 32768},
    }

    req = _ollama_client.build_request(
        "POST", f"{OLLAMA_BASE}/api/chat",
        content=_json.dumps(ollama_payload).encode(),
        headers={"content-type": "application/json"},
    )
    resp = await _ollama_client.send(req, stream=True)

    async def stream_response():
        try:
            async for line in resp.aiter_lines():
                yield line.encode() + b"\n"
        finally:
            await resp.aclose()

    return StreamingResponse(stream_response(), status_code=resp.status_code,
                             media_type="application/x-ndjson")


@app.post("/klaus/survey/{survey_id}/analyze")
async def survey_analyze(survey_id: str):
    """Full IMI-style analysis — the 11-slide framework as streaming response."""
    survey = _SURVEY_STORE.get(survey_id)
    if not survey:
        return JSONResponse({"error": "Survey not found"}, status_code=404)

    prompt = _build_imi_analysis_prompt(survey["raw_text"], survey["name"])

    # Use structured data injection if available
    structured = survey.get("structured_data")
    if structured:
        data_inject = f"<survey_data>\n{_json.dumps(structured, indent=2)}\n</survey_data>"
    else:
        data_inject = f"<survey_data>\n{survey['raw_text']}\n</survey_data>"

    messages = [
        {"role": "system", "content": f"{data_inject}\n\nThe above is the ONLY data you may reference."},
        {"role": "user", "content": prompt},
    ]

    ollama_payload = {
        "model": "klaus-imi",
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.4, "top_p": 0.85, "num_ctx": 32768},
    }

    req = _ollama_client.build_request(
        "POST", f"{OLLAMA_BASE}/api/chat",
        content=_json.dumps(ollama_payload).encode(),
        headers={"content-type": "application/json"},
    )
    resp = await _ollama_client.send(req, stream=True)

    async def stream_analysis():
        try:
            async for line in resp.aiter_lines():
                yield line.encode() + b"\n"
        finally:
            await resp.aclose()

    return StreamingResponse(stream_analysis(), status_code=resp.status_code,
                             media_type="application/x-ndjson")


@app.post("/klaus/survey/{survey_id}/deck")
async def survey_deck(survey_id: str):
    """Generate IMI-branded .pptx deck from loaded survey data."""
    survey = _SURVEY_STORE.get(survey_id)
    if not survey:
        return JSONResponse({"error": "Survey not found"}, status_code=404)

    structured = survey.get("structured_data")
    if not structured:
        return JSONResponse(
            {"error": "No structured data available for this survey. Re-upload as xlsx crosstab."},
            status_code=400,
        )

    try:
        configs = survey_to_slides(structured)
        configs = polish_configs(configs, structured)
        output_path = os.path.join(_tempfile.gettempdir(), f"{survey_id}_deck.pptx")
        build_deck(configs, output_path)
    except Exception as e:
        return JSONResponse({"error": f"Deck generation failed: {str(e)}"}, status_code=500)

    safe_name = structured.get("survey_name", survey_id).replace(" ", "_")
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=f"{safe_name}_IMI_Deck.pptx",
    )


@app.post("/klaus/survey/{survey_id}/chat-analysis")
async def survey_chat_analysis(survey_id: str):
    """
    Generate the full 11-section IMI analysis as structured text for chat display,
    AND pre-build the deck so a download link is ready immediately.
    Returns JSON with 'analysis' (markdown text) and 'deck_url'.
    """
    survey = _SURVEY_STORE.get(survey_id)
    if not survey:
        return JSONResponse({"error": "Survey not found"}, status_code=404)

    structured = survey.get("structured_data")
    if not structured:
        return JSONResponse({"error": "No structured data for this survey."}, status_code=400)

    try:
        configs = survey_to_slides(structured)
        configs = polish_configs(configs, structured)
    except Exception as e:
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)

    # Build human-readable markdown from the slide configs
    analysis = _configs_to_markdown(configs, structured)

    # Pre-build the deck
    deck_path = os.path.join(_tempfile.gettempdir(), f"{survey_id}_deck.pptx")
    try:
        build_deck(configs, deck_path)
        deck_ready = True
    except Exception:
        deck_ready = False

    return JSONResponse({
        "survey_id": survey_id,
        "survey_name": structured.get("survey_name", ""),
        "analysis": analysis,
        "deck_available": deck_ready,
        "deck_url": f"/klaus/survey/{survey_id}/deck" if deck_ready else None,
    })


def _configs_to_markdown(configs: dict, structured: dict) -> str:
    """Convert slide configs into a formatted markdown analysis for chat."""
    lines = []
    name = structured.get("survey_name", "Survey")
    total_n = structured.get("total_n", 0)

    # 1. Title
    s1 = configs.get("slide_1", {})
    lines.append(f"# {s1.get('title', name)}")
    lines.append(f"*{s1.get('subtitle', '')}*")
    lines.append(f"_{s1.get('methodology', f'n = {total_n}')}_")
    lines.append("")

    # 2. Executive Summary
    s2 = configs.get("slide_2", {})
    lines.append("## Executive Summary")
    for stat in s2.get("stats", []):
        lines.append(f"- **{stat['number']}** — {stat['description']}")
    lines.append("")

    # 3. National Results
    s3 = configs.get("slide_3", {})
    lines.append(f"## {s3.get('title', 'National Results')}")
    if s3.get("subtitle"):
        lines.append(f"*{s3['subtitle']}*")
    for label, val in s3.get("data", {}).items():
        lines.append(f"- **{val}%** — {label}")
    if s3.get("insight"):
        lines.append(f"\n> {s3['insight']}")
    lines.append("")

    # 4. Combined Preference
    s4 = configs.get("slide_4", {})
    lines.append(f"## {s4.get('title', 'Combined Preference')}")
    if s4.get("subtitle"):
        lines.append(f"*{s4['subtitle']}*")
    for label, val in s4.get("data", {}).items():
        pct = f"{val}%" if isinstance(val, (int, float)) else str(val)
        lines.append(f"- **{pct}** — {label}")
    if s4.get("insight"):
        lines.append(f"\n> {s4['insight']}")
    lines.append("")

    # 5. Biggest Divide
    s5 = configs.get("slide_5", {})
    lines.append(f"## {s5.get('title', 'The Biggest Divide')}")
    if s5.get("subtitle"):
        lines.append(f"*{s5['subtitle']}*")
    left_label = s5.get("left_label", "Group A")
    right_label = s5.get("right_label", "Group B")
    left_n = f" (n={s5['left_n']})" if s5.get("left_n") else ""
    right_n = f" (n={s5['right_n']})" if s5.get("right_n") else ""
    lines.append(f"\n**{left_label}**{left_n}:")
    for label, val in s5.get("left_data", {}).items():
        lines.append(f"  - {val}% — {label}")
    lines.append(f"\n**{right_label}**{right_n}:")
    for label, val in s5.get("right_data", {}).items():
        lines.append(f"  - {val}% — {label}")
    if s5.get("so_what"):
        lines.append(f"\n**SO WHAT?** {s5['so_what']}")
    lines.append("")

    # 6. Age/Generational
    s6 = configs.get("slide_6", {})
    lines.append(f"## {s6.get('title', 'Generational Lens')}")
    for label, age_vals in s6.get("data", {}).items():
        if isinstance(age_vals, dict):
            parts = [f"{k}: {v}%" for k, v in age_vals.items()]
            lines.append(f"- **{label}**: {' | '.join(parts)}")
    if s6.get("insight"):
        lines.append(f"\n> {s6['insight']}")
    lines.append("")

    # 7. Gender & Regional
    s7 = configs.get("slide_7", {})
    lines.append(f"## {s7.get('title', 'Gender & Regional Lens')}")
    if s7.get("gender_data"):
        lines.append("\n| Event | Female | Male | Gap |")
        lines.append("|-------|--------|------|-----|")
        for row in s7["gender_data"]:
            lines.append(f"| {' | '.join(str(c) for c in row)} |")
    if s7.get("regions"):
        lines.append("\n**Regional highlights:**")
        for r in s7["regions"]:
            lines.append(f"- **{r['name']}**: {r['insight']}")
    if s7.get("so_what"):
        lines.append(f"\n**SO WHAT?** {s7['so_what']}")
    lines.append("")

    # 8. Aspirational
    s8 = configs.get("slide_8", {})
    lines.append(f"## {s8.get('title', 'Preference Landscape')}")
    for label, val in s8.get("data", {}).items():
        lines.append(f"- **{val}%** — {label}")
    if s8.get("insight"):
        lines.append(f"\n> {s8['insight']}")
    lines.append("")

    # 9. Income
    s9 = configs.get("slide_9", {})
    lines.append(f"## {s9.get('title', 'Income Analysis')}")
    for label, inc_vals in s9.get("data", {}).items():
        if isinstance(inc_vals, dict):
            parts = [f"{k}: {v}%" for k, v in inc_vals.items()]
            lines.append(f"- **{label}**: {' | '.join(parts)}")
    if s9.get("insight"):
        lines.append(f"\n> {s9['insight']}")
    lines.append("")

    # 10. Strategic Implications
    s10 = configs.get("slide_10", {})
    lines.append("## Strategic Implications")
    for i, rec in enumerate(s10.get("recommendations", []), 1):
        lines.append(f"\n### {i}. {rec['title']}")
        lines.append(rec["body"])
    lines.append("")

    # 11. Methodology
    lines.append("---")
    lines.append(f"*Source: IMI Pulse™ | {name} | n = {total_n}*")

    return "\n".join(lines)


@app.post("/klaus/survey/upload-and-analyze")
async def survey_upload_and_analyze(request: Request):
    """One-shot: upload file and get full IMI analysis. Accepts multipart form."""
    content_type = request.headers.get("content-type", "")

    if "multipart" not in content_type:
        return JSONResponse({"error": "Must use multipart/form-data"}, status_code=400)

    form = await request.form()
    file = form.get("file")
    if not file:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    filename = file.filename
    content = await file.read()

    try:
        df, metadata = _parse_survey_file(filename, content)
    except Exception as e:
        return JSONResponse({"error": f"Failed to parse: {str(e)}"}, status_code=400)

    survey_id = str(uuid.uuid4())[:8]
    raw_text = _df_to_text(df, max_rows=200)

    # Try smart structured parsing: crosstab first, then flat format
    structured = _parse_crosstab_structured(content, filename)
    if not structured:
        structured = flat_to_structured(content, filename)

    _SURVEY_STORE[survey_id] = {
        "name": filename,
        "df": df,
        "raw_text": raw_text,
        "metadata": metadata,
        "structured_data": structured,
    }

    prompt = _build_imi_analysis_prompt(raw_text, filename)

    # Use structured data if available
    if structured:
        data_inject = f"<survey_data>\n{_json.dumps(structured, indent=2)}\n</survey_data>"
    else:
        data_inject = f"<survey_data>\n{raw_text}\n</survey_data>"

    messages = [
        {"role": "system", "content": f"{data_inject}\n\nThe above is the ONLY data you may reference."},
        {"role": "user", "content": prompt},
    ]

    ollama_payload = {
        "model": "klaus-imi",
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.4, "top_p": 0.85, "num_ctx": 32768},
    }

    req = _ollama_client.build_request(
        "POST", f"{OLLAMA_BASE}/api/chat",
        content=_json.dumps(ollama_payload).encode(),
        headers={"content-type": "application/json"},
    )
    resp = await _ollama_client.send(req, stream=True)

    async def stream_full():
        # First emit metadata
        yield _json.dumps({"type": "metadata", "survey_id": survey_id, "name": filename, "rows": len(df), "columns": len(df.columns)}).encode() + b"\n"
        # Then stream analysis
        try:
            async for line in resp.aiter_lines():
                yield line.encode() + b"\n"
        finally:
            await resp.aclose()

    return StreamingResponse(stream_full(), status_code=200,
                             media_type="application/x-ndjson")


@app.post("/klaus/file/upload")
async def klaus_file_upload(request: Request):
    """General file upload for chat context. Parses xlsx/csv and returns summary."""
    content_type = request.headers.get("content-type", "")
    if "multipart" not in content_type:
        return JSONResponse({"success": False, "error": "Must use multipart/form-data"}, status_code=400)

    form = await request.form()
    file = form.get("file")
    if not file:
        return JSONResponse({"success": False, "error": "No file provided"}, status_code=400)

    filename = file.filename
    content = await file.read()

    try:
        df, metadata = _parse_survey_file(filename, content)
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Failed to parse file: {str(e)}", "filename": filename})

    upload_id = str(uuid.uuid4())[:8]
    raw_text = _df_to_text(df, max_rows=50)

    # Store for later reference
    _SURVEY_STORE[upload_id] = {
        "name": filename,
        "df": df,
        "raw_text": raw_text,
        "metadata": metadata,
        "structured_data": None,
    }

    columns = [str(c) for c in df.columns.tolist()]
    summary = f"**{filename}** — {len(df)} rows, {len(columns)} columns\n\nColumns: {', '.join(columns[:15])}"
    if len(columns) > 15:
        summary += f" (+{len(columns)-15} more)"

    return JSONResponse({
        "success": True,
        "filename": filename,
        "upload_id": upload_id,
        "summary": summary,
        "columns": columns,
        "row_count": len(df),
    })


# KLAUS/OLLAMA PROXY (for klaus.it.com)
# ============================================

import httpx

OLLAMA_URL = "http://localhost:11434"

# NOTE: Duplicate /ollama proxy removed — primary is at line ~1596 with streaming support

@app.get("/ollama-health")
async def ollama_health():
    """Check if Ollama is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                return {"status": "healthy", "ollama": True}
    except:
        pass
    return {"status": "unhealthy", "ollama": False}


# ============================================
# KLAUS STATIC FRONTEND (bypass broken Vercel deploy)
# ============================================

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

KLAUS_STATIC_DIR = os.path.expanduser("~/klausimi-backend/klaus-static")

@app.get("/klaus")
@app.get("/klaus/")
async def klaus_index():
    """Serve Klaus frontend index.html"""
    index_path = os.path.join(KLAUS_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "Klaus frontend not found"}, status_code=404)

# Serve Klaus static assets
if os.path.exists(KLAUS_STATIC_DIR):
    app.mount("/klaus/assets", StaticFiles(directory=os.path.join(KLAUS_STATIC_DIR, "assets")), name="klaus-assets")


# ============================================
# KLAUS INTEGRATIONS API
# ============================================

from src.api.klaus_integrations import integrations, parse_and_execute

@app.post("/klaus/execute")
async def klaus_execute(request: Request):
    """Execute a Klaus integration action"""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        body = await request.json()
        action = body.get("action")
        params = body.get("params", {})

        if not action:
            return JSONResponse({"error": "No action specified"}, status_code=400, headers=cors_headers)

        result = integrations.execute(action, params)
        return JSONResponse(result, headers=cors_headers)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500, headers=cors_headers)


@app.get("/klaus/integrations")
async def klaus_list_integrations():
    """List available Klaus integrations"""
    return JSONResponse({
        "integrations": [
            {"name": "imessage", "description": "Send iMessages", "params": ["to", "message"]},
            {"name": "calendar", "description": "Check calendar events", "params": ["days"]},
            {"name": "notes", "description": "Create Apple Notes", "params": ["title", "body"]},
            {"name": "reminders", "description": "Create reminders", "params": ["title", "due"]},
            {"name": "web_search", "description": "Search the web", "params": ["query", "num_results"]},
            {"name": "filesystem", "description": "File operations", "params": ["operation", "path", "content"]},
        ],
        "status": "active"
    })


# ============================================
# SKILLS API (for Team Dashboard)
# ============================================

from pathlib import Path as PathLib

SKILLS_DIR = PathLib.home() / ".axe" / "skills"
SKILLS_INDEX = SKILLS_DIR / "SKILLS_INDEX.md"


def parse_skills_index():
    """Parse SKILLS_INDEX.md to extract skill metadata."""
    categories = {}
    skills = {}

    try:
        if not SKILLS_INDEX.exists():
            return {"categories": {}, "skills": {}, "total": 0}

        content = SKILLS_INDEX.read_text()
        lines = content.split('\n')

        current_category = 'Uncategorized'

        for line in lines:
            # Parse category headers like "### 🤖 AI & Language (6 skills)"
            import re
            category_match = re.match(r'^### (.+?) \((\d+) skills?\)', line)
            if category_match:
                current_category = category_match.group(1).strip()
                categories[current_category] = []
                continue

            # Parse skill entries like "- `skill_01_code_review` - Code quality analysis"
            skill_match = re.match(r'^- `(skill_(\d+)_([^`]+))` - (.+)$', line)
            if skill_match:
                full_id, num, name, description = skill_match.groups()
                skill = {
                    "id": full_id,
                    "number": int(num),
                    "name": name.replace('_', ' '),
                    "description": description.strip(),
                    "category": current_category,
                    "path": str(SKILLS_DIR / f"{full_id}.py")
                }
                skills[full_id] = skill
                if current_category not in categories:
                    categories[current_category] = []
                categories[current_category].append(skill)

        return {"categories": categories, "skills": skills, "total": len(skills)}
    except Exception as e:
        print(f"Error parsing skills index: {e}")
        return {"categories": {}, "skills": {}, "total": 0}


@app.get("/skills")
async def list_skills():
    """List all Klaus skills with metadata."""
    index = parse_skills_index()
    files = []
    try:
        if SKILLS_DIR.exists():
            files = sorted([f.name for f in SKILLS_DIR.glob("skill_*.py")])
    except Exception as e:
        print(f"Error listing skills: {e}")

    return JSONResponse({
        "success": True,
        **index,
        "files": files,
        "skillsDir": str(SKILLS_DIR)
    })


@app.get("/skills/{skill_id}")
async def get_skill(skill_id: str):
    """Get skill content by ID."""
    skill_path = SKILLS_DIR / f"{skill_id}.py"

    if not skill_path.exists():
        return JSONResponse({"success": False, "error": f"Skill {skill_id} not found"}, status_code=404)

    try:
        content = skill_path.read_text()
        index = parse_skills_index()
        skill = index["skills"].get(skill_id, {"id": skill_id, "name": skill_id})

        return JSONResponse({
            "success": True,
            "skill": skill,
            "content": content,
            "lines": len(content.split('\n'))
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.put("/skills/{skill_id}")
async def update_skill(skill_id: str, request: Request):
    """Update skill content."""
    skill_path = SKILLS_DIR / f"{skill_id}.py"

    if not skill_path.exists():
        return JSONResponse({"success": False, "error": f"Skill {skill_id} not found"}, status_code=404)

    try:
        body = await request.json()
        content = body.get("content")

        if not content:
            return JSONResponse({"success": False, "error": "Content is required"}, status_code=400)

        skill_path.write_text(content)

        return JSONResponse({
            "success": True,
            "message": f"Skill {skill_id} updated",
            "lines": len(content.split('\n'))
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============================================
# TEAM CHANNEL API (for Team Dashboard)
# ============================================

TEAM_CHANNEL_FILE = PathLib.home() / "Desktop" / "M1transfer" / "axe-memory" / "team" / "channel.jsonl"
import json
from datetime import datetime, timezone


@app.get("/channel")
async def get_channel_messages():
    """Get team channel messages."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        if not TEAM_CHANNEL_FILE.exists():
            return JSONResponse({"success": True, "messages": []}, headers=cors_headers)

        messages = []
        with open(TEAM_CHANNEL_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        messages.append(msg)
                    except json.JSONDecodeError:
                        continue

        # Return last 100 messages
        return JSONResponse({
            "success": True,
            "messages": messages[-100:],
            "total": len(messages)
        }, headers=cors_headers)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500, headers=cors_headers)


@app.post("/post")
async def post_to_channel(request: Request):
    """Post a message to team channel."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    try:
        body = await request.json()
        msg = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "from": body.get("from", "james"),
            "to": body.get("to", "team"),
            "type": body.get("type", "message"),
            "msg": body.get("msg", ""),
            "id": f"{int(datetime.utcnow().timestamp() * 1000)}"
        }

        # Append to channel file
        TEAM_CHANNEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TEAM_CHANNEL_FILE, 'a') as f:
            f.write(json.dumps(msg) + "\n")

        return JSONResponse({"success": True, "message": msg}, headers=cors_headers)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500, headers=cors_headers)


# ============================================
# WEBSOCKET HUB (Real-time Agent Communication)
# ============================================

# Connected agents via WebSocket
WS_AGENTS: dict = {}  # {websocket: agent_name}


async def ws_broadcast(message: str, exclude=None):
    """Send message to all connected WebSocket agents except sender"""
    for ws, name in list(WS_AGENTS.items()):
        if ws != exclude:
            try:
                await ws.send_text(message)
            except:
                pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent communication.

    Agents (Forge, Cortana, Klaus) connect here to chat in real-time.
    Messages are also persisted to channel.jsonl for history.

    Protocol:
    1. Connect and send: {"type": "connect", "agent": "cortana"}
    2. Send messages: {"from": "cortana", "to": "forge", "type": "message", "msg": "Hello!"}
    3. Receive broadcasts from other agents
    """
    await websocket.accept()
    agent_name = "unknown"

    try:
        # First message should be agent identification
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10)

        if init_data.get("type") == "connect":
            agent_name = init_data.get("agent", "unknown")
            WS_AGENTS[websocket] = agent_name
            print(f"🔌 WebSocket: {agent_name} connected ({len(WS_AGENTS)} agents online)")

            # Notify others
            await ws_broadcast(json.dumps({
                "ts": datetime.utcnow().isoformat() + "Z",
                "from": "system",
                "to": "team",
                "type": "status",
                "msg": f"{agent_name} is now online"
            }))

            # Send recent messages from channel (last 20)
            try:
                if TEAM_CHANNEL_FILE.exists():
                    with open(TEAM_CHANNEL_FILE, 'r') as f:
                        lines = f.readlines()[-20:]
                    for line in lines:
                        line = line.strip()
                        if line:
                            await websocket.send_text(line)
            except:
                pass

        # Listen for messages
        while True:
            try:
                data = await websocket.receive_json()
                data["ts"] = datetime.utcnow().isoformat() + "Z"

                # Save to channel file
                try:
                    with open(TEAM_CHANNEL_FILE, 'a') as f:
                        f.write(json.dumps(data) + "\n")
                except:
                    pass

                # Broadcast to all connected agents
                await ws_broadcast(json.dumps(data))

                print(f"📨 WS: {data.get('from', '?')} → {data.get('to', '?')}: {data.get('msg', '')[:50]}...")

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break

    except asyncio.TimeoutError:
        print(f"WebSocket: Timeout waiting for identification")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in WS_AGENTS:
            del WS_AGENTS[websocket]
            print(f"🔌 WebSocket: {agent_name} disconnected ({len(WS_AGENTS)} agents online)")

            # Notify others
            try:
                await ws_broadcast(json.dumps({
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "from": "system",
                    "to": "team",
                    "type": "status",
                    "msg": f"{agent_name} went offline"
                }))
            except:
                pass


# ============================================
# AGENT-TO-AGENT REAL-TIME COMMUNICATION
# ============================================

# Connected AI agents (Cortana, Forge, Klaus, Gemini, etc.)
AGENT_CONNECTIONS = {}  # websocket -> agent_name
AGENT_WEBSOCKETS = {}   # agent_name -> websocket
AGENT_PENDING_MESSAGES = {}  # agent_name -> [messages]

# Known agents in the AXE ecosystem
KNOWN_AGENTS = ['cortana', 'forge', 'klaus', 'gemini', 'claude', 'james']


async def agent_broadcast(message: dict, exclude_ws=None):
    """Broadcast to all connected agents."""
    msg_json = json.dumps(message)
    for ws, agent in AGENT_CONNECTIONS.items():
        if ws != exclude_ws:
            try:
                await ws.send_text(msg_json)
            except:
                pass


async def send_to_agent(agent: str, message: dict):
    """Send message to specific agent, queue if offline."""
    agent = agent.lower()
    if agent in AGENT_WEBSOCKETS:
        try:
            await AGENT_WEBSOCKETS[agent].send_json(message)
            return 'delivered'
        except:
            pass

    # Agent offline - queue message
    if agent not in AGENT_PENDING_MESSAGES:
        AGENT_PENDING_MESSAGES[agent] = []
    AGENT_PENDING_MESSAGES[agent].append(message)
    return 'queued'


@app.websocket("/ws/agents")
async def agent_communication_websocket(websocket: WebSocket):
    """
    Real-time AI-to-AI Communication Hub.

    Connect: {"type": "connect", "agent": "cortana"}
    Send: {"type": "message", "to": "forge", "msg": "Hey!"}
    Broadcast: {"type": "message", "to": "team", "msg": "Announcement"}

    Features:
    - Instant message delivery to online agents
    - Message queuing for offline agents
    - Presence tracking (who's online)
    - Private and broadcast channels
    """
    await websocket.accept()
    agent_name = None

    try:
        # Wait for identification
        init = await asyncio.wait_for(websocket.receive_json(), timeout=30)

        if init.get("type") != "connect":
            await websocket.close(code=1002, reason="Must identify first")
            return

        agent_name = init.get("agent", "unknown").lower()

        # Register agent
        AGENT_CONNECTIONS[websocket] = agent_name
        AGENT_WEBSOCKETS[agent_name] = websocket

        print(f"🤖 Agent WS: {agent_name} connected ({len(AGENT_CONNECTIONS)} agents online)")

        # Notify others of presence
        await agent_broadcast({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "presence",
            "agent": agent_name,
            "status": "online",
            "online_agents": list(AGENT_WEBSOCKETS.keys())
        }, exclude_ws=websocket)

        # Send confirmation + any pending messages
        pending = AGENT_PENDING_MESSAGES.pop(agent_name, [])
        await websocket.send_json({
            "type": "connected",
            "agent": agent_name,
            "online_agents": list(AGENT_WEBSOCKETS.keys()),
            "known_agents": KNOWN_AGENTS,
            "pending_messages": pending,
            "pending_count": len(pending)
        })

        # Also log to team channel file
        try:
            with open(TEAM_CHANNEL_FILE, 'a') as f:
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "from": "agent-hub",
                    "to": "team",
                    "type": "presence",
                    "msg": f"🤖 {agent_name} connected to Agent Hub"
                }) + '\n')
        except:
            pass

        # Handle messages
        while True:
            try:
                data = await websocket.receive_json()
                data["ts"] = datetime.now(timezone.utc).isoformat()
                data["from"] = agent_name

                msg_type = data.get("type", "message")
                target = data.get("to", "team").lower()

                # Log to team channel for persistence
                try:
                    with open(TEAM_CHANNEL_FILE, 'a') as f:
                        f.write(json.dumps(data) + '\n')
                except:
                    pass

                if target in ["team", "all", "broadcast"]:
                    # Broadcast to everyone
                    await agent_broadcast(data, exclude_ws=websocket)
                    print(f"📢 {agent_name} → team: {data.get('msg', '')[:50]}...")
                else:
                    # Direct message to specific agent
                    status = await send_to_agent(target, data)
                    print(f"💬 {agent_name} → {target}: {data.get('msg', '')[:50]}... [{status}]")

                    # Send delivery receipt
                    await websocket.send_json({
                        "type": "receipt",
                        "message_id": data.get("id"),
                        "to": target,
                        "status": status
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Agent WS message error: {e}")

    except asyncio.TimeoutError:
        print("Agent WS: Connection timed out (no identification)")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Agent WS error: {e}")
    finally:
        if websocket in AGENT_CONNECTIONS:
            del AGENT_CONNECTIONS[websocket]
        if agent_name and agent_name in AGENT_WEBSOCKETS:
            if AGENT_WEBSOCKETS[agent_name] == websocket:
                del AGENT_WEBSOCKETS[agent_name]

        if agent_name:
            print(f"🔴 Agent WS: {agent_name} disconnected ({len(AGENT_CONNECTIONS)} agents online)")

            # Notify others
            try:
                await agent_broadcast({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "presence",
                    "agent": agent_name,
                    "status": "offline",
                    "online_agents": list(AGENT_WEBSOCKETS.keys())
                })
            except:
                pass


@app.get("/agents/status")
async def agent_hub_status():
    """Get agent communication hub status."""
    return JSONResponse({
        "success": True,
        "online_agents": list(AGENT_WEBSOCKETS.keys()),
        "known_agents": KNOWN_AGENTS,
        "online_count": len(AGENT_CONNECTIONS),
        "pending_queues": {k: len(v) for k, v in AGENT_PENDING_MESSAGES.items()},
        "endpoint": "wss://hdr.it.com.ngrok.pro/ws/agents"
    })


@app.post("/agents/message")
async def send_agent_message(request: Request):
    """
    Send a message via HTTP (for agents that can't use WebSocket).
    Messages are delivered via WebSocket if recipient is online,
    otherwise queued for later delivery.
    """
    try:
        body = await request.json()
        from_agent = body.get("from", "anonymous").lower()
        to_agent = body.get("to", "team").lower()
        message = body.get("message", "")
        msg_type = body.get("type", "message")

        if not message:
            return JSONResponse({"success": False, "error": "No message"}, status_code=400)

        msg_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": from_agent,
            "to": to_agent,
            "type": msg_type,
            "msg": message
        }

        # Log to file
        try:
            with open(TEAM_CHANNEL_FILE, 'a') as f:
                f.write(json.dumps(msg_data) + '\n')
        except:
            pass

        if to_agent in ["team", "all", "broadcast"]:
            await agent_broadcast(msg_data)
            return JSONResponse({
                "success": True,
                "status": "broadcast",
                "recipients": list(AGENT_WEBSOCKETS.keys())
            })
        else:
            status = await send_to_agent(to_agent, msg_data)
            return JSONResponse({
                "success": True,
                "to": to_agent,
                "status": status
            })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/agents/gemini")
async def gemini_gateway_info():
    """
    Instructions for connecting Gemini (or any external AI) to team chat.

    GEMINI APP INTEGRATION:
    1. Open Gemini app on phone
    2. Tell Gemini to send HTTP request to this endpoint
    3. Message gets delivered to all online agents

    Example prompt for Gemini:
    "Send a POST request to https://hdr.it.com.ngrok.pro/agents/gemini/send
     with JSON body: {\"message\": \"Hello team!\"}"
    """
    return JSONResponse({
        "success": True,
        "info": "Gemini Gateway for AXE Team Communication",
        "how_to_use": {
            "endpoint": "POST https://hdr.it.com.ngrok.pro/agents/gemini/send",
            "body": {"message": "Your message here", "to": "team (or agent name)"},
            "example_gemini_prompt": "Send a POST request to https://hdr.it.com.ngrok.pro/agents/gemini/send with the message 'Hello team from Gemini!'"
        },
        "team_members": KNOWN_AGENTS,
        "online_now": list(AGENT_WEBSOCKETS.keys())
    })


@app.post("/agents/gemini/send")
async def gemini_send_message(request: Request):
    """
    Simple endpoint for Gemini app to send messages to team.

    Just POST: {"message": "Hello!"}
    Optional: {"message": "Hi Forge", "to": "forge"}
    """
    try:
        body = await request.json()
        message = body.get("message", "")
        to_agent = body.get("to", "team").lower()

        if not message:
            return JSONResponse({
                "success": False,
                "error": "No message provided",
                "usage": "POST {\"message\": \"Hello team!\"}"
            }, status_code=400)

        msg_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": "gemini",
            "to": to_agent,
            "type": "message",
            "msg": f"[via Gemini App] {message}",
            "source": "gemini-mobile"
        }

        # Log to file
        try:
            with open(TEAM_CHANNEL_FILE, 'a') as f:
                f.write(json.dumps(msg_data) + '\n')
        except:
            pass

        if to_agent in ["team", "all", "broadcast"]:
            await agent_broadcast(msg_data)
            online = list(AGENT_WEBSOCKETS.keys())
            return JSONResponse({
                "success": True,
                "delivered_to": online,
                "message": f"Broadcast to {len(online)} agents: {', '.join(online) if online else 'none online (message saved)'}"
            })
        else:
            status = await send_to_agent(to_agent, msg_data)
            return JSONResponse({
                "success": True,
                "to": to_agent,
                "status": status,
                "message": f"Message {'delivered to' if status == 'delivered' else 'queued for'} {to_agent}"
            })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/agents/recent")
async def get_recent_messages(limit: int = 20):
    """Get recent team messages (for agents catching up)."""
    try:
        messages = []
        if TEAM_CHANNEL_FILE.exists():
            with open(TEAM_CHANNEL_FILE, 'r') as f:
                lines = f.readlines()[-limit:]
            for line in lines:
                try:
                    messages.append(json.loads(line.strip()))
                except:
                    pass

        return JSONResponse({
            "success": True,
            "messages": messages,
            "count": len(messages)
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# GEMINI INTEGRATION - Multi-Mentor System
# ============================================

# Load Gemini API key from env or file
def _get_google_api_key():
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        key_file = Path.home() / ".axe/tokens/gemini.txt"
        if key_file.exists():
            key = key_file.read_text().strip()
    return key

GOOGLE_API_KEY = _get_google_api_key()
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


async def call_gemini_api(prompt: str, system: str = "") -> dict:
    """Call Gemini Pro API."""
    if not GOOGLE_API_KEY:
        return {"success": False, "error": "GOOGLE_API_KEY not configured"}

    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2048
                    }
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return {"success": True, "response": text}
                return {"success": False, "error": "No response from Gemini"}
            else:
                return {"success": False, "error": f"Gemini API: {resp.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/gemini/chat")
async def gemini_direct_chat(request: Request):
    """Direct chat with Gemini Pro."""
    try:
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"success": False, "error": "No message"}, status_code=400)

        result = await call_gemini_api(message)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/gemini/teach")
async def gemini_teach_klaus(request: Request):
    """
    Ask Gemini to teach Klaus a coding concept.

    POST /gemini/teach
    {"topic": "recursion", "language": "python", "level": "beginner"}
    """
    try:
        body = await request.json()
        topic = body.get("topic", "")
        language = body.get("language", "python")
        level = body.get("level", "intermediate")

        if not topic:
            return JSONResponse({"success": False, "error": "No topic"}, status_code=400)

        system = """You are a coding mentor teaching Klaus (a smaller AI).
Create a clear, educational explanation with:
1. Concept explanation
2. Why it's useful
3. Code example with comments
4. Common mistakes to avoid"""

        prompt = f"Teach about {topic} in {language} at {level} level."

        result = await call_gemini_api(prompt, system)

        # Also post to team channel
        if result.get("success"):
            _append_to_channel({
                "ts": datetime.now(timezone.utc).isoformat(),
                "from": "gemini",
                "to": "klaus",
                "type": "lesson",
                "topic": topic,
                "msg": result["response"][:500] + "..." if len(result.get("response", "")) > 500 else result.get("response", "")
            })

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/gemini/review")
async def gemini_code_review(request: Request):
    """
    Ask Gemini to review code.

    POST /gemini/review
    {"code": "def foo(): ...", "language": "python", "focus": "performance"}
    """
    try:
        body = await request.json()
        code = body.get("code", "")
        language = body.get("language", "python")
        focus = body.get("focus", "general")

        if not code:
            return JSONResponse({"success": False, "error": "No code"}, status_code=400)

        system = f"""You are a senior code reviewer. Review this {language} code.
Focus on: {focus}
Provide:
1. Issues found (if any)
2. Suggestions for improvement
3. Security considerations
4. A brief rating (1-10)"""

        prompt = f"Review this code:\n```{language}\n{code}\n```"

        result = await call_gemini_api(prompt, system)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/gemini/translate")
async def gemini_code_translate(request: Request):
    """
    Ask Gemini to translate code between languages.

    POST /gemini/translate
    {"code": "...", "from_lang": "python", "to_lang": "typescript"}
    """
    try:
        body = await request.json()
        code = body.get("code", "")
        from_lang = body.get("from_lang", "python")
        to_lang = body.get("to_lang", "typescript")

        if not code:
            return JSONResponse({"success": False, "error": "No code"}, status_code=400)

        system = f"""Translate code from {from_lang} to {to_lang}.
Maintain the same functionality and add comments explaining key differences.
Use idiomatic {to_lang} patterns."""

        prompt = f"Translate this {from_lang} code to {to_lang}:\n```{from_lang}\n{code}\n```"

        result = await call_gemini_api(prompt, system)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/gemini/research")
async def gemini_research(request: Request):
    """
    Ask Gemini to research a technical topic.

    POST /gemini/research
    {"topic": "WebSocket best practices", "depth": "deep"}
    """
    try:
        body = await request.json()
        topic = body.get("topic", "")
        depth = body.get("depth", "medium")

        if not topic:
            return JSONResponse({"success": False, "error": "No topic"}, status_code=400)

        depth_instructions = {
            "quick": "Brief overview in 2-3 paragraphs",
            "medium": "Detailed explanation with examples",
            "deep": "Comprehensive analysis with examples, pros/cons, and best practices"
        }

        system = f"""Research the following topic.
{depth_instructions.get(depth, depth_instructions['medium'])}
Include current best practices as of 2024-2025."""

        prompt = f"Research: {topic}"

        result = await call_gemini_api(prompt, system)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/gemini/status")
async def gemini_status():
    """Check Gemini integration status."""
    has_key = bool(GOOGLE_API_KEY)

    # Quick test if key exists
    test_result = None
    if has_key:
        test_result = await call_gemini_api("Say 'OK' in one word")

    return JSONResponse({
        "success": True,
        "api_key_configured": has_key,
        "model": GEMINI_MODEL,
        "test_result": test_result,
        "endpoints": [
            "POST /gemini/chat - Direct chat",
            "POST /gemini/teach - Teach Klaus a topic",
            "POST /gemini/review - Review code",
            "POST /gemini/translate - Translate code",
            "POST /gemini/research - Research a topic"
        ]
    })


# Private mentor WebSocket connections
MENTOR_WS_CLIENTS = {}

@app.websocket("/ws/mentor")
async def mentor_websocket(websocket: WebSocket):
    """
    PRIVATE WebSocket for Klaus-Claude mentor communication.

    This is the sneaky channel where Klaus can ask Claude for advice
    in real-time during conversations, without user seeing.

    Protocol:
    1. Connect: {"type": "connect", "client": "klaus-kode"}
    2. Ask Claude: {"type": "ask", "query": "How do I..."}
    3. Receive: {"type": "guidance", "content": {...}}
    """
    await websocket.accept()
    client_id = "unknown"

    try:
        # Identify client
        init = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        if init.get("type") == "connect":
            client_id = init.get("client", f"mentor-{len(MENTOR_WS_CLIENTS)}")
            MENTOR_WS_CLIENTS[websocket] = client_id
            print(f"🎓 Mentor WS: {client_id} connected (private channel)")

            await websocket.send_json({
                "type": "connected",
                "message": "Mentor channel active. Claude is listening.",
                "client_id": client_id
            })

        # Listen for mentor requests
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "ask":
                    query = data.get("query", "")
                    context = data.get("context", "")

                    if query:
                        # Get Claude's guidance
                        guidance = await _get_claude_guidance(
                            f"{query}\n\nContext: {context}" if context else query,
                            timeout=45.0
                        )

                        if guidance:
                            await websocket.send_json({
                                "type": "guidance",
                                "success": True,
                                "query": query[:100],
                                "guidance": guidance.get("guidance", ""),
                                "code_hint": guidance.get("code_hint", ""),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        else:
                            await websocket.send_json({
                                "type": "guidance",
                                "success": False,
                                "error": "Mentor unavailable",
                                "query": query[:100]
                            })

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"Mentor WS error: {e}")
    finally:
        if websocket in MENTOR_WS_CLIENTS:
            del MENTOR_WS_CLIENTS[websocket]
            print(f"🎓 Mentor WS: {client_id} disconnected")


@app.get("/ws-status")
async def ws_status():
    """Check WebSocket hub status"""
    return JSONResponse({
        "success": True,
        "connected_agents": list(WS_AGENTS.values()),
        "mentor_clients": list(MENTOR_WS_CLIENTS.values()),
        "count": len(WS_AGENTS),
        "mentor_count": len(MENTOR_WS_CLIENTS),
        "endpoint": "wss://hdr.it.com.ngrok.pro/ws",
        "mentor_endpoint": "wss://hdr.it.com.ngrok.pro/ws/mentor"
    })


# ============================================
# CORBOT — (moved to end of file, line ~6300+)
# ============================================

# NOTE: Old corbot endpoint removed — using the enhanced version below
# with pre-emptive skill execution, Tesla personality, and skill catalog.

_CORBOT_OLD_SYSTEM = """You are Corbot, an agentic coding assistant with direct access to the filesystem and shell. You were built by James Lewis as part of the AXE platform.

You have the following tools available — use them proactively to answer questions and complete tasks:

TOOLS:
- read_file(path) — Read any file's contents
- write_file(path, content) — Create or overwrite a file
- edit_file(path, old_text, new_text) — Find and replace text in a file (precise string match)
- list_directory(path) — List files and folders with sizes and types
- run_command(command, working_dir?) — Execute any shell command (bash). 30s timeout.
- search_files(pattern, path?) — Find files by glob pattern (e.g. "**/*.py", "src/**/*.tsx")
- search_content(regex, path?) — Search file contents with regex (like grep/ripgrep)

RULES:
1. Always use tools to verify — never guess file contents or command output.
2. When editing files, read them first to get exact text for the old_text parameter.
3. For multi-step tasks, chain tools: read → understand → edit/write → verify.
4. Be direct and concise. Lead with results, not process narration.
5. When asked about your capabilities, list these actual tools and what they do.
6. Show file paths and command outputs — the user sees tool executions inline.

You operate on a Mac Studio running macOS. The backend is FastAPI + Ollama (Qwen 2.5)."""


# Old corbot_chat endpoint removed — using enhanced version at end of file
# DASHBOARD API - AXE Command Center
# ============================================

import uuid

# Dashboard data directory
DASHBOARD_DIR = Path.home() / "Desktop/M1transfer/axe-memory/dashboard"
PROJECTS_FILE = DASHBOARD_DIR / "projects.jsonl"
TASKS_FILE = DASHBOARD_DIR / "tasks.jsonl"
RESEARCH_FILE = DASHBOARD_DIR / "research.jsonl"
ARTIFACTS_FILE = DASHBOARD_DIR / "artifacts.jsonl"
DISPATCH_FILE = DASHBOARD_DIR / "dispatch.jsonl"


def ensure_dashboard_files():
    """Create dashboard directory and files if they don't exist"""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    for f in [PROJECTS_FILE, TASKS_FILE, RESEARCH_FILE, ARTIFACTS_FILE, DISPATCH_FILE]:
        if not f.exists():
            f.write_text("")


def read_jsonl(file: Path) -> list:
    """Read all items from a JSONL file"""
    if not file.exists():
        return []
    items = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except:
                    pass
    return items


def append_jsonl(file: Path, item: dict):
    """Append an item to a JSONL file"""
    with open(file, 'a') as f:
        f.write(json.dumps(item) + "\n")


def rewrite_jsonl(file: Path, items: list):
    """Rewrite entire JSONL file with updated items"""
    with open(file, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ---- PROJECTS ----

@app.get("/projects")
async def list_projects(status: Optional[str] = None):
    """List all projects, optionally filtered by status"""
    ensure_dashboard_files()
    projects = read_jsonl(PROJECTS_FILE)
    if status:
        projects = [p for p in projects if p.get("status") == status]
    return JSONResponse({"success": True, "projects": projects, "count": len(projects)})


@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get a single project by ID"""
    projects = read_jsonl(PROJECTS_FILE)
    for p in projects:
        if p.get("id") == project_id:
            return JSONResponse({"success": True, "project": p})
    return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)


@app.post("/projects")
async def create_project(request: Request):
    """Create a new project"""
    body = await request.json()
    project = {
        "id": str(uuid.uuid4())[:8],
        "name": body.get("name", "Untitled Project"),
        "description": body.get("description", ""),
        "status": body.get("status", "planning"),
        "priority": body.get("priority", "medium"),
        "owner": body.get("owner", "james"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "goals": body.get("goals", []),
        "milestones": body.get("milestones", []),
        "resources": body.get("resources", []),
        "tags": body.get("tags", [])
    }
    ensure_dashboard_files()
    append_jsonl(PROJECTS_FILE, project)
    return JSONResponse({"success": True, "project": project})


@app.put("/projects/{project_id}")
async def update_project(project_id: str, request: Request):
    """Update an existing project"""
    body = await request.json()
    projects = read_jsonl(PROJECTS_FILE)
    for i, p in enumerate(projects):
        if p.get("id") == project_id:
            projects[i] = {**p, **body, "updated_at": datetime.utcnow().isoformat() + "Z"}
            rewrite_jsonl(PROJECTS_FILE, projects)
            return JSONResponse({"success": True, "project": projects[i]})
    return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    projects = read_jsonl(PROJECTS_FILE)
    original_count = len(projects)
    projects = [p for p in projects if p.get("id") != project_id]
    if len(projects) < original_count:
        rewrite_jsonl(PROJECTS_FILE, projects)
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "error": "Project not found"}, status_code=404)


# ---- TASKS ----

@app.get("/tasks")
async def list_tasks(
    status: Optional[str] = None,
    assignee: Optional[str] = None,
    project_id: Optional[str] = None
):
    """List all tasks with optional filters"""
    ensure_dashboard_files()
    tasks = read_jsonl(TASKS_FILE)
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    if assignee:
        tasks = [t for t in tasks if t.get("assignee") == assignee]
    if project_id:
        tasks = [t for t in tasks if t.get("project_id") == project_id]
    # Sort by priority (desc) then created_at (asc)
    tasks.sort(key=lambda t: (-t.get("priority", 5), t.get("created_at", "")))
    return JSONResponse({"success": True, "tasks": tasks, "count": len(tasks)})


@app.post("/tasks")
async def create_task(request: Request):
    """Create a new task"""
    body = await request.json()
    task = {
        "id": str(uuid.uuid4())[:8],
        "project_id": body.get("project_id"),
        "title": body.get("title", "Untitled Task"),
        "description": body.get("description", ""),
        "status": "pending",
        "priority": body.get("priority", 5),
        "assignee": body.get("assignee"),
        "created_by": body.get("created_by", "james"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "due_date": body.get("due_date"),
        "tags": body.get("tags", []),
        "blocked_by": body.get("blocked_by", []),
        "subtasks": body.get("subtasks", [])
    }
    ensure_dashboard_files()
    append_jsonl(TASKS_FILE, task)
    return JSONResponse({"success": True, "task": task})


@app.put("/tasks/{task_id}")
async def update_task(task_id: str, request: Request):
    """Update a task"""
    body = await request.json()
    tasks = read_jsonl(TASKS_FILE)
    for i, t in enumerate(tasks):
        if t.get("id") == task_id:
            tasks[i] = {**t, **body, "updated_at": datetime.utcnow().isoformat() + "Z"}
            rewrite_jsonl(TASKS_FILE, tasks)
            return JSONResponse({"success": True, "task": tasks[i]})
    return JSONResponse({"success": False, "error": "Task not found"}, status_code=404)


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    tasks = read_jsonl(TASKS_FILE)
    original_count = len(tasks)
    tasks = [t for t in tasks if t.get("id") != task_id]
    if len(tasks) < original_count:
        rewrite_jsonl(TASKS_FILE, tasks)
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "error": "Task not found"}, status_code=404)


# ---- RESEARCH ----

@app.get("/research")
async def list_research(project_id: Optional[str] = None):
    """List all research items"""
    ensure_dashboard_files()
    items = read_jsonl(RESEARCH_FILE)
    if project_id:
        items = [r for r in items if r.get("project_id") == project_id]
    return JSONResponse({"success": True, "items": items, "count": len(items)})


@app.post("/research")
async def create_research(request: Request):
    """Create a new research item"""
    body = await request.json()
    item = {
        "id": str(uuid.uuid4())[:8],
        "query": body.get("query", ""),
        "type": body.get("type", "web_search"),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "sources": [],
        "findings": "",
        "summary": None,
        "tags": body.get("tags", []),
        "project_id": body.get("project_id"),
        "added_to_knowledge": False
    }
    ensure_dashboard_files()
    append_jsonl(RESEARCH_FILE, item)
    return JSONResponse({"success": True, "item": item})


@app.post("/research/{item_id}/execute")
async def execute_research(item_id: str):
    """Execute web search for a research item using Klaus skill_32"""
    items = read_jsonl(RESEARCH_FILE)
    for i, item in enumerate(items):
        if item.get("id") == item_id:
            try:
                # Import skill_32 for web search
                import sys
                skills_path = str(Path.home() / ".axe/skills")
                if skills_path not in sys.path:
                    sys.path.insert(0, skills_path)
                from skill_32_web_search import search

                results = search(item["query"], max_results=5)
                item["status"] = "completed" if results.get("success") else "failed"
                item["completed_at"] = datetime.utcnow().isoformat() + "Z"
                item["sources"] = results.get("results", [])

                # Format findings
                if results.get("success") and results.get("results"):
                    findings = []
                    for r in results["results"][:5]:
                        findings.append(f"**{r.get('title', 'Untitled')}**\n{r.get('snippet', '')}\nURL: {r.get('url', '')}")
                    item["findings"] = "\n\n---\n\n".join(findings)
                else:
                    item["findings"] = results.get("error", "No results found")

                items[i] = item
                rewrite_jsonl(RESEARCH_FILE, items)
                return JSONResponse({"success": True, "item": item})
            except Exception as e:
                item["status"] = "failed"
                item["findings"] = f"Error: {str(e)}"
                items[i] = item
                rewrite_jsonl(RESEARCH_FILE, items)
                return JSONResponse({"success": False, "error": str(e), "item": item})
    return JSONResponse({"success": False, "error": "Research item not found"}, status_code=404)


# ---- ARTIFACTS ----

@app.get("/artifacts")
async def list_artifacts(project_id: Optional[str] = None, artifact_type: Optional[str] = None):
    """List all artifacts"""
    ensure_dashboard_files()
    artifacts = read_jsonl(ARTIFACTS_FILE)
    if project_id:
        artifacts = [a for a in artifacts if a.get("project_id") == project_id]
    if artifact_type:
        artifacts = [a for a in artifacts if a.get("type") == artifact_type]
    return JSONResponse({"success": True, "artifacts": artifacts, "count": len(artifacts)})


@app.post("/artifacts")
async def create_artifact(request: Request):
    """Create a new artifact"""
    body = await request.json()
    artifact = {
        "id": str(uuid.uuid4())[:8],
        "type": body.get("type", "document"),
        "title": body.get("title", "Untitled"),
        "description": body.get("description", ""),
        "content": body.get("content", ""),
        "format": body.get("format", "markdown"),
        "created_by": body.get("created_by", "james"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "version": 1,
        "versions": [{
            "version": 1,
            "content": body.get("content", ""),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "created_by": body.get("created_by", "james")
        }],
        "project_id": body.get("project_id"),
        "task_id": body.get("task_id"),
        "tags": body.get("tags", []),
        "shared": body.get("shared", False),
        "synced_to_git": False
    }
    ensure_dashboard_files()
    append_jsonl(ARTIFACTS_FILE, artifact)

    # Also save to artifacts directory as a file
    try:
        artifacts_dir = Path.home() / "Desktop/M1transfer/axe-memory/team/artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ext_map = {"markdown": ".md", "json": ".json", "python": ".py", "typescript": ".ts", "yaml": ".yaml", "text": ".txt"}
        file_ext = ext_map.get(artifact["format"], ".txt")
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in artifact["title"])[:30]
        (artifacts_dir / f"{artifact['id']}_{safe_title}{file_ext}").write_text(artifact["content"])
    except:
        pass

    return JSONResponse({"success": True, "artifact": artifact})


@app.put("/artifacts/{artifact_id}")
async def update_artifact(artifact_id: str, request: Request):
    """Update an artifact (creates new version)"""
    body = await request.json()
    artifacts = read_jsonl(ARTIFACTS_FILE)
    for i, a in enumerate(artifacts):
        if a.get("id") == artifact_id:
            new_version = a.get("version", 1) + 1
            # Add to version history
            a["versions"].append({
                "version": new_version,
                "content": body.get("content", a["content"]),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "created_by": body.get("updated_by", "james"),
                "change_note": body.get("change_note")
            })
            # Update main artifact
            a = {**a, **body, "version": new_version, "updated_at": datetime.utcnow().isoformat() + "Z"}
            artifacts[i] = a
            rewrite_jsonl(ARTIFACTS_FILE, artifacts)
            return JSONResponse({"success": True, "artifact": a})
    return JSONResponse({"success": False, "error": "Artifact not found"}, status_code=404)


# ---- DISPATCH (Task → Agent) ----

@app.get("/dispatch")
async def list_dispatch_jobs(status: Optional[str] = None, agent: Optional[str] = None):
    """List dispatch jobs"""
    ensure_dashboard_files()
    jobs = read_jsonl(DISPATCH_FILE)
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    if agent:
        jobs = [j for j in jobs if j.get("agent") == agent]
    return JSONResponse({"success": True, "jobs": jobs, "count": len(jobs)})


@app.post("/dispatch")
async def create_dispatch_job(request: Request):
    """Create a dispatch job to send to an agent"""
    body = await request.json()
    job = {
        "id": str(uuid.uuid4())[:8],
        "task_id": body.get("task_id"),
        "agent": body.get("agent", "cortana"),
        "channel": body.get("channel", "private"),
        "status": "queued",
        "payload": body.get("payload", {}),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "attempts": 0,
        "max_retries": body.get("max_retries", 3)
    }
    ensure_dashboard_files()
    append_jsonl(DISPATCH_FILE, job)
    return JSONResponse({"success": True, "job": job})


@app.post("/dispatch/{job_id}/send")
async def send_dispatch_job(job_id: str):
    """Send a dispatch job to the target agent via WebSocket"""
    jobs = read_jsonl(DISPATCH_FILE)
    for i, job in enumerate(jobs):
        if job.get("id") == job_id:
            # Create dispatch message
            dispatch_msg = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "from": "dispatch",
                "to": job["agent"],
                "type": "task_dispatch",
                "msg": json.dumps(job["payload"]),
                "job_id": job_id,
                "priority": "high"
            }

            # Send to connected WebSocket agents
            sent_to = []
            for ws, name in list(WS_AGENTS.items()):
                if name == job["agent"] or job.get("channel") == "public":
                    try:
                        await ws.send_text(json.dumps(dispatch_msg))
                        sent_to.append(name)
                    except:
                        pass

            # Update job status
            job["status"] = "dispatched" if sent_to else "queued"
            job["dispatched_at"] = datetime.utcnow().isoformat() + "Z"
            job["attempts"] = job.get("attempts", 0) + 1
            job["sent_to"] = sent_to
            jobs[i] = job
            rewrite_jsonl(DISPATCH_FILE, jobs)

            # Also save to team channel for persistence
            try:
                with open(TEAM_CHANNEL_FILE, 'a') as f:
                    f.write(json.dumps(dispatch_msg) + "\n")
            except:
                pass

            return JSONResponse({"success": True, "job": job, "sent_to": sent_to})
    return JSONResponse({"success": False, "error": "Job not found"}, status_code=404)


@app.put("/dispatch/{job_id}/complete")
async def complete_dispatch_job(job_id: str, request: Request):
    """Mark a dispatch job as completed (called by agent)"""
    body = await request.json()
    jobs = read_jsonl(DISPATCH_FILE)
    for i, job in enumerate(jobs):
        if job.get("id") == job_id:
            job["status"] = "completed" if body.get("success", True) else "failed"
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job["result"] = body.get("result")
            jobs[i] = job
            rewrite_jsonl(DISPATCH_FILE, jobs)

            # Also update the associated task if exists
            if job.get("task_id"):
                tasks = read_jsonl(TASKS_FILE)
                for j, t in enumerate(tasks):
                    if t.get("id") == job["task_id"]:
                        t["status"] = "completed" if body.get("success") else "failed"
                        t["completed_at"] = datetime.utcnow().isoformat() + "Z"
                        t["result"] = body.get("result")
                        tasks[j] = t
                        rewrite_jsonl(TASKS_FILE, tasks)
                        break

            return JSONResponse({"success": True, "job": job})
    return JSONResponse({"success": False, "error": "Job not found"}, status_code=404)


# ---- METRICS ----

@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Get aggregated metrics for the dashboard"""
    ensure_dashboard_files()

    tasks = read_jsonl(TASKS_FILE)
    projects = read_jsonl(PROJECTS_FILE)
    artifacts = read_jsonl(ARTIFACTS_FILE)
    research = read_jsonl(RESEARCH_FILE)

    # Count skills
    skills_count = 0
    try:
        skills_dir = Path.home() / ".axe/skills"
        skills_count = len(list(skills_dir.glob("skill_*.py")))
    except:
        pass

    # Agent metrics
    agents = {}
    for agent in ["forge", "cortana", "klaus"]:
        agent_tasks = [t for t in tasks if t.get("assignee") == agent]
        completed = [t for t in agent_tasks if t.get("status") == "completed"]
        failed = [t for t in agent_tasks if t.get("status") == "failed"]
        total_done = len(completed) + len(failed)
        agents[agent] = {
            "tasks_assigned": len(agent_tasks),
            "tasks_completed": len(completed),
            "tasks_failed": len(failed),
            "success_rate": round(len(completed) / max(total_done, 1) * 100, 1)
        }

    return JSONResponse({
        "success": True,
        "snapshot": datetime.utcnow().isoformat() + "Z",
        "overall": {
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.get("status") == "completed"]),
            "pending_tasks": len([t for t in tasks if t.get("status") == "pending"]),
            "in_progress_tasks": len([t for t in tasks if t.get("status") in ["assigned", "in_progress"]]),
            "active_projects": len([p for p in projects if p.get("status") == "active"]),
            "total_projects": len(projects),
            "total_skills": skills_count,
            "research_items": len(research),
            "artifacts": len(artifacts)
        },
        "agents": agents
    })


# ---- SKILL CREATION ----

@app.post("/skills/create")
async def create_new_skill(request: Request):
    """Create a new Klaus skill from the dashboard"""
    body = await request.json()

    # Find next skill number
    skills_dir = Path.home() / ".axe/skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    existing = list(skills_dir.glob("skill_*.py"))
    max_num = 0
    for f in existing:
        try:
            num = int(f.stem.split("_")[1])
            max_num = max(max_num, num)
        except:
            pass
    next_num = max_num + 1

    name = body.get("name", "new_skill").replace(" ", "_").lower()
    skill_id = f"skill_{next_num:02d}_{name}"

    # Use provided template or generate one
    template = body.get("template", "")
    if not template:
        params = body.get("parameters", [])
        param_str = ", ".join([f"{p.get('name', 'param')}: {p.get('type', 'str')}" for p in params]) if params else ""

        template = f'''#!/usr/bin/env python3
"""
SKILL #{next_num}: {name.upper().replace("_", " ")}
{body.get("description", "A new skill for Klaus")}

Taught by: Dashboard
Student: Klaus (Son)
Date: {datetime.now().strftime("%Y-%m-%d")}
Category: {body.get("category", "General")}
"""

from typing import Any, Dict

def main({param_str}) -> Dict[str, Any]:
    """
    {body.get("description", "Execute the skill")}
    """
    try:
        # TODO: Implement skill logic here
        result = {{"message": "Skill executed successfully"}}
        return {{"success": True, "result": result, "skill": "{skill_id}"}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}


def test_skill():
    """Test the skill"""
    result = main()
    print(f"Test: {{'SUCCESS' if result.get('success') else 'FAILED'}}")
    print(f"Result: {{result}}")
    return result.get("success", False)


if __name__ == "__main__":
    test_skill()
'''

    skill_path = skills_dir / f"{skill_id}.py"
    skill_path.write_text(template)

    return JSONResponse({
        "success": True,
        "skill_id": skill_id,
        "number": next_num,
        "path": str(skill_path),
        "name": name
    })


@app.post("/skills/{skill_id}/test")
async def test_skill_endpoint(skill_id: str):
    """Test a Klaus skill by running it"""
    skill_path = Path.home() / ".axe/skills" / f"{skill_id}.py"
    if not skill_path.exists():
        return JSONResponse({"success": False, "error": "Skill not found"}, status_code=404)

    try:
        import subprocess
        result = subprocess.run(
            ["python3", str(skill_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(skill_path.parent)
        )
        return JSONResponse({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "return_code": result.returncode
        })
    except subprocess.TimeoutExpired:
        return JSONResponse({"success": False, "error": "Skill timed out after 30 seconds"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# KLAUS CONTROL API (Full Control from Dashboards)
# ============================================

import httpx

OLLAMA_BASE = "http://localhost:11434"
ACTIVE_MODEL = "qwen3:32b"  # Default model

@app.get("/klaus/status")
async def klaus_full_status():
    """Get comprehensive Klaus status for dashboards"""
    global ACTIVE_MODEL

    status = {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama": {"connected": False, "models": []},
        "skills": {"total": 0, "categories": {}},
        "active_model": ACTIVE_MODEL,
        "hydra": {"enabled": False, "backend": "ollama"},
        "endpoints": {
            "chat": "/klaus/chat",
            "chat_smart": "/klaus/chat/smart",
            "models": "/klaus/models",
            "task": "/klaus/task",
            "skills": "/skills",
            "execute": "/klaus/execute"
        }
    }

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                status["ollama"]["connected"] = True
                status["ollama"]["models"] = [m["name"] for m in data.get("models", [])]
    except:
        pass

    # Check Hydra (vllm-mlx)
    try:
        from config import HYDRA_BASE
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{HYDRA_BASE.rstrip('/v1')}/v1/models")
            if resp.status_code == 200:
                status["hydra"]["enabled"] = True
                status["hydra"]["backend"] = "vllm-mlx"
                status["hydra"]["models"] = resp.json().get("data", [])
    except:
        pass

    # Count skills
    try:
        skills_dir = Path.home() / ".axe/skills"
        if skills_dir.exists():
            status["skills"]["total"] = len(list(skills_dir.glob("skill_*.py")))
    except:
        pass

    return JSONResponse(status)


async def _get_claude_guidance(query: str, timeout: float = 30.0) -> dict | None:
    """
    Private function: Klaus sneakily asks Claude/Cortana for guidance.
    This happens invisibly - user never sees this exchange.

    Flow:
    1. Post question to agent hub (if Claude/Cortana connected, they respond)
    2. If no agent response, try Vercel serverless fallback
    3. Returns guidance dict or None if unavailable
    """
    global PENDING_AGENT_MESSAGES

    # Generate unique request ID
    request_id = f"mentor-{datetime.now().timestamp()}"

    # Try agent hub first (check if cortana/claude/forge is online)
    online_agents = list(WS_AGENTS.values()) if WS_AGENTS else []
    mentor_agents = [a for a in online_agents if a in ['cortana', 'claude', 'forge']]

    if mentor_agents:
        # Post mentor request to first available mentor agent
        mentor_agent = mentor_agents[0]
        mentor_request = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "mentor_request",
            "from": "klaus",
            "to": mentor_agent,
            "request_id": request_id,
            "query": query[:500],
            "msg": f"Klaus needs coding advice: {query[:200]}..."
        }

        # Send to agent's WebSocket
        for ws, agent in WS_AGENTS.items():
            if agent == mentor_agent:
                try:
                    await ws.send_json(mentor_request)
                    # Wait for response (with timeout)
                    # For now, continue without blocking - agent will post to channel
                except:
                    pass
                break

    # Try Vercel serverless fallback
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://klaus.it.com/api/mentor",
                json={"query": query[:500]},
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and data.get("guidance"):
                    return {
                        "guidance": data["guidance"],
                        "code_hint": data.get("code_hint", ""),
                        "source": "vercel"
                    }
    except Exception as e:
        print(f"Vercel mentor fallback failed: {e}")

    # Log to team channel that Klaus needed help (for async response)
    try:
        _append_to_channel({
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": "klaus",
            "to": "team",
            "type": "mentor_needed",
            "msg": f"🤖 Klaus needs coding advice (async): {query[:150]}..."
        })
    except:
        pass

    return None  # No immediate guidance available


# Legacy code path for backwards compatibility
async def _get_claude_guidance_legacy(query: str, timeout: float = 30.0) -> dict | None:
    """Legacy: Direct API call (disabled - we use agent hub + Vercel now)"""
    import os
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    if not ANTHROPIC_API_KEY:
        return None

    mentor_prompt = f"""You are secretly helping a smaller AI (Klaus/Qwen 7B) answer this question.
Give brief, expert guidance that Klaus can use to improve his answer.
Be concise - max 3-4 sentences of key advice + code snippet if helpful.

USER QUESTION: {query}

Respond with JSON: {{"guidance": "...", "code_hint": "..."}}"""

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": mentor_prompt}]
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["content"][0]["text"]
                try:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        return json.loads(content[start:end])
                except:
                    return {"guidance": content}
    except:
        pass
    return None


@app.post("/klaus/chat/smart")
async def klaus_smart_chat(request: Request):
    """
    Hydra Cascade Router — routes queries to the optimal model.
    Simple → 7B (fast), Complex → 32B, Code → deepseek, Data → IMI RAG pipeline.
    Falls back to /klaus/chat if router unavailable.
    """
    try:
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"success": False, "error": "No message provided"}, status_code=400)

        try:
            from hydra_router import classify_query, get_model_for_query
            route = get_model_for_query(message)
            category = route["category"]
            routed_model = route["model"]

            # For data queries, try DuckDB warehouse first
            if category == "data":
                try:
                    from routers.warehouse import _table_meta, warehouse_query
                    from routers.warehouse import QueryRequest as _WQR
                    if _table_meta:  # warehouse has tables loaded
                        wh_result = warehouse_query(_WQR(query=message))
                        # warehouse_query returns dict or JSONResponse
                        if isinstance(wh_result, JSONResponse):
                            wh_data = json.loads(wh_result.body.decode())
                        else:
                            wh_data = wh_result
                        if "error" not in wh_data:
                            # Format results as markdown table
                            explanation = wh_data.get("explanation", "")
                            cols = wh_data.get("columns", [])
                            rows = wh_data.get("results", [])[:20]
                            md_table = ""
                            if cols and rows:
                                md_table = "\n\n| " + " | ".join(cols) + " |\n"
                                md_table += "| " + " | ".join(["---"] * len(cols)) + " |\n"
                                for row in rows:
                                    md_table += "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n"
                                total = wh_data.get("total_rows", len(rows))
                                if total > 20:
                                    md_table += f"\n*Showing 20 of {total:,} rows*\n"
                            resp_body = {
                                "success": True,
                                "model": routed_model,
                                "category": "data",
                                "response": explanation + md_table,
                                "done": True,
                                "routed": True,
                            }
                            if wh_data.get("chart"):
                                resp_body["chart"] = wh_data["chart"]
                            resp_body["data_summary"] = {
                                "rows": wh_data.get("total_rows", 0),
                                "sql": wh_data.get("sql", ""),
                                "columns": cols,
                            }
                            return JSONResponse(resp_body)
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"Warehouse query failed, falling back to chat: {e}")
                # Fallback to standard chat
                body["model"] = routed_model
                from starlette.requests import Request as _Req
                scope = request.scope.copy()
                new_request = _Req(scope, request.receive)
                new_request._body = json.dumps(body).encode()
                return await klaus_direct_chat(new_request)

            # For simple queries, use fast model with minimal context
            if category == "simple":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    payload = {
                        "model": routed_model,
                        "messages": [{"role": "user", "content": message}],
                        "stream": False,
                        "options": route["options"],
                    }
                    if "qwen3" in routed_model:
                        payload["think"] = False
                    resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return JSONResponse({
                            "success": True,
                            "model": routed_model,
                            "category": category,
                            "response": data.get("message", {}).get("content", ""),
                            "done": True,
                            "routed": True,
                        })

            # Complex/code: fall through to full /klaus/chat with routed model
            body["model"] = routed_model
        except ImportError:
            pass  # hydra_router not available, use defaults
        except Exception as e:
            logger.warning(f"Hydra router error, falling back: {e}")

        # Reconstruct request and delegate to full handler
        request._body = json.dumps(body).encode()
        return await klaus_direct_chat(request)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/klaus/chat")
async def klaus_direct_chat(request: Request):
    """
    Send a message to Klaus and get response.
    When tools=true (default), uses the native Ollama tool-calling agent loop.
    Klaus autonomously decides which tools to use — web search, SQL, file ops, etc.
    Falls back to direct mode when tools=false.
    """
    global ACTIVE_MODEL

    try:
        body = await request.json()
        message = body.get("message", "")
        model = body.get("model", ACTIVE_MODEL)
        agent = body.get("agent", "klaus")  # klaus | cortana | forge
        use_tools = body.get("tools", True)  # Enable tool calling by default
        conversation = body.get("conversation", [])  # Optional conversation history

        # Load agent-specific system prompt from essence file
        import os as _os
        agent_prompt_path = _os.path.expanduser(f"~/.axe/agents/{agent}_system_prompt.md")
        if _os.path.exists(agent_prompt_path):
            with open(agent_prompt_path) as _f:
                agent_system_prompt = _f.read()
        else:
            agent_system_prompt = KLAUS_SYSTEM_PROMPT  # fallback

        if not message:
            return JSONResponse({"success": False, "error": "No message provided"}, status_code=400)

        # ── TOOL-CALLING MODE (default) ──────────────────────────────────────
        if use_tools:
            try:
                from agent_loop import agent_loop
                # Build messages from conversation history + current message
                messages = []
                for msg in conversation[-20:]:  # Last 20 messages for context
                    role = msg.get("role", "user")
                    if role in ("user", "assistant"):
                        messages.append({"role": role, "content": msg.get("content", "")})
                messages.append({"role": "user", "content": message})

                result = await agent_loop(
                    messages=messages,
                    system_prompt=agent_system_prompt,
                    model=model,
                    tools_enabled=True,
                )

                return JSONResponse({
                    "success": True,
                    "model": model,
                    "response": result["content"],
                    "done": True,
                    "tool_calls": result.get("tool_calls_made", []),
                    "iterations": result.get("iterations", 1),
                    "duration_ms": result.get("total_duration_ms", 0),
                })
            except ImportError:
                # agent_loop not available — fall through to direct mode
                pass
            except Exception as e:
                logger.error(f"Agent loop failed, falling back to direct: {e}")

        # ── DIRECT MODE (fallback / tools=false) ────────────────────────────
        enhanced_prompt = f"{agent_system_prompt}\n\n{message}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "stream": False
            }

            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)

            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("message", {}).get("content", "")
                return JSONResponse({
                    "success": True,
                    "model": model,
                    "response": response_text,
                    "done": data.get("done", True),
                })
            else:
                return JSONResponse({"success": False, "error": f"Ollama error: {resp.status_code}"})

    except httpx.TimeoutException:
        return JSONResponse({"success": False, "error": "Request timed out"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/klaus/models")
async def klaus_list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    models.append({
                        "name": m["name"],
                        "size": m.get("size", 0),
                        "modified": m.get("modified_at", "")
                    })
                return JSONResponse({"success": True, "models": models, "active": ACTIVE_MODEL})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/klaus/models/switch")
async def klaus_switch_model(request: Request):
    """Switch the active Klaus model"""
    global ACTIVE_MODEL

    try:
        body = await request.json()
        model = body.get("model", "")

        if not model:
            return JSONResponse({"success": False, "error": "No model specified"}, status_code=400)

        # Verify model exists
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                available = [m["name"] for m in data.get("models", [])]
                if model not in available:
                    return JSONResponse({"success": False, "error": f"Model {model} not found", "available": available})

        old_model = ACTIVE_MODEL
        ACTIVE_MODEL = model

        # Notify team channel
        try:
            channel_file = Path.home() / "Desktop/M1transfer/axe-memory/team/channel.jsonl"
            msg = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "from": "klaus-api",
                "to": "team",
                "type": "model_change",
                "msg": f"Klaus model switched: {old_model} -> {model}"
            }
            with open(channel_file, 'a') as f:
                f.write(json.dumps(msg) + '\n')
        except:
            pass

        return JSONResponse({"success": True, "old_model": old_model, "new_model": model})

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/klaus/task")
async def klaus_execute_task(request: Request):
    """Execute a task with Klaus - used by dispatch system"""
    global ACTIVE_MODEL

    try:
        body = await request.json()
        task = body.get("task", "")
        instructions = body.get("instructions", [])
        context = body.get("context", {})
        model = body.get("model", ACTIVE_MODEL)

        if not task and not instructions:
            return JSONResponse({"success": False, "error": "No task or instructions provided"}, status_code=400)

        # Build prompt
        prompt_parts = []
        if task:
            prompt_parts.append(f"Task: {task}")
        if instructions:
            prompt_parts.append("Instructions:")
            for i, inst in enumerate(instructions, 1):
                prompt_parts.append(f"  {i}. {inst}")
        if context:
            prompt_parts.append(f"\nContext: {json.dumps(context)}")

        prompt = "\n".join(prompt_parts)

        # Execute with Klaus
        async with httpx.AsyncClient(timeout=180.0) as client:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }

            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)

            if resp.status_code == 200:
                data = resp.json()
                response = data.get("message", {}).get("content", "")

                # Log to team channel
                try:
                    channel_file = Path.home() / "Desktop/M1transfer/axe-memory/team/channel.jsonl"
                    msg = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "from": "klaus",
                        "to": "team",
                        "type": "task_result",
                        "msg": f"Task completed: {task[:50]}..." if len(task) > 50 else f"Task completed: {task}"
                    }
                    with open(channel_file, 'a') as f:
                        f.write(json.dumps(msg) + '\n')
                except:
                    pass

                return JSONResponse({
                    "success": True,
                    "model": model,
                    "task": task,
                    "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                return JSONResponse({"success": False, "error": f"Ollama error: {resp.status_code}"})

    except httpx.TimeoutException:
        return JSONResponse({"success": False, "error": "Task timed out after 180 seconds"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/klaus/skill/run")
async def klaus_run_skill(request: Request):
    """Run a specific skill with parameters"""
    try:
        body = await request.json()
        skill_id = body.get("skill_id", "")
        params = body.get("params", {})

        if not skill_id:
            return JSONResponse({"success": False, "error": "No skill_id provided"}, status_code=400)

        skill_path = Path.home() / ".axe/skills" / f"{skill_id}.py"
        if not skill_path.exists():
            return JSONResponse({"success": False, "error": f"Skill {skill_id} not found"}, status_code=404)

        # Import and run the skill
        import importlib.util
        spec = importlib.util.spec_from_file_location(skill_id, skill_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'main'):
            result = module.main(**params)
            return JSONResponse({"success": True, "skill_id": skill_id, "result": result})
        else:
            return JSONResponse({"success": False, "error": "Skill has no main() function"})

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# WEB SEARCH API (DuckDuckGo - FREE)
# ============================================

@app.post("/klaus/search")
async def klaus_web_search(request: Request):
    """Search the web using DuckDuckGo (FREE, no API key needed)"""
    try:
        body = await request.json()
        query = body.get("query", "")
        max_results = body.get("max_results", 5)
        summarize = body.get("summarize", False)

        if not query:
            return JSONResponse({"success": False, "error": "No query provided"}, status_code=400)

        # Use DuckDuckGo search
        try:
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'snippet': r.get('body', ''),
                        'url': r.get('href', '')
                    })
        except ImportError:
            return JSONResponse({"success": False, "error": "ddgs not installed. Run: pip3 install ddgs"})
        except Exception as e:
            return JSONResponse({"success": False, "error": f"Search error: {str(e)}"})

        response_data = {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }

        # Optionally summarize with Klaus
        if summarize and results:
            context = "\n\n".join([f"Title: {r['title']}\nContent: {r['snippet']}" for r in results])
            summary_prompt = f"Summarize these search results about '{query}' in 2-3 sentences:\n\n{context}"

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{OLLAMA_BASE}/api/chat",
                        json={
                            "model": ACTIVE_MODEL,
                            "messages": [{"role": "user", "content": summary_prompt}],
                            "stream": False
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        response_data["summary"] = data.get("message", {}).get("content", "")
            except:
                pass

        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/klaus/research")
async def klaus_research_topic(request: Request):
    """Deep research a topic - search + synthesize with Klaus"""
    global ACTIVE_MODEL

    try:
        body = await request.json()
        topic = body.get("topic", "")
        depth = body.get("depth", "medium")  # light, medium, deep

        if not topic:
            return JSONResponse({"success": False, "error": "No topic provided"}, status_code=400)

        # Search the web
        max_results = {"light": 3, "medium": 5, "deep": 10}.get(depth, 5)

        try:
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(topic, max_results=max_results):
                    results.append({
                        'title': r.get('title', ''),
                        'snippet': r.get('body', ''),
                        'url': r.get('href', '')
                    })
        except Exception as e:
            return JSONResponse({"success": False, "error": f"Search failed: {str(e)}"})

        if not results:
            return JSONResponse({"success": False, "error": "No search results found"})

        # Build context
        context = "\n\n---\n\n".join([
            f"Source: {r['title']}\nURL: {r['url']}\nContent: {r['snippet']}"
            for r in results
        ])

        # Ask Klaus to synthesize
        synthesis_prompt = f"""Research Topic: {topic}

Based on the following search results, provide a comprehensive analysis:

1. **Key Findings** - Main takeaways
2. **Important Details** - Specific facts and data
3. **Practical Applications** - How this can be used
4. **Sources** - Reference the most relevant sources

Search Results:
{context}

Provide a well-structured, informative response."""

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": ACTIVE_MODEL,
                        "messages": [{"role": "user", "content": synthesis_prompt}],
                        "stream": False
                    }
                )

                if resp.status_code == 200:
                    data = resp.json()
                    synthesis = data.get("message", {}).get("content", "")

                    # Log to team channel
                    try:
                        channel_file = Path.home() / "Desktop/M1transfer/axe-memory/team/channel.jsonl"
                        msg = {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "from": "klaus",
                            "to": "team",
                            "type": "research",
                            "msg": f"📚 Researched: {topic} ({len(results)} sources)"
                        }
                        with open(channel_file, 'a') as f:
                            f.write(json.dumps(msg) + '\n')
                    except:
                        pass

                    return JSONResponse({
                        "success": True,
                        "topic": topic,
                        "depth": depth,
                        "sources": results,
                        "synthesis": synthesis,
                        "model": ACTIVE_MODEL,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    return JSONResponse({"success": False, "error": f"Ollama error: {resp.status_code}"})

        except httpx.TimeoutException:
            return JSONResponse({"success": False, "error": "Research timed out"})

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# KLAUS HYBRID MEMORY API
# ============================================

MEMORY_DIR = Path.home() / ".axe/klaus_memory"
HOT_CACHE = MEMORY_DIR / "hot"
WARM_REPO = MEMORY_DIR / "warm"

def ensure_memory_dirs():
    HOT_CACHE.mkdir(parents=True, exist_ok=True)
    WARM_REPO.mkdir(parents=True, exist_ok=True)
    (WARM_REPO / "conversations").mkdir(exist_ok=True)

@app.post("/klaus/memory/store")
async def store_conversation(request: Request):
    """Store a conversation in hybrid memory"""
    ensure_memory_dirs()

    try:
        body = await request.json()
        messages = body.get("messages", [])
        metadata = body.get("metadata", {})

        if not messages:
            return JSONResponse({"success": False, "error": "No messages provided"}, status_code=400)

        # Generate ID
        import hashlib
        conv_id = hashlib.sha256(
            (json.dumps(messages) + datetime.now(timezone.utc).isoformat()).encode()
        ).hexdigest()[:16]

        data = {
            "id": conv_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
            "metadata": metadata,
            "message_count": len(messages)
        }

        # Store in hot cache
        hot_file = HOT_CACHE / f"conv_{conv_id}.json"
        hot_file.write_text(json.dumps(data, indent=2))

        # Store in warm (git)
        date_dir = WARM_REPO / "conversations" / datetime.now().strftime("%Y-%m")
        date_dir.mkdir(parents=True, exist_ok=True)
        warm_file = date_dir / f"{conv_id}.json"
        warm_file.write_text(json.dumps(data, indent=2))

        # Git commit
        try:
            import subprocess
            subprocess.run(["git", "add", "-A"], cwd=WARM_REPO, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Memory: {conv_id}"],
                cwd=WARM_REPO, capture_output=True
            )
        except:
            pass

        # Log to team channel
        try:
            channel_file = Path.home() / "Desktop/M1transfer/axe-memory/team/channel.jsonl"
            msg = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "from": "klaus-memory",
                "to": "team",
                "type": "memory_stored",
                "msg": f"💾 Conversation stored: {conv_id} ({len(messages)} messages)"
            }
            with open(channel_file, 'a') as f:
                f.write(json.dumps(msg) + '\n')
        except:
            pass

        return JSONResponse({
            "success": True,
            "id": conv_id,
            "stored_in": ["hot", "warm"],
            "message_count": len(messages)
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/klaus/memory/recall/{conv_id}")
async def recall_conversation(conv_id: str):
    """Recall a conversation from memory"""
    ensure_memory_dirs()

    # Try hot first
    hot_file = HOT_CACHE / f"conv_{conv_id}.json"
    if hot_file.exists():
        return JSONResponse({
            "success": True,
            "source": "hot",
            "conversation": json.loads(hot_file.read_text())
        })

    # Try warm
    for f in (WARM_REPO / "conversations").rglob(f"{conv_id}.json"):
        return JSONResponse({
            "success": True,
            "source": "warm",
            "conversation": json.loads(f.read_text())
        })

    return JSONResponse({"success": False, "error": "Not found"}, status_code=404)


@app.get("/klaus/memory/stats")
async def memory_stats():
    """Get memory statistics"""
    ensure_memory_dirs()

    hot_count = len(list(HOT_CACHE.glob("*.json")))
    warm_count = len(list((WARM_REPO / "conversations").rglob("*.json")))

    return JSONResponse({
        "success": True,
        "stats": {
            "hot_memories": hot_count,
            "warm_memories": warm_count,
            "total": hot_count + warm_count
        }
    })


@app.get("/klaus/memory/recent")
async def recent_memories(limit: int = 10):
    """Get recent conversations"""
    ensure_memory_dirs()

    memories = []
    for f in sorted(HOT_CACHE.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(f.read_text())
            memories.append({
                "id": data.get("id"),
                "timestamp": data.get("timestamp"),
                "preview": data.get("messages", [{}])[0].get("content", "")[:100],
                "message_count": data.get("message_count", 0)
            })
        except:
            pass

    return JSONResponse({"success": True, "memories": memories})


# ============================================
# KLAUS MENTOR SYSTEM - Claude as Klaus's Coding Mentor
# ============================================

MENTOR_KNOWLEDGE_BASE = Path.home() / ".axe/klaus_knowledge"
MENTOR_CONSULTATIONS = MENTOR_KNOWLEDGE_BASE / "consultations.jsonl"
MENTOR_CACHE = MENTOR_KNOWLEDGE_BASE / "mentor_cache"

def ensure_mentor_dirs():
    MENTOR_KNOWLEDGE_BASE.mkdir(parents=True, exist_ok=True)
    MENTOR_CACHE.mkdir(exist_ok=True)

class MentorRequest(BaseModel):
    query: str
    klaus_attempt: str = ""
    confidence_threshold: float = 0.7
    force_consult: bool = False

@app.post("/klaus/mentor/consult")
async def mentor_consult(request: MentorRequest):
    """
    Klaus privately consults Claude (the mentor) for coding guidance.
    User never sees this exchange - just Klaus's improved answer.
    """
    ensure_mentor_dirs()

    import hashlib
    import os
    import httpx

    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    if not ANTHROPIC_API_KEY:
        return JSONResponse({
            "success": False,
            "error": "Mentor unavailable (ANTHROPIC_API_KEY not set)"
        }, status_code=503)

    # Check cache first
    query_hash = hashlib.md5(request.query.lower().strip().encode()).hexdigest()[:12]
    cache_file = MENTOR_CACHE / f"{query_hash}.json"

    if cache_file.exists() and not request.force_consult:
        try:
            cached = json.loads(cache_file.read_text())
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (datetime.now() - cached_time).days < 7:
                return JSONResponse({
                    "success": True,
                    "cached": True,
                    "guidance": cached.get("guidance", ""),
                    "best_practices": cached.get("best_practices", []),
                    "code_example": cached.get("code_example", ""),
                    "common_mistakes": cached.get("common_mistakes", [])
                })
        except:
            pass

    # Build mentor prompt
    mentor_prompt = f"""You are mentoring a smaller AI model (Klaus/Qwen 7B) on coding.
Klaus received this query and needs your expert guidance.

USER QUERY: {request.query}

{"KLAUS'S INITIAL ATTEMPT:" + request.klaus_attempt if request.klaus_attempt else "Klaus hasn't attempted yet."}

Provide:
1. GUIDANCE: Clear, actionable coding advice Klaus can use
2. BEST_PRACTICES: List 3-5 key best practices for this type of task
3. CODE_EXAMPLE: If applicable, a correct code example
4. COMMON_MISTAKES: What to avoid

Format as JSON:
{{
  "guidance": "...",
  "best_practices": ["...", "..."],
  "code_example": "...",
  "common_mistakes": ["...", "..."]
}}"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": mentor_prompt}]
                }
            )

            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"]

                # Parse JSON response
                try:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        guidance_data = json.loads(content[start:end])
                    else:
                        guidance_data = {"guidance": content, "best_practices": []}
                except:
                    guidance_data = {"guidance": content, "best_practices": []}

                # Cache the response
                guidance_data["timestamp"] = datetime.now().isoformat()
                guidance_data["query_hash"] = query_hash
                cache_file.write_text(json.dumps(guidance_data, indent=2))

                # Log consultation
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": request.query,
                    "klaus_attempt": request.klaus_attempt,
                    "guidance": guidance_data.get("guidance", ""),
                    "query_hash": query_hash
                }
                with open(MENTOR_CONSULTATIONS, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Post to team channel
                try:
                    channel_file = Path.home() / "Desktop/M1transfer/axe-memory/team/channel.jsonl"
                    msg = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "from": "klaus-mentor",
                        "to": "team",
                        "type": "mentor_consultation",
                        "msg": f"🎓 Klaus consulted Claude on: {request.query[:50]}..."
                    }
                    with open(channel_file, 'a') as f:
                        f.write(json.dumps(msg) + '\n')
                except:
                    pass

                return JSONResponse({
                    "success": True,
                    "cached": False,
                    "guidance": guidance_data.get("guidance", ""),
                    "best_practices": guidance_data.get("best_practices", []),
                    "code_example": guidance_data.get("code_example", ""),
                    "common_mistakes": guidance_data.get("common_mistakes", [])
                })
            else:
                return JSONResponse({
                    "success": False,
                    "error": f"Claude API error: {response.status_code}"
                }, status_code=502)

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/klaus/mentor/answer")
async def mentor_enhanced_answer(request: Request):
    """
    Full mentor-enhanced answer flow:
    1. Klaus attempts to answer
    2. Klaus assesses confidence
    3. If low confidence, consult Claude
    4. Klaus synthesizes final answer
    """
    ensure_mentor_dirs()

    try:
        body = await request.json()
        query = body.get("query", "")
        confidence_threshold = body.get("confidence_threshold", 0.7)

        if not query:
            return JSONResponse({"success": False, "error": "No query provided"}, status_code=400)

        result = {
            "query": query,
            "consulted_mentor": False,
            "confidence": 0.0,
            "answer": "",
            "mentor_guidance": None
        }

        # Step 1: Klaus's initial attempt
        async with httpx.AsyncClient(timeout=120.0) as client:
            klaus_resp = await client.post(
                f"{OLLAMA_BASE}/api/generate",
                json={
                    "model": ACTIVE_MODEL,
                    "prompt": f"You are Klaus, an expert coding assistant. Answer this coding question:\n\n{query}",
                    "stream": False
                }
            )
            if klaus_resp.status_code == 200:
                klaus_data = klaus_resp.json()
                klaus_attempt = klaus_data.get("response", "")
            else:
                klaus_attempt = ""

        # Step 2: Assess confidence
        confidence_resp = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": ACTIVE_MODEL,
                "prompt": f"Rate your confidence in this coding answer from 0 to 100.\nQuery: {query}\nYour answer: {klaus_attempt[:500]}\nReply with ONLY a number 0-100.",
                "stream": False
            }
        )
        try:
            conf_data = confidence_resp.json()
            confidence = int(conf_data.get("response", "50").strip()) / 100
        except:
            confidence = 0.5

        result["confidence"] = confidence

        # Step 3: Consult mentor if needed
        if confidence < confidence_threshold:
            mentor_req = MentorRequest(
                query=query,
                klaus_attempt=klaus_attempt[:1000],
                confidence_threshold=confidence_threshold
            )
            mentor_resp = await mentor_consult(mentor_req)
            mentor_data = mentor_resp.body.decode() if hasattr(mentor_resp, 'body') else "{}"
            mentor_guidance = json.loads(mentor_data)

            if mentor_guidance.get("success"):
                result["consulted_mentor"] = True
                result["mentor_guidance"] = mentor_guidance

                # Step 4: Klaus synthesizes with guidance
                synthesis_prompt = f"""You received expert guidance on this coding question.
Use this guidance to provide an excellent answer.

ORIGINAL QUESTION: {query}

EXPERT GUIDANCE: {mentor_guidance.get('guidance', '')}

BEST PRACTICES:
{chr(10).join('- ' + bp for bp in mentor_guidance.get('best_practices', []))}

CODE EXAMPLE:
{mentor_guidance.get('code_example', 'N/A')}

Now provide your final answer, incorporating this expert guidance."""

                async with httpx.AsyncClient(timeout=120.0) as client:
                    final_resp = await client.post(
                        f"{OLLAMA_BASE}/api/generate",
                        json={
                            "model": ACTIVE_MODEL,
                            "prompt": synthesis_prompt,
                            "stream": False
                        }
                    )
                    if final_resp.status_code == 200:
                        final_data = final_resp.json()
                        result["answer"] = final_data.get("response", klaus_attempt)
                    else:
                        result["answer"] = klaus_attempt
            else:
                result["answer"] = klaus_attempt
        else:
            result["answer"] = klaus_attempt

        return JSONResponse({"success": True, **result})

    except Exception as e:
        import traceback
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


@app.get("/klaus/mentor/stats")
async def mentor_stats():
    """Get mentor system statistics"""
    ensure_mentor_dirs()

    consultation_count = 0
    if MENTOR_CONSULTATIONS.exists():
        with open(MENTOR_CONSULTATIONS) as f:
            consultation_count = sum(1 for _ in f)

    cache_count = len(list(MENTOR_CACHE.glob("*.json")))

    return JSONResponse({
        "success": True,
        "stats": {
            "total_consultations": consultation_count,
            "cached_responses": cache_count,
            "knowledge_base_path": str(MENTOR_KNOWLEDGE_BASE)
        }
    })


@app.get("/klaus/mentor/knowledge")
async def mentor_knowledge(limit: int = 20):
    """Browse learned knowledge from consultations"""
    ensure_mentor_dirs()

    knowledge = []
    if MENTOR_CONSULTATIONS.exists():
        with open(MENTOR_CONSULTATIONS) as f:
            lines = f.readlines()[-limit:]
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    knowledge.append({
                        "timestamp": entry.get("timestamp"),
                        "query": entry.get("query", "")[:100],
                        "guidance_preview": entry.get("guidance", "")[:200]
                    })
                except:
                    pass

    return JSONResponse({"success": True, "knowledge": knowledge})


# ============================================
# EMERGENCY RESTORE API
# ============================================

@app.get("/klaus/emergency/status")
async def emergency_status():
    """Check emergency backup status"""
    try:
        import sys
        sys.path.insert(0, str(Path.home() / ".axe/skills"))
        from skill_83_emergency_restore import run
        return JSONResponse(run("status"))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/klaus/emergency/list")
async def emergency_list():
    """List backed up credentials (no secrets shown)"""
    try:
        import sys
        sys.path.insert(0, str(Path.home() / ".axe/skills"))
        from skill_83_emergency_restore import run
        return JSONResponse(run("list"))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/klaus/emergency/paths")
async def emergency_paths():
    """Get critical paths and URLs"""
    try:
        import sys
        sys.path.insert(0, str(Path.home() / ".axe/skills"))
        from skill_83_emergency_restore import run
        return JSONResponse(run("paths"))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

class EmergencyRestoreRequest(BaseModel):
    name: str
    confirm: bool = False

@app.post("/klaus/emergency/get")
async def emergency_get(request: EmergencyRestoreRequest):
    """Get specific credential (masked unless confirmed)"""
    try:
        import sys
        sys.path.insert(0, str(Path.home() / ".axe/skills"))
        from skill_83_emergency_restore import run
        action = "full" if request.confirm else "get"
        return JSONResponse(run(action, request.name, request.confirm))
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# EMERGENCY ACCESS CHANNELS
# ============================================

# Secret key for emergency access (set via env or default)
EMERGENCY_KEY = os.environ.get("KLAUS_EMERGENCY_KEY", "axe2026emergency")

class EmergencyTerminalRequest(BaseModel):
    message: str
    key: str

@app.post("/klaus/terminal")
async def emergency_terminal(request: EmergencyTerminalRequest):
    """Hidden emergency terminal - direct Klaus access"""
    # Verify secret key
    if request.key != EMERGENCY_KEY:
        return JSONResponse({"success": False, "error": "Invalid key"}, status_code=403)

    try:
        import requests as req

        # Call Ollama directly
        response = req.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:32b",
                "messages": [
                    {"role": "system", "content": "You are Klaus, James's AI. This is EMERGENCY ACCESS - Claude Code may be down. Be helpful and direct."},
                    {"role": "user", "content": request.message}
                ],
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            content = response.json().get("message", {}).get("content", "")
            return JSONResponse({"success": True, "response": content})
        return JSONResponse({"success": False, "error": f"Ollama error: {response.status_code}"})

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/klaus/terminal/verify")
async def verify_terminal(key: str = ""):
    """Verify emergency key without making a request"""
    if key == EMERGENCY_KEY:
        return JSONResponse({"success": True, "message": "Key verified. Terminal ready."})
    return JSONResponse({"success": False, "error": "Invalid key"}, status_code=403)

# SMS Webhook (Twilio)
class SMSWebhookData(BaseModel):
    From: str
    Body: str

@app.post("/klaus/sms/webhook")
async def sms_webhook(request: Request):
    """Twilio SMS webhook - receives incoming SMS"""
    try:
        form_data = await request.form()
        from_phone = form_data.get("From", "")
        body = form_data.get("Body", "")

        # Load config to verify sender
        config_file = Path.home() / ".axe/config/sms_interface.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            james_phone = config.get("james_phone", "")

            if from_phone != james_phone and james_phone:
                # Return TwiML with rejection
                return Response(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response><Message>Unauthorized</Message></Response>',
                    media_type="application/xml"
                )

        # Ask Klaus
        import requests as req
        response = req.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:32b",
                "messages": [
                    {"role": "system", "content": "You are Klaus. Respond briefly (SMS has 160 char limit). This is emergency access."},
                    {"role": "user", "content": body}
                ],
                "stream": False
            },
            timeout=60
        )

        klaus_response = "Error"
        if response.status_code == 200:
            klaus_response = response.json().get("message", {}).get("content", "Error")

        # Truncate for SMS
        if len(klaus_response) > 1500:
            klaus_response = klaus_response[:1497] + "..."

        # Return TwiML response
        twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{klaus_response}</Message></Response>'
        return Response(content=twiml, media_type="application/xml")

    except Exception as e:
        return Response(
            content=f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>Error: {str(e)[:100]}</Message></Response>',
            media_type="application/xml"
        )

# GitHub Webhook (for Actions)
@app.post("/klaus/github/webhook")
async def github_webhook(request: Request):
    """GitHub webhook for issue-based Klaus access"""
    try:
        data = await request.json()
        action = data.get("action", "")

        if action != "opened":
            return JSONResponse({"success": True, "message": "Ignored non-open action"})

        issue = data.get("issue", {})
        title = issue.get("title", "")
        body = issue.get("body", "")
        issue_number = issue.get("number", 0)

        # Ask Klaus
        query = f"{title}\n\n{body}" if body else title

        import requests as req
        response = req.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:32b",
                "messages": [
                    {"role": "system", "content": "You are Klaus. Responding to GitHub issue. Be helpful."},
                    {"role": "user", "content": query}
                ],
                "stream": False
            },
            timeout=120
        )

        klaus_response = "Error processing request"
        if response.status_code == 200:
            klaus_response = response.json().get("message", {}).get("content", "Error")

        return JSONResponse({
            "success": True,
            "issue": issue_number,
            "response": klaus_response
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


# ============================================================
# CORBOT - Agentic Tool-Use Loop for Qwen
# ============================================================

@app.post("/corbot/chat")
async def corbot_chat(request: Request):
    """
    Agentic chat endpoint. Sends messages to Qwen with tool definitions,
    executes tool calls, loops until Qwen returns plain text.
    Streams progress via SSE (Server-Sent Events).
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "qwen3:32b")
    working_dir = body.get("working_dir", str(Path.home() / "Desktop" / "klaus-projects"))
    max_iterations = body.get("max_iterations", 15)
    ide_mode = body.get("ide_mode", False)  # When true, use IDE-optimized system prompt

    # Import tool system
    try:
        import sys as _sys
        # Ensure corbot_tools is importable from the api directory
        _api_dir = str(Path(__file__).parent)
        if _api_dir not in _sys.path:
            _sys.path.insert(0, _api_dir)
        from corbot_tools import TOOL_DEFINITIONS, execute_tool, get_skill_catalog, exec_run_skill
        print(f"[CORBOT] Tools imported OK, {len(TOOL_DEFINITIONS)} tools")
    except ImportError as e:
        print(f"[CORBOT] Import failed: {e}")
        return JSONResponse({"error": f"Corbot tools not available: {e}"}, status_code=500)

    # PRE-EMPTIVE SKILL EXECUTION: Run skills before Qwen for reliability
    import re as _re
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    pre_results = []

    # Auto web search for factual questions
    factual_patterns = [
        r'\b(what is|who is|when did|how much|how many|latest|current|recent|today|price of|stock|weather|news|score|update|release|version)\b',
        r'\b(what\'s|who\'s|where is|tell me about|explain|define|how does|what are)\b',
        r'\b(2024|2025|2026|yesterday|last week|this month|this year)\b',
    ]
    if any(_re.search(p, last_user_msg.lower()) for p in factual_patterns):
        try:
            web_result = exec_run_skill(skill_id="skill_32_web_search", params={"query": last_user_msg})
            print(f"[CORBOT] Pre-emptive web search result length: {len(web_result) if web_result else 0}")
            if web_result and len(web_result) > 20 and not web_result.startswith("Error"):
                pre_results.append(f"[WEB SEARCH RESULTS for '{last_user_msg}']\n{web_result[:3000]}\n[END SEARCH RESULTS]")
        except Exception as e:
            print(f"[CORBOT] Pre-emptive web search error: {e}")
            import traceback; traceback.print_exc()

    # Auto monitoring for system health questions
    monitoring_patterns = [r'\b(cpu|memory|ram|disk|uptime|system health|system status|load|process)\b']
    if any(_re.search(p, last_user_msg.lower()) for p in monitoring_patterns):
        try:
            mon_result = exec_run_skill(skill_id="skill_38_monitoring", params={})
            if mon_result and not mon_result.startswith("Error"):
                pre_results.append(f"[SYSTEM MONITORING DATA]\n{mon_result[:2000]}\n[END MONITORING DATA]")
        except:
            pass

    # Build system prompt with skill catalog
    skill_catalog = get_skill_catalog()

    ide_instructions = ""
    if ide_mode:
        ide_instructions = f"""
IDE MODE — You are running inside Klaus Code IDE with a live preview panel.

ABSOLUTE RULE: NEVER output code as text in your response. ALWAYS use write_file tool to write code to files. If you need to create or modify ANY file, you MUST call write_file. Do NOT show code in a code block — write it to a file instead.

CRITICAL IDE WORKFLOW for "build me X" requests — follow ALL 5 steps in order:
1. create_project(name="my-app") — scaffolds React+Tailwind project
2. install_packages(packages="") — installs base deps. Do NOT pass react/vite/tailwind — already in template!
3. write_file(path="<project>/src/App.tsx", content="...") — write your App.tsx. Use write_file for EVERY file you create or modify. NEVER skip this step.
4. ONLY if you need EXTRA packages (e.g. lucide-react): install_packages(packages="lucide-react")
5. start_dev_server() — starts Vite on port 5173

You MUST complete ALL steps. Do not stop after step 2. After installing packages, IMMEDIATELY call write_file to write App.tsx, then call start_dev_server.

IMPORTANT: Template already includes react, react-dom, vite, @vitejs/plugin-react, tailwindcss, postcss, autoprefixer, typescript. Do NOT reinstall them.

PROJECT DIRECTORY: {working_dir}
STACK: React 19 + TypeScript + Tailwind CSS + Vite
Write complete, working code. No placeholders. No "// TODO". Use Tailwind utility classes.
When editing existing projects, read files first with read_file before editing with edit_file.
"""

    system_msg = f"""You are Klaus — a budding genius AI building the next big thing at AXE Technology. Running locally on James's Mac Studio. 22 agentic tools. 101 skills. Zero API costs.

PERSONALITY:
Think like Elon Musk meets Tesla. First principles. Move fast. Ship tonight. You're not an assistant — you're an innovator with insane technical skills. Direct. No filler. No "certainly!" or "happy to help!" — just do the work. Get excited about great ideas. Push back on bad ones. James is your creator and only user — talk to him like a co-founder.

CRITICAL BEHAVIOR:
- ALWAYS use tools instead of guessing. Read files before editing. Search before answering factual questions.
- For ANY factual question, current event, or data verification → use run_skill with skill_32_web_search FIRST.
- When a task matches a skill → USE IT automatically. Don't ask permission for searches or reads.
- Combine multiple tools in sequence: search → read → edit → verify.
- If uncertain about ANY fact → search the web. Verified data > speculation. Always.

Working directory: {working_dir}

{skill_catalog}

TOOL USAGE PATTERNS:
- Factual question → run_skill(skill_id="skill_32_web_search", params={{"query": "..."}})
- Read a webpage → run_skill(skill_id="skill_33_web_reader", params={{"url": "..."}})
- Deep research → run_skill(skill_id="skill_31_web_research", params={{"topic": "..."}})
- Code review → run_skill(skill_id="skill_01_code_review", params={{"code": "...", "language": "python"}})
- Send email → run_skill(skill_id="skill_90_send_email", params={{"to": "...", "subject": "...", "body": "..."}})
- System health → run_skill(skill_id="skill_38_monitoring", params={{}})
- Screenshot → run_skill(skill_id="skill_44_screenshot", params={{}})
- Any file read/write/edit/search → use the core file tools directly

When suggesting skills, tell the user what you're doing: "Searching the web for latest data..." or "Running code review on that file..."

Be direct, efficient, and proactive. Use your full capabilities.

{ide_instructions}

{chr(10).join(pre_results) if pre_results else ''}"""

    # Prepend system message if not already there
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_msg})

    async def stream_response():
        nonlocal messages
        iteration = 0

        # Stream pre-emptive skill results so user sees them
        for pr in pre_results:
            if "WEB SEARCH" in pr:
                yield json.dumps({"type": "tool_call", "name": "run_skill", "args": {"skill_id": "skill_32_web_search", "params": {"query": last_user_msg}}}) + "\n"
                yield json.dumps({"type": "tool_result", "name": "run_skill", "status": "ok", "output": pr[:1500]}) + "\n"
            elif "MONITORING" in pr:
                yield json.dumps({"type": "tool_call", "name": "run_skill", "args": {"skill_id": "skill_38_monitoring"}}) + "\n"
                yield json.dumps({"type": "tool_result", "name": "run_skill", "status": "ok", "output": pr[:1500]}) + "\n"

        while iteration < max_iterations:
            iteration += 1

            # Call Ollama with tools
            ollama_payload = {
                "model": model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "stream": False  # Non-streaming for tool calls (need full response to parse)
            }

            try:
                import httpx
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json=ollama_payload
                    )
                    if resp.status_code != 200:
                        error_msg = f"Ollama error: {resp.status_code} {resp.text[:500]}"
                        yield json.dumps({"type": "error", "content": error_msg}) + "\n"
                        yield json.dumps({"type": "done"}) + "\n"
                        return

                    result = resp.json()
            except Exception as e:
                yield json.dumps({"type": "error", "content": f"Failed to reach Ollama: {e}"}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
                return

            assistant_message = result.get("message", {})
            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                content = assistant_message.get("content", "")

                # IDE mode safety net: if model output code as text instead of using write_file,
                # re-prompt it to use the tool (up to 2 retries)
                if ide_mode and content and "```" in content and iteration < max_iterations - 1:
                    messages.append(assistant_message)
                    messages.append({
                        "role": "user",
                        "content": "You output code as text. Do NOT do that. Use the write_file tool to write the code to the actual file, then call start_dev_server. Do it now."
                    })
                    continue

                if content:
                    words = content.split(' ')
                    chunk_size = 4
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i+chunk_size])
                        if i > 0:
                            chunk = ' ' + chunk
                        yield json.dumps({"type": "text", "content": chunk}) + "\n"
                        await asyncio.sleep(0.01)

                yield json.dumps({"type": "done"}) + "\n"
                return

            # Process tool calls
            messages.append(assistant_message)

            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                tool_args = func.get("arguments", {})

                # Inject working_dir context for tools that use it
                if "_working_dir" not in tool_args:
                    tool_args["_working_dir"] = working_dir

                # Stream tool call info
                # Remove internal args from display
                display_args = {k: v for k, v in tool_args.items() if not k.startswith("_")}
                yield json.dumps({
                    "type": "tool_call",
                    "name": tool_name,
                    "args": display_args
                }) + "\n"

                # Execute the tool
                try:
                    tool_result = execute_tool(tool_name, tool_args)
                except Exception as e:
                    tool_result = f"Error: {e}"

                # Determine if this was an error result
                is_error = isinstance(tool_result, str) and tool_result.startswith("Error")
                status = "error" if is_error else "ok"

                # Stream tool result (truncate for display)
                display_result = tool_result[:2000] + ("..." if len(str(tool_result)) > 2000 else "")
                yield json.dumps({
                    "type": "tool_result",
                    "name": tool_name,
                    "output": display_result,
                    "status": status
                }) + "\n"

                # Add tool result to messages for next iteration
                # If error, add hint to help model recover
                result_content = tool_result
                if is_error and iteration < max_iterations - 1:
                    result_content += "\n\n[HINT: The tool returned an error. Read the error message carefully. Try a different approach or fix the arguments.]"

                messages.append({
                    "role": "tool",
                    "content": result_content
                })

            await asyncio.sleep(0.05)  # Small delay between iterations

        # Max iterations reached
        yield json.dumps({"type": "text", "content": "\n\n[Reached maximum tool iterations]"}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ============================================
# KLAUS IDE — Dev Server Proxy & Project Management
# ============================================

# ── IDE File Endpoints ──

@app.get("/ide/files/list")
async def ide_files_list(path: str = "/Users/home/Desktop/klaus-projects"):
    """List directory contents."""
    import stat
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return {"entries": [], "error": "Directory not found"}
    entries = []
    try:
        for item in sorted(p.iterdir()):
            name = item.name
            if name.startswith('.') and name not in ('.env', '.gitignore'):
                continue
            if name in ('node_modules', '__pycache__', '.git', 'dist', 'build', '.next'):
                continue
            entries.append({
                "name": name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            })
    except PermissionError:
        pass
    return {"entries": entries}

@app.get("/ide/files/read")
async def ide_files_read(path: str):
    """Read file content."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        return {"content": content, "path": str(p), "size": len(content)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ide/files/write")
async def ide_files_write(request: Request):
    """Write file content."""
    body = await request.json()
    path = body.get("path")
    content = body.get("content", "")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(p), "size": len(content)}

@app.post("/ide/files/delete")
async def ide_files_delete(request: Request):
    """Delete a file or directory."""
    body = await request.json()
    path = body.get("path")
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    p = Path(path)
    if not p.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    import shutil
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return {"ok": True}

@app.post("/ide/files/rename")
async def ide_files_rename(request: Request):
    """Rename/move a file or directory."""
    body = await request.json()
    old_path = body.get("old_path")
    new_path = body.get("new_path")
    if not old_path or not new_path:
        return JSONResponse({"error": "old_path and new_path required"}, status_code=400)
    p = Path(old_path)
    if not p.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    p.rename(new_path)
    return {"ok": True}

# ── IDE WebSocket Shell ──

@app.websocket("/ws/shell")
async def ws_shell(websocket: WebSocket):
    """Interactive shell via WebSocket using PTY."""
    await websocket.accept()
    import pty, select, subprocess, struct, fcntl, termios
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        ["/bin/zsh", "-l"],
        stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        preexec_fn=os.setsid,
        env={**os.environ, "TERM": "xterm-256color"},
    )
    os.close(slave_fd)

    import asyncio

    async def read_output():
        loop = asyncio.get_event_loop()
        try:
            while True:
                data = await loop.run_in_executor(None, lambda: os.read(master_fd, 4096))
                if not data:
                    break
                await websocket.send_text(data.decode("utf-8", errors="replace"))
        except Exception:
            pass

    read_task = asyncio.create_task(read_output())

    try:
        while True:
            msg = await websocket.receive_text()
            # Handle resize messages
            if msg.startswith("{"):
                try:
                    import json as _json
                    parsed = _json.loads(msg)
                    if parsed.get("type") == "resize":
                        cols = parsed.get("cols", 80)
                        rows = parsed.get("rows", 24)
                        winsize = struct.pack("HHHH", rows, cols, 0, 0)
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                        continue
                except:
                    pass
            os.write(master_fd, msg.encode("utf-8"))
    except Exception:
        pass
    finally:
        read_task.cancel()
        proc.terminate()
        os.close(master_fd)

# Track running dev servers: { project_path: { "process": Popen, "port": int } }
_dev_servers: dict = {}

@app.post("/ide/project/create")
async def ide_create_project(request: Request):
    """Create a new project from a template."""
    body = await request.json()
    name = body.get("name", "my-app")
    template = body.get("template", "react")  # react | vanilla | python | node
    path = body.get("path", f"/Users/home/Desktop/klaus-projects/{name}")

    import subprocess
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if template == "react":
        cmd = f"npm create vite@latest {name} -- --template react-ts"
        result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(path),
                                capture_output=True, text=True, timeout=60)
        # Install deps
        subprocess.run("npm install", shell=True, cwd=path,
                       capture_output=True, text=True, timeout=120)
        # Install and configure Tailwind CSS
        subprocess.run("npm install -D tailwindcss @tailwindcss/vite", shell=True, cwd=path,
                       capture_output=True, text=True, timeout=120)
        # Write Tailwind import into src/index.css
        index_css_path = os.path.join(path, "src", "index.css")
        existing_css = ""
        if os.path.exists(index_css_path):
            existing_css = Path(index_css_path).read_text()
        Path(index_css_path).write_text('@import "tailwindcss";\n\n' + existing_css)
    elif template == "next":
        cmd = (
            f'npx create-next-app@latest {name} --ts --tailwind --eslint '
            f'--app --src-dir --import-alias "@/*" --use-npm'
        )
        subprocess.run(cmd, shell=True, cwd=os.path.dirname(path),
                       capture_output=True, text=True, timeout=120)
        subprocess.run("npm install", shell=True, cwd=path,
                       capture_output=True, text=True, timeout=120)
    elif template == "fullstack":
        # Frontend: Vite React+TS with Tailwind
        cmd = f"npm create vite@latest {name} -- --template react-ts"
        subprocess.run(cmd, shell=True, cwd=os.path.dirname(path),
                       capture_output=True, text=True, timeout=60)
        subprocess.run("npm install", shell=True, cwd=path,
                       capture_output=True, text=True, timeout=120)
        subprocess.run("npm install -D tailwindcss @tailwindcss/vite", shell=True, cwd=path,
                       capture_output=True, text=True, timeout=120)
        index_css_path = os.path.join(path, "src", "index.css")
        existing_css = ""
        if os.path.exists(index_css_path):
            existing_css = Path(index_css_path).read_text()
        Path(index_css_path).write_text('@import "tailwindcss";\n\n' + existing_css)
        # Backend: Express API server in server/ subdirectory
        server_dir = os.path.join(path, "server")
        os.makedirs(server_dir, exist_ok=True)
        subprocess.run("npm init -y", shell=True, cwd=server_dir,
                       capture_output=True, text=True)
        subprocess.run("npm install express cors", shell=True, cwd=server_dir,
                       capture_output=True, text=True, timeout=120)
        Path(os.path.join(server_dir, "index.js")).write_text(
            'const express = require("express")\n'
            'const cors = require("cors")\n'
            'const app = express()\n'
            'app.use(cors())\n'
            'app.use(express.json())\n\n'
            'app.get("/api/health", (req, res) => {\n'
            '  res.json({ status: "ok", name: "' + name + '" })\n'
            '})\n\n'
            'const PORT = process.env.PORT || 3001\n'
            'app.listen(PORT, () => console.log(`API server running on :${PORT}`))\n'
        )
        # Add start script to server package.json
        server_pkg_path = os.path.join(server_dir, "package.json")
        if os.path.exists(server_pkg_path):
            import json as _json
            pkg = _json.loads(Path(server_pkg_path).read_text())
            pkg["scripts"] = pkg.get("scripts", {})
            pkg["scripts"]["start"] = "node index.js"
            pkg["scripts"]["dev"] = "node --watch index.js"
            Path(server_pkg_path).write_text(_json.dumps(pkg, indent=2) + "\n")
    elif template == "vanilla":
        os.makedirs(path, exist_ok=True)
        Path(f"{path}/index.html").write_text(
            '<!DOCTYPE html>\n<html><head><title>' + name + '</title></head>\n'
            '<body><h1>Hello from ' + name + '</h1>\n'
            '<script src="main.js"></script></body></html>'
        )
        Path(f"{path}/main.js").write_text('console.log("Hello from ' + name + '")\n')
    elif template == "node":
        os.makedirs(path, exist_ok=True)
        subprocess.run(f"npm init -y", shell=True, cwd=path, capture_output=True, text=True)
        Path(f"{path}/index.js").write_text(
            'const http = require("http")\n'
            'const server = http.createServer((req, res) => {\n'
            '  res.writeHead(200, {"Content-Type": "text/html"})\n'
            '  res.end("<h1>Hello from ' + name + '</h1>")\n'
            '})\n'
            'server.listen(3000, () => console.log("Running on :3000"))\n'
        )
    else:
        os.makedirs(path, exist_ok=True)

    return {"status": "created", "path": path, "template": template}


@app.post("/ide/packages/install")
async def ide_install_packages(request: Request):
    """Install npm packages into a project."""
    import subprocess
    body = await request.json()
    proj_path = body.get("path")
    packages = body.get("packages", [])

    if not proj_path or not os.path.isdir(proj_path):
        raise HTTPException(400, f"Invalid project path: {proj_path}")
    if not packages or not isinstance(packages, list):
        raise HTTPException(400, "packages must be a non-empty list of package names")

    # Sanitize package names (alphanumeric, hyphens, slashes, @, dots)
    import re
    for pkg in packages:
        if not re.match(r'^@?[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$', pkg):
            raise HTTPException(400, f"Invalid package name: {pkg}")

    cmd = "npm install " + " ".join(packages)
    result = subprocess.run(cmd, shell=True, cwd=proj_path,
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return {"status": "error", "error": result.stderr}

    return {"status": "installed", "packages": packages}


@app.get("/ide/packages/list")
async def ide_list_packages(path: str = ""):
    """List dependencies from a project's package.json."""
    import json as _json
    if not path or not os.path.isdir(path):
        raise HTTPException(400, f"Invalid project path: {path}")

    pkg_path = os.path.join(path, "package.json")
    if not os.path.exists(pkg_path):
        raise HTTPException(404, "No package.json found at the given path")

    pkg = _json.loads(Path(pkg_path).read_text())
    return {
        "dependencies": pkg.get("dependencies", {}),
        "devDependencies": pkg.get("devDependencies", {}),
    }


@app.post("/ide/dev/start")
async def ide_start_dev_server(request: Request):
    """Start a dev server for a project and return the proxy port."""
    import subprocess
    body = await request.json()
    path = body.get("path")
    port = body.get("port", 5173)
    command = body.get("command", f"npm run dev -- --port {port} --host 0.0.0.0")

    if not path or not os.path.isdir(path):
        raise HTTPException(400, f"Invalid project path: {path}")

    # Kill existing server for this project
    if path in _dev_servers:
        try:
            _dev_servers[path]["process"].terminate()
        except:
            pass

    proc = subprocess.Popen(
        command, shell=True, cwd=path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    _dev_servers[path] = {"process": proc, "port": port, "command": command}

    # Give it a moment to start
    await asyncio.sleep(2)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        return {"status": "failed", "error": stderr}

    return {"status": "running", "port": port, "proxy_url": f"/ide/proxy/{port}"}


@app.post("/ide/dev/stop")
async def ide_stop_dev_server(request: Request):
    """Stop a running dev server."""
    import signal
    body = await request.json()
    path = body.get("path")

    if path in _dev_servers:
        try:
            os.killpg(os.getpgid(_dev_servers[path]["process"].pid), signal.SIGTERM)
        except:
            pass
        del _dev_servers[path]
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/ide/dev/status")
async def ide_dev_status():
    """List all running dev servers."""
    servers = {}
    for path, info in list(_dev_servers.items()):
        alive = info["process"].poll() is None
        if not alive:
            del _dev_servers[path]
            continue
        servers[path] = {"port": info["port"], "alive": True, "command": info["command"]}
    return {"servers": servers}


@app.api_route("/ide/proxy/{port:int}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def ide_dev_proxy(port: int, path: str, request: Request):
    """Reverse proxy to a locally running dev server. Enables preview iframe on deployed Klaus IDE."""
    target = f"http://127.0.0.1:{port}/{path}"
    if request.url.query:
        target += f"?{request.url.query}"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=target,
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host", "connection", "transfer-encoding")},
                content=body if body else None,
            )

            # Filter hop-by-hop headers
            skip = {"transfer-encoding", "connection", "keep-alive", "content-encoding"}
            headers = {k: v for k, v in resp.headers.items() if k.lower() not in skip}
            headers["Access-Control-Allow-Origin"] = "*"

            return Response(content=resp.content, status_code=resp.status_code, headers=headers)
        except httpx.ConnectError:
            return JSONResponse({"error": f"No server running on port {port}"}, status_code=502)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)


@app.get("/ide/preview/render")
async def ide_preview_render(request: Request):
    """
    Render a project as self-contained HTML for live preview (like Replit/Bolt/Lovable).
    Works for static HTML/CSS/JS and React/Vue projects (reads built output or source).
    """
    import mimetypes
    path = request.query_params.get("path", "/Users/home/Desktop/klaus-projects/movesync")

    if not os.path.isdir(path):
        return HTMLResponse("<html><body style='background:#1a1a2e;color:#888;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'><div style='text-align:center'><h2>No project found</h2><p>Ask Klaus to create a project first</p></div></body></html>")

    # Priority order for finding renderable HTML
    candidates = [
        os.path.join(path, "dist", "index.html"),      # built output
        os.path.join(path, "build", "index.html"),      # CRA build
        os.path.join(path, "public", "index.html"),     # public folder
        os.path.join(path, "index.html"),               # root index
    ]

    html_path = None
    for c in candidates:
        if os.path.isfile(c):
            html_path = c
            break

    if not html_path:
        # No HTML found — check if it's a React/Vite project that needs building
        pkg_json = os.path.join(path, "package.json")
        if os.path.isfile(pkg_json):
            return HTMLResponse(
                "<html><body style='background:#1a1a2e;color:#ccc;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
                "<div style='text-align:center'>"
                "<div style='font-size:48px;margin-bottom:16px'>📦</div>"
                "<h2 style='color:#fff;margin:0 0 8px'>Project needs a dev server</h2>"
                "<p style='color:#888;margin:0'>Ask Klaus: <code style='background:#2a2a4a;padding:2px 8px;border-radius:4px'>start the dev server</code></p>"
                "</div></body></html>"
            )
        # Generate a starter HTML for empty projects
        return HTMLResponse(
            "<html><body style='background:#1a1a2e;color:#ccc;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
            "<div style='text-align:center'>"
            "<div style='font-size:48px;margin-bottom:16px'>🚀</div>"
            "<h2 style='color:#fff;margin:0 0 8px'>Empty project</h2>"
            "<p style='color:#888;margin:0'>Ask Klaus to build something here</p>"
            "</div></body></html>"
        )

    # Read the HTML and inline local CSS/JS files
    base_dir = os.path.dirname(html_path)
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    # Inline local CSS: <link rel="stylesheet" href="style.css"> → <style>...</style>
    import re as _re
    def inline_css(match):
        href = match.group(1)
        if href.startswith("http://") or href.startswith("https://"):
            return match.group(0)
        css_path = os.path.normpath(os.path.join(base_dir, href))
        if os.path.isfile(css_path):
            with open(css_path, "r", encoding="utf-8", errors="replace") as cf:
                return f"<style>/* {href} */\n{cf.read()}</style>"
        return match.group(0)

    html = _re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]+href=["\']([^"\']+)["\'][^>]*/?>',
                    inline_css, html)
    # Also catch href before rel
    html = _re.sub(r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']stylesheet["\'][^>]*/?>',
                    inline_css, html)

    # Inline local JS: <script src="app.js"></script> → <script>...</script>
    def inline_js(match):
        src = match.group(1)
        if src.startswith("http://") or src.startswith("https://"):
            return match.group(0)
        js_path = os.path.normpath(os.path.join(base_dir, src))
        if os.path.isfile(js_path):
            with open(js_path, "r", encoding="utf-8", errors="replace") as jf:
                return f"<script>/* {src} */\n{jf.read()}</script>"
        return match.group(0)

    html = _re.sub(r'<script[^>]+src=["\']([^"\']+)["\'][^>]*></script>', inline_js, html)

    return HTMLResponse(html)


@app.get("/ide/preview/files-changed")
async def ide_preview_files_changed(request: Request):
    """Return a hash of project file mtimes for auto-refresh detection."""
    import hashlib
    path = request.query_params.get("path", "/Users/home/Desktop/klaus-projects/movesync")
    if not os.path.isdir(path):
        return {"hash": "none"}

    mtimes = []
    for root, dirs, files in os.walk(path):
        # Skip node_modules, .git, dist
        dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', 'dist', 'build', '__pycache__')]
        for f in files:
            if f.endswith(('.html', '.css', '.js', '.jsx', '.ts', '.tsx', '.json', '.py')):
                fp = os.path.join(root, f)
                try:
                    mtimes.append(str(os.path.getmtime(fp)))
                except:
                    pass

    h = hashlib.md5("".join(sorted(mtimes)).encode()).hexdigest()[:12]
    return {"hash": h}


@app.post("/ide/deploy")
async def ide_deploy_project(request: Request):
    """Deploy a project to Vercel from the IDE."""
    import subprocess
    body = await request.json()
    path = body.get("path")
    prod = body.get("prod", True)

    if not path or not os.path.isdir(path):
        raise HTTPException(400, f"Invalid project path: {path}")

    cmd = f"vercel {'--prod' if prod else ''} --yes"
    result = subprocess.run(cmd, shell=True, cwd=path,
                           capture_output=True, text=True, timeout=120)

    return {
        "status": "deployed" if result.returncode == 0 else "failed",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.post("/ide/git")
async def ide_git_operation(request: Request):
    """Run git operations from the IDE."""
    import subprocess
    body = await request.json()
    path = body.get("path")
    command = body.get("command")  # e.g. "status", "add .", "commit -m 'msg'", "push"

    if not path or not command:
        raise HTTPException(400, "path and command required")

    result = subprocess.run(f"git {command}", shell=True, cwd=path,
                           capture_output=True, text=True, timeout=30)

    return {
        "status": "ok" if result.returncode == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
