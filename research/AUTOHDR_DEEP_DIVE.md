# AutoHDR.com Deep Dive Research
## Reverse Engineering the Real Estate Photo AI

**Date:** February 5, 2026
**Project:** AutoHDR Clone (Open Source)
**Goal:** Recreate AutoHDR.com's functionality with open-source tools

---

## WHAT IS AUTOHDR.COM?

AutoHDR (autohdr.com) is an **AI-powered real estate photo editing service** based in Austin, TX. NOT to be confused with Microsoft's Windows HDR gaming feature.

### Business Model
- **Pricing:** ~$0.37-$2.00 per edit (vs $0.70-$1.00 industry standard)
- **Turnaround:** 20-30 minutes (vs 12+ hours for human editors)
- **Model:** Pay per download - preview before paying
- **Target:** Real estate photographers

### Core Value Proposition
> "Half the cost of an editor with 30-minute turnaround times"

---

## FEATURES TO RECREATE

### Tier 1: Core HDR Editing
| Feature | Description | Priority |
|---------|-------------|----------|
| **HDR Blending** | Merge bracketed exposures (3-9 shots) | CRITICAL |
| **Window Pulls** | Balance interior/exterior through windows | CRITICAL |
| **Color Correction** | White balance, exposure, contrast | CRITICAL |
| **Perspective Correction** | Straighten verticals/horizontals | HIGH |

### Tier 2: Enhancement Features
| Feature | Description | Priority |
|---------|-------------|----------|
| **Sky Replacement** | Replace overcast/blown skies | HIGH |
| **Grass Greening** | Enhance lawn appearance | MEDIUM |
| **Day to Dusk** | Convert daytime → twilight | HIGH |
| **Virtual Twilight** | Add warm window glow, exterior lights | HIGH |

### Tier 3: Advanced Features
| Feature | Description | Priority |
|---------|-------------|----------|
| **Object Removal** | Remove clutter, cars, people | HIGH |
| **Virtual Staging** | Add furniture to empty rooms | MEDIUM |
| **Fire Effects** | Add flames to fireplaces | LOW |
| **TV Replacement** | Add content to blank screens | LOW |

---

## HOW AUTOHDR WORKS (Deduced Architecture)

### Tech Stack (Inferred)
```
┌─────────────────────────────────────────────────────────────┐
│                     AUTOHDR ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│  FRONTEND                                                    │
│  └── React/Next.js (dark theme, upload UI)                  │
│  └── Preview system (before payment)                        │
├─────────────────────────────────────────────────────────────┤
│  API GATEWAY                                                 │
│  └── Upload handling (multi-file brackets)                  │
│  └── Job queue management                                   │
│  └── Webhook notifications                                  │
├─────────────────────────────────────────────────────────────┤
│  PROCESSING PIPELINE                                         │
│  └── 1. Image Analysis (detect scene type, lighting)        │
│  └── 2. Bracket Alignment (for HDR merge)                   │
│  └── 3. HDR Fusion (multi-exposure blend)                   │
│  └── 4. Enhancement (color, perspective, sharpening)        │
│  └── 5. Optional Effects (sky, twilight, removal)           │
├─────────────────────────────────────────────────────────────┤
│  ML MODELS                                                   │
│  └── HDR Fusion Network (custom trained)                    │
│  └── Stable Diffusion (sky replacement, inpainting)         │
│  └── Segmentation (SAM or custom)                           │
│  └── Color/Tone Mapping (custom CNN)                        │
├─────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE                                              │
│  └── Google Cloud (128+ GPUs per their claim)               │
│  └── 24/7 processing                                        │
│  └── ~$1M+ in training data                                 │
└─────────────────────────────────────────────────────────────┘
```

### Training Data (Their Claim)
- **1 million+ professionally edited images**
- **$1 million+ invested in training data**
- Continuous retraining from user feedback

---

## OPEN SOURCE BUILDING BLOCKS

### HDR Blending/Fusion

| Tool | Stars | Description | Link |
|------|-------|-------------|------|
| **hdrcnn** | 542 | Single-image HDR using deep CNNs | [GitHub](https://github.com/gabrieleilertsen/hdrcnn) |
| **DeepHDR-pytorch** | 27 | Multi-exposure HDR with motion handling | [GitHub](https://github.com/Galaxies99/DeepHDR-pytorch) |
| **AHDRNet** | - | Channel attention for ghost-free HDR | CVPR 2019 |
| **HDR-Transformer** | - | Context-aware transformer | ECCV 2022 |
| **OpenCV** | - | Traditional tone mapping (Reinhard, Drago) | Built-in |

### Sky Replacement

| Tool | Stars | Description | Link |
|------|-------|-------------|------|
| **SkyAR** | 2k+ | Real-time sky replacement + harmonization | [GitHub](https://github.com/jiupinjia/SkyAR) |
| **Sky-Segmentation** | - | 900 sky segmentation datasets | [GitHub](https://github.com/ChengChen-ai/Sky-Segmentation) |
| **Segment Anything (SAM)** | 40k+ | Zero-shot segmentation | [GitHub](https://github.com/facebookresearch/segment-anything) |
| **Grounded-SAM** | 12k+ | Text-guided segmentation | [GitHub](https://github.com/IDEA-Research/Grounded-Segment-Anything) |

### Object Removal (Inpainting)

| Tool | Stars | Description | Link |
|------|-------|-------------|------|
| **LaMa** | 5k+ | Large Mask Inpainting with Fourier Convolutions | [GitHub](https://github.com/advimman/lama) |
| **IOPaint** | 15k+ | LaMa + SD inpainting in one tool | [GitHub](https://github.com/Sanster/IOPaint) |
| **Inpaint-Anything** | 4k+ | SAM + inpainting combined | [GitHub](https://github.com/geekyutao/Inpaint-Anything) |

### Virtual Staging

| Tool | Description | Link |
|------|-------------|------|
| **Stable Diffusion** | Image generation backbone | HuggingFace |
| **ControlNet** | Preserve room structure | HuggingFace |
| **Depth-Anything** | Room depth estimation | HuggingFace |
| **ComfyUI** | Visual workflow builder | [GitHub](https://github.com/comfyanonymous/ComfyUI) |

### Color/Tone Correction

| Tool | Description | Use Case |
|------|-------------|----------|
| **OpenCV** | Traditional color ops | White balance, exposure |
| **Pillow** | Basic image ops | Resize, format conversion |
| **scikit-image** | Advanced image processing | Histogram equalization |
| **rawpy** | RAW file processing | Camera RAW support |

---

## DAY-TO-DUSK ALGORITHM

### How It Works (Deduced)

```
INPUT: Daytime exterior photo

STEP 1: Scene Analysis
├── Detect sky region (segmentation)
├── Detect windows (object detection)
├── Detect light fixtures (object detection)
├── Analyze shadows and light direction

STEP 2: Sky Replacement
├── Segment sky using SAM/U-Net
├── Replace with twilight sky (gradient: deep blue → orange horizon)
├── Match lighting direction
├── Blend edges naturally

STEP 3: Window Illumination
├── Detect all windows
├── Add warm interior glow (2700K-3000K color temp)
├── Create light spill on exterior surfaces
├── Add subtle reflections

STEP 4: Exterior Lighting
├── Add glow to outdoor light fixtures
├── Simulate landscape lighting
├── Create warm pools of light on walkways

STEP 5: Global Adjustments
├── Shift overall color temperature cooler
├── Reduce saturation slightly
├── Adjust shadows to match dusk lighting
├── Add subtle atmospheric haze

OUTPUT: Twilight photo
```

---

## HDR BLENDING ALGORITHM

### Traditional Approach (Debevec 1997)

```python
# Classic HDR pipeline
def traditional_hdr(brackets, exposures):
    # 1. Align images (ECC or feature matching)
    aligned = align_images(brackets)

    # 2. Estimate camera response curve
    response_curve = estimate_response(aligned, exposures)

    # 3. Merge to HDR radiance map
    hdr_radiance = merge_exposures(aligned, exposures, response_curve)

    # 4. Tone map to displayable range
    # Options: Reinhard, Drago, Mantiuk, Fattal
    output = tone_map(hdr_radiance, method='reinhard')

    return output
```

### Modern AI Approach

```python
# Neural HDR pipeline
def neural_hdr(brackets):
    # 1. Align using optical flow or homography
    aligned = align_images(brackets)

    # 2. Feed through HDR fusion network
    # Network learns: exposure fusion + deghosting + tone mapping
    hdr_output = hdr_network(aligned)

    # 3. Optional: enhance with another network
    enhanced = enhancement_network(hdr_output)

    return enhanced
```

---

## IMPLEMENTATION PLAN

### Phase 1: MVP (Week 1-2)
**Goal:** Basic HDR blending + color correction

```
src/
├── core/
│   ├── hdr_merge.py       # OpenCV HDR merge
│   ├── tone_mapping.py    # Reinhard, Drago, custom
│   ├── color_correct.py   # White balance, exposure
│   └── align.py           # Image alignment
├── api/
│   └── main.py            # FastAPI endpoints
└── web/
    └── upload.html        # Simple upload UI
```

**Tech:**
- OpenCV for HDR merging
- FastAPI for backend
- Basic HTML/JS upload

### Phase 2: AI Enhancement (Week 3-4)
**Goal:** Neural HDR + sky replacement

```
src/
├── models/
│   ├── hdr_net.py         # DeepHDR or AHDRNet
│   ├── sky_segmentation.py # SAM-based sky detection
│   └── sky_replace.py     # Sky swap + harmonization
```

**Tech:**
- PyTorch for neural networks
- Segment Anything for segmentation
- Pre-trained sky replacement model

### Phase 3: Day-to-Dusk (Week 5-6)
**Goal:** Full twilight conversion

```
src/
├── models/
│   ├── window_detection.py   # Detect windows
│   ├── light_addition.py     # Add interior glow
│   └── twilight_pipeline.py  # Full day→dusk
```

**Tech:**
- YOLO or Detectron2 for window detection
- Custom light spill rendering
- Color temperature adjustment

### Phase 4: Object Removal (Week 7-8)
**Goal:** Remove clutter, cars, people

```
src/
├── models/
│   ├── inpainting.py      # LaMa integration
│   └── object_mask.py     # SAM + prompt masking
```

**Tech:**
- LaMa for inpainting
- Grounded-SAM for text→mask
- IOPaint as fallback

### Phase 5: Production (Week 9-10)
**Goal:** Web UI + batch processing + API

```
web/
├── app/                   # Next.js frontend
├── components/
│   ├── Upload.tsx
│   ├── Preview.tsx
│   └── Download.tsx
api/
├── queue/                 # Redis job queue
├── workers/               # Processing workers
└── webhooks/             # Completion notifications
```

---

## COST ANALYSIS

### Infrastructure Costs

| Component | Option | Monthly Cost |
|-----------|--------|--------------|
| GPU Compute | RunPod/Lambda | $50-500 |
| Storage | S3/R2 | $10-50 |
| CDN | Cloudflare | Free-$20 |
| Hosting | Vercel/Railway | Free-$50 |
| **Total** | | **$60-620/month** |

### Per-Image Processing Cost

| Model | GPU Time | Cost @$0.50/hr |
|-------|----------|----------------|
| HDR Merge | ~2s | $0.0003 |
| Sky Replace | ~5s | $0.0007 |
| Day-to-Dusk | ~8s | $0.001 |
| Object Removal | ~10s | $0.001 |
| **Total** | ~25s | **~$0.003** |

**Margin:** At $0.37/edit, that's **99% gross margin** on compute alone.

---

## COMPETITIVE LANDSCAPE

| Competitor | Pricing | Turnaround | Key Feature |
|------------|---------|------------|-------------|
| **AutoHDR** | $0.37+ | 30 min | HDR accuracy |
| **Imagen AI** | $0.05-0.10 | Instant | Style learning |
| **Autoenhance.ai** | ~$0.20 | 5 min | API-first |
| **Fotello** | $20-40/mo | Instant | Unlimited |
| **PhotoUp** | $1.50+ | 12-24 hr | Human editors |

### Our Differentiator
- **Open Source** - Self-host option
- **API-First** - Developer friendly
- **Local Processing** - Privacy option
- **Transparent AI** - Show what's being done

---

## SOURCES

- [AutoHDR](https://www.autohdr.com)
- [TryAutoHDR](https://www.tryautohdr.com)
- [Autoenhance.ai - How We Built](https://autoenhance.ai/blog/how-we-built-a-photo-editing-ai-for-the-property-marketing-industry)
- [Eyeconic Shutter - AI Comparison](https://www.eyeconicshutter.com/eyeconic-blog/12-19-2025-ai-editing-for-real-estate-photography-which-program-is-right-for-you)
- [Awesome HDR Imaging](https://github.com/rebeccaeexu/Awesome-High-Dynamic-Range-Imaging)
- [LaMa Inpainting](https://github.com/advimman/lama)
- [SkyAR](https://github.com/jiupinjia/SkyAR)
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [IOPaint](https://github.com/Sanster/IOPaint)
- [hdrcnn](https://github.com/gabrieleilertsen/hdrcnn)

---

*Research compiled by Forge + James*
*Project: Open Source Real Estate Photo AI*
