# Real Estate Photo Editing Pipeline
## From Raw Brackets to MLS-Ready Delivery

**Date:** February 5, 2026
**Goal:** Automate professional-grade real estate photo editing

---

## THE GOLD STANDARD: What Perfect Real Estate Photos Look Like

### Visual Characteristics
1. **Balanced Exposure** - No blown highlights, no crushed shadows
2. **Natural Colors** - Whites are white, wood looks like wood
3. **Window Pulls** - See both interior AND exterior through windows
4. **Straight Lines** - Verticals are vertical, horizontals are horizontal
5. **Consistent White Balance** - No color casts from mixed lighting
6. **Natural Feel** - Not "HDR-y" or oversaturated

### MLS Requirements
| Spec | Requirement |
|------|-------------|
| **Dimensions** | 2048×1365 px (3:2 ratio) minimum |
| **Resolution** | 72 DPI for web, 300 DPI for print |
| **Color Space** | sRGB (mandatory for web) |
| **Format** | JPEG, 75-85% quality |
| **File Size** | Under 10MB (typically 1-3MB optimal) |

### What's NOT Allowed
- Removing permanent fixtures or damage
- Misleading virtual staging (must disclose)
- Visible signage/watermarks
- Pets or people in frame
- Over-processed "HDR look"

---

## BRACKETING: Capturing the Raw Material

### Standard Bracket Sequence
```
Shot 1: -2 EV (underexposed) → Captures window/highlight detail
Shot 2: -1 EV
Shot 3:  0 EV (base exposure) → Middle ground
Shot 4: +1 EV
Shot 5: +2 EV (overexposed) → Captures shadow detail
```

### Camera Settings (Locked)
| Setting | Value | Reason |
|---------|-------|--------|
| **Aperture** | f/8 - f/11 | Sharpness + depth of field |
| **ISO** | 100-320 | Minimize noise |
| **White Balance** | Manual/Kelvin | Consistency across brackets |
| **Mode** | AEB (Auto Exposure Bracketing) | Automated capture |
| **Format** | RAW | Maximum editing flexibility |

### Flash Bracket (Flambient)
```
Ambient Set: 3-5 brackets at natural light only
Flash Set:   1-2 shots with bounced flash (ceiling/wall)
```

---

## THE BLENDING PIPELINE

### Method 1: HDR Merge (Automated)
```
INPUT: 3-5 bracketed exposures

STEP 1: Alignment
├── Feature detection (ORB/SIFT)
├── Homography estimation
└── Warp to reference frame

STEP 2: HDR Merge
├── Debevec: Camera response curve estimation
├── Robertson: Iterative optimization
└── Mertens: Exposure fusion (no curve needed)

STEP 3: Tone Mapping
├── Reinhard: Natural, conservative
├── Drago: More contrast
└── Mantiuk: Local contrast enhancement

OUTPUT: Single HDR-merged image
```

### Method 2: Luminosity Masking (Professional)
```
INPUT: Bracketed exposures as Photoshop layers

STEP 1: Create Luminosity Channels
├── Brights: Ctrl+Click RGB channel → Save as "Lights 1"
├── Refine:  Ctrl+Alt+Shift+Click to intersect → "Lights 2-5"
├── Darks:   Invert Lights → "Darks 1-5"
└── Mids:    Subtract Lights & Darks from full selection

STEP 2: Layer Stack (bottom to top)
├── Base exposure (0 EV)
├── Shadow layer (+2 EV) with Darks mask
├── Highlight layer (-2 EV) with Lights mask
└── Adjustment layers for fine-tuning

STEP 3: Blend Modes
├── Normal: Standard compositing
├── Luminosity: Color preservation
├── Soft Light: Subtle contrast
└── Multiply/Screen: Targeted adjustments

OUTPUT: Manually blended composite
```

### Method 3: Flambient (Flash + Ambient)
```
INPUT: Ambient brackets + Flash shot

STEP 1: 50/50 Base Blend
├── Ambient shot as base layer
├── Flash shot above, blend mode: LUMINOSITY
└── Opacity: 50% (adjust 40-60% to taste)

STEP 2: Window Pull Layer
├── New layer, blend mode: DARKEN
├── Add hide-all mask (black)
├── Select windows with polygon tool
├── Paint white at 30% opacity to reveal

STEP 3: Color Correction
├── Curves adjustment for contrast
├── Color balance for cast removal
└── Selective saturation adjustments

OUTPUT: Flambient composite
```

---

## WINDOW PULL TECHNIQUE

### The Problem
Interior: dark, warm light (2700K-3200K)
Exterior: bright, daylight (5500K-6500K)
Camera can't expose both correctly in one shot.

### The Solution
```
STEP 1: Select Windows
├── Use pen tool or polygonal lasso
├── Include frame edges for natural blend
└── Feather selection 1-3px

STEP 2: Apply Underexposed Layer
├── Paste -2 EV shot into selection
├── Blend mode: Normal or Darken
└── Mask edges with soft brush

STEP 3: Color Match
├── Adjust temperature (exterior is cooler)
├── Match contrast to interior
└── Reduce highlights if still blown

STEP 4: Blend Edges
├── Soft brush at 10-30% opacity
├── Paint mask edges for seamless transition
└── Check at 100% zoom for halos
```

---

## COLOR CORRECTION WORKFLOW

### Step 1: White Balance
```python
# Target: Neutral whites without color cast
# Method: Sample neutral gray/white area

def correct_white_balance(image):
    # Find neutral reference point
    neutral = sample_white_or_gray(image)

    # Calculate correction
    r_mult = 255 / neutral.r
    g_mult = 255 / neutral.g
    b_mult = 255 / neutral.b

    # Apply to entire image
    return apply_multipliers(image, r_mult, g_mult, b_mult)
```

### Step 2: Exposure Correction
```python
# Target: Histogram with detail in shadows and highlights

def correct_exposure(image):
    # Analyze histogram
    shadows = percentile(image, 5)   # Should be > 10
    highlights = percentile(image, 95) # Should be < 245

    # Adjust levels
    black_point = max(0, shadows - 10)
    white_point = min(255, highlights + 10)

    return levels_adjust(image, black_point, white_point)
```

### Step 3: Color Cast Removal
```python
# Mixed lighting creates color casts
# Tungsten (orange), Fluorescent (green), Daylight (blue)

def remove_color_cast(image):
    # Analyze by region
    regions = segment_by_lighting(image)

    for region in regions:
        # Detect dominant cast
        cast = detect_color_cast(region)

        # Apply complementary correction
        if cast == 'orange':
            adjust_blue(region, +15)
        elif cast == 'green':
            adjust_magenta(region, +10)
        elif cast == 'blue':
            adjust_yellow(region, +10)

    return composite_regions(regions)
```

---

## PERSPECTIVE CORRECTION

### Vertical Correction (Most Important)
```
Problem: Walls leaning inward/outward (keystoning)
Cause:   Camera tilted up or down
Fix:     Perspective transform to make verticals parallel

Detection:
├── Detect wall edges using Hough lines
├── Measure angle from vertical (should be 0°)
└── Apply inverse transform
```

### Horizontal Correction
```
Problem: Horizontal lines not level
Cause:   Camera not level, lens distortion
Fix:     Rotation + distortion correction
```

---

## AI ENHANCEMENT PIPELINE

### Our Automated Process
```
INPUT: Raw bracket files (3-5 exposures)

STAGE 1: Pre-Processing
├── Decode RAW files (rawpy/LibRaw)
├── Apply lens correction profiles
├── Extract EXIF for exposure values
└── Detect scene type (interior/exterior)

STAGE 2: Alignment
├── Convert to grayscale for feature detection
├── ORB/SIFT feature matching
├── RANSAC homography estimation
├── Warp all frames to reference

STAGE 3: HDR Merge
├── Mertens fusion (no exposure data needed)
├── OR Debevec merge (if exposure data available)
└── Ghost detection and removal

STAGE 4: AI Enhancement
├── Neural tone mapping (learned from pro edits)
├── Window detection + adaptive exposure
├── Sky detection + optional replacement
├── Color cast detection + correction

STAGE 5: Perspective Correction
├── Vertical line detection
├── Automatic straightening
└── Crop to remove black edges

STAGE 6: Final Polish
├── Sharpening (conservative)
├── Noise reduction (if needed)
├── Contrast fine-tuning
└── Output in delivery specs

OUTPUT: MLS-ready JPEG + full-res archive
```

---

## DELIVERY SPECIFICATIONS

### MLS Package (Standard Delivery)
```
Format:     JPEG
Color:      sRGB
Dimensions: 2048 × 1365 px (3:2)
Quality:    80-85%
DPI:        72
File Size:  1-3 MB typical
Naming:     {address}_{room}_{sequence}.jpg
            Example: 123_main_st_living_room_01.jpg
```

### Print Package (Optional)
```
Format:     JPEG or TIFF
Color:      sRGB or AdobeRGB
Dimensions: Full resolution (original)
Quality:    95-100% (JPEG) or lossless (TIFF)
DPI:        300
Naming:     {address}_{room}_{sequence}_print.jpg
```

### Web Optimized (Social/Thumbnails)
```
Format:     JPEG or WebP
Dimensions: 1200 × 800 px
Quality:    70-75%
File Size:  Under 500KB
```

### Delivery Structure
```
delivery/
├── 123_main_street/
│   ├── mls/
│   │   ├── 01_exterior_front.jpg
│   │   ├── 02_living_room.jpg
│   │   ├── 03_kitchen.jpg
│   │   └── ...
│   ├── print/
│   │   └── (same files, full res)
│   └── web/
│       └── (optimized versions)
```

---

## QUALITY CHECKLIST

### Before Delivery
- [ ] White balance neutral (no color casts)
- [ ] Exposure balanced (shadows and highlights visible)
- [ ] Windows show exterior detail
- [ ] Verticals are straight
- [ ] No visible halos or artifacts
- [ ] Natural saturation (not oversaturated)
- [ ] Correct dimensions for MLS
- [ ] sRGB color space
- [ ] File size under limit
- [ ] Proper file naming

### Common Failures to Avoid
| Issue | Cause | Fix |
|-------|-------|-----|
| Haloing | Over-processed HDR | Reduce local contrast |
| Orange cast | Uncorrected tungsten | Adjust white balance |
| Blown windows | No window pull | Add underexposed layer |
| Crushed blacks | Over-contrast | Lift shadows |
| Soft/blurry | Misaligned brackets | Improve alignment |
| Chromatic aberration | Lens issue | Defringe in post |

---

## IMPLEMENTATION FOR AUTOHDR CLONE

### Phase 1: Core Pipeline
```python
# src/core/pipeline.py

class RealEstatePipeline:
    def process(self, bracket_files):
        # 1. Load and decode
        images = self.load_brackets(bracket_files)

        # 2. Align
        aligned = self.align_images(images)

        # 3. Merge HDR
        merged = self.merge_hdr(aligned)

        # 4. Enhance
        enhanced = self.enhance(merged)

        # 5. Correct perspective
        corrected = self.correct_perspective(enhanced)

        # 6. Export
        return self.export(corrected, specs='mls')
```

### Phase 2: Window Detection
```python
# src/models/window_detector.py

class WindowDetector:
    def __init__(self):
        self.model = load_yolo('windows_v1.pt')

    def detect(self, image):
        boxes = self.model.predict(image)
        masks = self.create_masks(boxes)
        return masks
```

### Phase 3: Intelligent Blending
```python
# src/models/smart_blend.py

class SmartBlender:
    def blend(self, brackets, window_masks):
        # Base merge
        base = self.mertens_merge(brackets)

        # Window-aware blending
        for mask in window_masks:
            underexposed = brackets[0]  # -2 EV
            base = self.blend_region(base, underexposed, mask)

        return base
```

---

## SOURCES

- [PhotoUp - Mastering HDR Blending](https://www.photoup.net/learn/mastering-hdr-blending)
- [PhotoUp - Exposure Blending in Photoshop](https://www.photoup.net/learn/how-to-master-exposure-blending-in-photoshop)
- [Photo & Video Edits - Luminosity Masks](https://www.photoandvideoedits.com/blog/real-estate-photo-editing-a-guide-to-using-luminosity-mask)
- [Fotober - Flambient Blending](https://fotober.com/the-quickest-flambient-blending-in-real-estate-photo-editing)
- [Greg Benz - Luminosity Masking](https://gregbenzphotography.com/luminosity-masking-tutorial/)
- [HomeJab - MLS Photo Requirements](https://homejab.com/how-to-meet-mls-photo-size-requirements/)
- [Stellar MLS - Photo Rules](https://www.stellarmls.com/photorules)
- [HD Estates - Bracketing Explained](https://hdestates.com/blog/bracketing-real-estate/)

---

*Research compiled by Forge + James*
*Project: AutoHDR Clone - Open Source Real Estate Photo AI*
