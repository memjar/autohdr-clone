# AutoHDR Technical Architecture & Processing Pipeline
## Complete Reverse-Engineered Specification

**Date:** February 5, 2026
**Source:** Direct platform analysis

---

## TECHNOLOGY STACK

### Frontend
- Next.js (React-based)
- TypeScript/JavaScript
- Canvas API & WebGL (real-time preview)
- Web Workers (non-blocking operations)
- FormData API (file uploads)

### Backend Integration
- Stripe (payments)
- Dropbox API (automation)
- Google Drive API (file selection)
- Google OAuth (authentication)

---

## PROCESSING PIPELINE (14 Stages)

### Stage 1: HDR Tone Mapping (CORE EFFECT)
```python
def apply_hdr_effect(image, strength=0.7):
    """
    THE PRIMARY EFFECT - Simulates HDR tone mapping
    - Recovers shadow details
    - Compresses highlights
    - Creates the "enhanced" look
    """
    img_float = image.astype(np.float32) / 255.0

    # Compute luminance
    luminance = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)

    # Boost shadows (inverse tone mapping)
    shadows = 1.0 - luminance
    shadow_boost = np.power(shadows, 2.0) * strength

    # Compress highlights
    highlights = luminance
    highlight_compression = np.power(highlights, 0.6)

    # Apply to all channels
    for i in range(3):
        img_float[:,:,i] += shadow_boost
        img_float[:,:,i] *= highlight_compression

    img_float = np.clip(img_float, 0, 1)
    return (img_float * 255).astype(np.uint8)
```

### Stage 2: Brightness Adjustment
```python
def adjust_brightness(image, value):
    """
    Brightness: -2 to +2 scale
    """
    adjusted = image.astype(np.float32)
    adjusted += (value / 2) * 50  # Scale to pixel range
    return np.clip(adjusted, 0, 255).astype(np.uint8)
```

### Stage 3: Contrast Adjustment
```python
def adjust_contrast(image, value):
    """
    Contrast: -2 to +2 scale
    Formula: output = (input - 128) * factor + 128
    """
    factor = 1.0 + (value / 4.0)
    adjusted = image.astype(np.float32)
    adjusted = (adjusted - 128) * factor + 128
    return np.clip(adjusted, 0, 255).astype(np.uint8)
```

### Stage 4: Vibrance Adjustment
```python
def adjust_vibrance(image, value):
    """
    Vibrance: boost saturation without clipping
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= (1.0 + value / 10.0)
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

### Stage 5: White Balance
```python
def adjust_white_balance(image, value):
    """
    Value: -2 to +2 (cooler to warmer)
    Negative = cooler (blue), Positive = warmer (orange)
    """
    adjusted = image.astype(np.float32)

    if value > 0:  # Warmer
        adjusted[:,:,2] *= (1.0 + value * 0.15)  # Red boost
        adjusted[:,:,0] *= (1.0 - value * 0.08)  # Blue reduce
    else:  # Cooler
        adjusted[:,:,0] *= (1.0 - value * 0.15)  # Blue boost
        adjusted[:,:,2] *= (1.0 + value * 0.08)  # Red reduce

    return np.clip(adjusted, 0, 255).astype(np.uint8)
```

### Stage 6: Sky Detection & Enhancement
```python
def detect_sky(image):
    """
    Detect sky using HSV color space
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Sky: high value, low-mid saturation, blue hue
    lower_sky = np.array([90, 0, 100])
    upper_sky = np.array([130, 255, 255])

    sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    return sky_mask

def enhance_sky(image, style='fluffy', retain_original=False):
    if retain_original:
        return image

    sky_mask = detect_sky(image)
    # Generate clouds based on style
    # Blend into sky region
    ...
```

### Stage 7: Window Pull (Enhancement)
```python
def enhance_windows(image, intensity='natural'):
    """
    Brighten windows based on intensity
    Natural = 1.2x, Medium = 1.4x, Strong = 1.6x
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find bright rectangular areas (windows)
    _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright_areas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    window_mask = np.zeros_like(gray)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 0.6 < h/w < 1.4 and w > 20:  # Window-like shape
            cv2.rectangle(window_mask, (x, y), (x+w, y+h), 255, -1)

    intensity_map = {'natural': 1.2, 'medium': 1.4, 'strong': 1.6}
    factor = intensity_map.get(intensity, 1.2)

    adjusted = image.astype(np.float32)
    mask_3ch = cv2.cvtColor(window_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    adjusted = adjusted * (1 + (factor - 1) * mask_3ch)

    return np.clip(adjusted, 0, 255).astype(np.uint8)
```

### Stage 8: Grass Enhancement
```python
def detect_grass(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    return cv2.inRange(hsv, lower_green, upper_green)

def replace_grass(image):
    grass_mask = detect_grass(image)

    # Enhance green in grass regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask_float = grass_mask.astype(np.float32) / 255.0

    # Boost saturation in grass areas
    hsv[:,:,1] = hsv[:,:,1] * (1 - mask_float) + hsv[:,:,1] * 1.4 * mask_float
    # Shift hue slightly toward vibrant green
    hsv[:,:,0] = hsv[:,:,0] * (1 - mask_float) + 60 * mask_float

    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
```

### Stage 9: Sign Removal (Inpainting)
```python
def remove_signs(image):
    """
    Detect rectangular objects and inpaint
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 0.2 < h/w < 5 and w * h > 500:  # Sign-like
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
```

### Stage 10: Perspective Correction
```python
def correct_perspective(image):
    """
    Detect vertical lines and straighten
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image

    # Find dominant vertical angle
    # Apply inverse transform
    h, w = image.shape[:2]
    # ... perspective warp

    return image
```

### Stage 11: Declutter
```python
def declutter(image):
    """
    Bilateral filter - preserves edges, smooths details
    """
    return cv2.bilateralFilter(image, 9, 75, 75)
```

### Stage 12: Twilight Effect
```python
def apply_twilight(image, style='pink'):
    """
    Golden hour color shift
    """
    adjusted = image.astype(np.float32)

    if style == 'pink':
        adjusted[:,:,2] *= 1.3  # Red boost
        adjusted[:,:,1] *= 1.1  # Green slight
        adjusted[:,:,0] *= 0.7  # Blue reduce
    elif style == 'blue':
        adjusted[:,:,0] *= 1.2  # Blue boost
        adjusted[:,:,2] *= 0.85 # Red reduce
    elif style == 'orange':
        adjusted[:,:,2] *= 1.4  # Red boost
        adjusted[:,:,1] *= 1.1  # Green
        adjusted[:,:,0] *= 0.6  # Blue reduce

    # Slight darkening for dusk
    adjusted *= 0.95

    return np.clip(adjusted, 0, 255).astype(np.uint8)
```

### Stage 13: Fire in Fireplace
```python
def add_fireplace_fire(image):
    """
    Detect fireplace region, add fire overlay
    """
    # Detect dark rectangular regions (fireplace opening)
    # Composite fire image/animation
    ...
```

### Stage 14: TV Screen Replacement
```python
def replace_tv_screen(image, replacement_image=None):
    """
    Detect TV screens, replace content
    """
    # Detect rectangular black/dark regions
    # Apply perspective transform to replacement
    # Composite into scene
    ...
```

---

## COMPLETE PIPELINE CLASS

```python
class AutoHDRProcessor:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def process(self, image):
        """Execute full pipeline in correct order"""
        result = image.copy()

        # STAGE 1: HDR TONE MAPPING (CORE)
        result = self.apply_hdr_effect(result, strength=0.7)

        # STAGE 2-5: ADJUSTMENTS
        brightness = self.settings.get('brightness', 0)
        contrast = self.settings.get('contrast', 0)
        vibrance = self.settings.get('vibrance', 0)
        white_balance = self.settings.get('white_balance', 0)

        if brightness != 0:
            result = self.adjust_brightness(result, brightness)
        if contrast != 0:
            result = self.adjust_contrast(result, contrast)
        if vibrance != 0:
            result = self.adjust_vibrance(result, vibrance)
        if white_balance != 0:
            result = self.adjust_white_balance(result, white_balance)

        # STAGE 6: PERSPECTIVE CORRECTION
        if self.settings.get('perspective_correction', True):
            result = self.correct_perspective(result)

        # STAGE 7: GRASS REPLACEMENT
        if self.settings.get('grass_replacement', False):
            result = self.replace_grass(result)

        # STAGE 8: SIGN REMOVAL
        if self.settings.get('sign_removal', False):
            result = self.remove_signs(result)

        # STAGE 9: DECLUTTER
        if self.settings.get('declutter', False):
            result = self.declutter(result)

        # STAGE 10: SKY ENHANCEMENT
        if not self.settings.get('retain_original_sky', False):
            result = self.enhance_sky(
                result,
                style=self.settings.get('cloud_style', 'fluffy')
            )

        # STAGE 11: WINDOW ENHANCEMENT
        window_intensity = self.settings.get('window_pull_intensity', 'natural')
        result = self.enhance_windows(result, window_intensity)

        # STAGE 12: TWILIGHT
        twilight = self.settings.get('twilight_style')
        if twilight:
            result = self.apply_twilight(result, twilight)

        # STAGE 13: FIRE
        if self.settings.get('fire_in_fireplace', False):
            result = self.add_fireplace_fire(result)

        # STAGE 14: TV REPLACEMENT
        if self.settings.get('tv_replacement'):
            result = self.replace_tv_screen(result, self.settings['tv_replacement'])

        return result
```

---

## KEY INSIGHTS

1. **Core Effect = HDR Tone Mapping**: Shadow boost + highlight compression
2. **Order Matters**: HDR → Adjustments → Environmental → Effects
3. **Mask-Based Operations**: Binary masks isolate regions
4. **HSV Color Space**: Used for sky, grass, color detection
5. **Credits = 1 Edit**: One processed image regardless of settings

---

## SETTINGS DATA MODEL

```python
DEFAULT_SETTINGS = {
    # Adjustments (-2 to +2)
    'brightness': 0,
    'contrast': 0,
    'vibrance': 0,
    'white_balance': 0,

    # Scene type
    'scene_type': 'interior',  # interior | exterior

    # Window
    'window_pull_intensity': 'natural',  # natural | medium | strong

    # Sky
    'cloud_style': 'fluffy',  # fluffy | dramatic | clear
    'retain_original_sky': False,
    'interior_clouds': True,
    'exterior_clouds': True,

    # Twilight
    'twilight_style': None,  # None | pink | blue | orange

    # Enhancements
    'perspective_correction': True,
    'grass_replacement': False,
    'sign_removal': False,
    'declutter': False,

    # Special
    'fire_in_fireplace': False,
    'tv_replacement': None,
    'deduplication': False,
}
```

---

*Technical specification for AutoHDR Clone*
