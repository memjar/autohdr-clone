# Target Output Specification
## Reference: AutoHDR Commercial Office Output

**Date:** February 5, 2026
**Source:** DSC07749-07751.ARW (Sony brackets)
**Type:** Commercial real estate - office space

---

## VISUAL TARGET

### Overall Look
- **Clean, bright, professional**
- **Natural** - looks like what the eye sees, not "HDR-y"
- **Balanced** - no area too dark or blown out

### White Balance
- **Target:** ~5200K (neutral daylight)
- **Ceiling whites:** Pure white, no yellow/blue cast
- **Walls:** Clean neutral white
- **Requirement:** Mixed lighting corrected seamlessly

### Exposure Balance
| Area | Target |
|------|--------|
| Ceiling/lights | Properly exposed, no bloom |
| Floor/shadows | Detail visible, not crushed |
| Background | Natural falloff, still visible |
| Windows | Interior/exterior balanced |

### Color Targets
| Element | Target |
|---------|--------|
| Blue accents | Rich, saturated, accurate |
| Wood tones | Natural birch/maple preserved |
| Plants | Vibrant green, not oversaturated |
| Gray floor | True neutral gray |
| Monitors | Black, not lifted |

### Geometry
- Verticals perfectly straight
- No keystoning
- No barrel/pincushion distortion

---

## TECHNICAL SPECIFICATIONS

### Processing Values
```
Shadows:     +15 to +20% lift
Highlights:  -10 to -15% pull
Contrast:    Medium (subtle S-curve)
Saturation:  +5 to +10% global
Clarity:     +5 to +10 (subtle)
Sharpening:  Light, edge-aware
```

### Color Correction
```
White Balance: 5200K
Tint: 0 (neutral)
Vibrance: +10
Saturation: +5
```

### Tone Curve (S-Curve)
```
Blacks:     5 (lifted slightly)
Shadows:    +15
Midtones:   0
Highlights: -10
Whites:     250 (not clipped)
```

---

## BLENDING RECIPE

### Input
```
BRACKET 1 (-2 EV): Underexposed - captures lights, windows, ceiling detail
BRACKET 2 ( 0 EV): Middle - base exposure
BRACKET 3 (+2 EV): Overexposed - shadow detail, under furniture
```

### Merge Process
```
1. ALIGN all frames (feature matching)
2. MERGE using Mertens exposure fusion
3. MASK windows with underexposed frame
4. BLEND shadows from overexposed frame
5. CORRECT white balance globally
6. APPLY subtle S-curve contrast
7. BOOST saturation slightly (+5-10%)
8. SHARPEN edges conservatively
```

### Layer Stack (Photoshop equivalent)
```
TOP:    Curves adjustment (S-curve)
        Color Balance (neutral)
        Vibrance (+10)
├──     Window mask (underexposed blend)
├──     Shadow mask (overexposed blend)
BASE:   Mertens fusion result
```

---

## QUALITY CHECKLIST

- [ ] Whites are white (no color cast)
- [ ] Shadows have detail (not crushed)
- [ ] Highlights controlled (no blowout)
- [ ] Windows show exterior
- [ ] Verticals are straight
- [ ] Colors accurate and vibrant
- [ ] No halos or artifacts
- [ ] Natural, not over-processed

---

## ANTI-PATTERNS (What NOT to do)

| Bad | Good |
|-----|------|
| Over-saturated colors | Natural vibrance |
| Crushed blacks | Lifted shadows with detail |
| Blown highlights | Controlled with detail |
| HDR halos | Clean edges |
| Unnatural contrast | Subtle S-curve |
| Color casts | Neutral white balance |
| Over-sharpened | Subtle edge sharpening |

---

*Reference for AutoHDR Clone pipeline development*
