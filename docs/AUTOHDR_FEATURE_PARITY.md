# AutoHDR.com Feature Parity Roadmap

## Current Status (HDRit v4.7.0)
- [x] RAW file support (ARW, CR2, NEF, DNG, etc.)
- [x] Multi-exposure HDR blending (Mertens fusion)
- [x] Bracket alignment (ECC)
- [x] Window detail recovery
- [x] Basic tone mapping
- [x] Color correction
- [x] Noise reduction
- [x] Sharpening

## Phase 1: Core HDR (Current)
- [x] RAW decoding via rawpy/LibRaw
- [x] Exposure fusion (Mertens algorithm)
- [x] Basic tone mapping
- [ ] Natural/Intense presets
- [ ] Shadow/highlight recovery sliders

## Phase 2: AI Segmentation
- [ ] Sky detection & replacement
- [ ] Grass detection & enhancement
- [ ] Window detection for selective processing
- [ ] Object detection (YOLOv8)

## Phase 3: Advanced Editing
- [ ] AI object removal (SAM + LaMa inpainting)
- [ ] Virtual twilight/golden hour
- [ ] Fire in fireplace generation
- [ ] TV screen content insertion
- [ ] Blur for sensitive info

## Phase 4: Generative AI
- [ ] AI re-edit (text-to-image adjustments)
- [ ] Generative fill
- [ ] Virtual staging
- [ ] Lot lines drawing

## Phase 5: Enterprise Features
- [ ] API for integrations
- [ ] Webhook system
- [ ] Batch processing from cloud storage
- [ ] Multi-format delivery (print/web optimized)

## Technical Stack Needed
- GPU acceleration (CUDA/Metal)
- Deep learning models:
  - Segment Anything Model (SAM)
  - YOLOv8 for object detection
  - LaMa for inpainting
  - Stable Diffusion for generative features
- Computer vision (OpenCV) for alignment

## Key Differentiators to Match
1. Process 90+ photos/day/workflow
2. Results indistinguishable from professional editing
3. Real estate industry focus
4. 10% of US listings use AutoHDR (market leader)
