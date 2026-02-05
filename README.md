# AutoHDR Clone

> Open-source AI real estate photo editing - HDR blending, sky replacement, day-to-dusk, object removal

## What This Is

An open-source recreation of [AutoHDR.com](https://autohdr.com) - the AI-powered real estate photo editing service. Built with PyTorch, Stable Diffusion, and modern computer vision.

## Features

### Core (Phase 1-2)
- [x] HDR bracket merging (3-9 exposures)
- [x] Automatic exposure alignment
- [x] Tone mapping (Reinhard, Drago, Neural)
- [x] Color correction & white balance
- [ ] Perspective correction

### Enhancement (Phase 3-4)
- [ ] Sky replacement (overcast → blue/dramatic)
- [ ] Day-to-dusk conversion (twilight effect)
- [ ] Window illumination (warm interior glow)
- [ ] Grass enhancement

### Advanced (Phase 5+)
- [ ] Object removal (LaMa inpainting)
- [ ] Virtual staging (ControlNet)
- [ ] Fire/flame effects
- [ ] TV content insertion

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/autohdr-clone.git
cd autohdr-clone

# Install dependencies
pip install -r requirements.txt

# Run API server
python -m api.main

# Or process a single image
python -m src.core.hdr_merge --input ./brackets/ --output ./output.jpg
```

## API Usage

```python
import requests

# HDR Merge
files = [
    ('images', open('bracket_1.jpg', 'rb')),
    ('images', open('bracket_2.jpg', 'rb')),
    ('images', open('bracket_3.jpg', 'rb')),
]
response = requests.post('http://localhost:8000/hdr/merge', files=files)
result = response.json()  # {'url': 'https://...'}

# Day to Dusk
response = requests.post(
    'http://localhost:8000/effects/day-to-dusk',
    files={'image': open('daytime.jpg', 'rb')}
)

# Object Removal
response = requests.post(
    'http://localhost:8000/edit/remove',
    files={'image': open('photo.jpg', 'rb')},
    data={'prompt': 'remove the car in the driveway'}
)
```

## Tech Stack

| Component | Tool |
|-----------|------|
| HDR Fusion | OpenCV + DeepHDR |
| Segmentation | Segment Anything (SAM) |
| Sky Replace | SkyAR + custom harmonization |
| Inpainting | LaMa / IOPaint |
| Virtual Staging | ControlNet + SD |
| API | FastAPI |
| Queue | Redis + Celery |
| Frontend | Next.js |

## Project Structure

```
autohdr-clone/
├── research/              # Deep dive research docs
│   └── AUTOHDR_DEEP_DIVE.md
├── src/
│   ├── core/             # Core image processing
│   │   ├── hdr_merge.py
│   │   ├── tone_mapping.py
│   │   ├── color_correct.py
│   │   └── align.py
│   ├── models/           # AI models
│   │   ├── hdr_net.py
│   │   ├── sky_segment.py
│   │   ├── twilight.py
│   │   └── inpaint.py
│   └── api/              # FastAPI backend
│       ├── main.py
│       ├── routes/
│       └── workers/
├── web/                  # Next.js frontend
├── tests/                # Test suite
├── docs/                 # API documentation
├── requirements.txt
└── README.md
```

## Research

See [research/AUTOHDR_DEEP_DIVE.md](research/AUTOHDR_DEEP_DIVE.md) for:
- Complete feature breakdown
- Architecture analysis
- Open source building blocks
- Algorithm explanations
- Implementation roadmap
- Cost analysis

## Key Open Source Dependencies

| Project | Purpose | Link |
|---------|---------|------|
| DeepHDR | Neural HDR fusion | [GitHub](https://github.com/Galaxies99/DeepHDR-pytorch) |
| hdrcnn | Single-image HDR | [GitHub](https://github.com/gabrieleilertsen/hdrcnn) |
| SkyAR | Sky replacement | [GitHub](https://github.com/jiupinjia/SkyAR) |
| LaMa | Object removal | [GitHub](https://github.com/advimman/lama) |
| IOPaint | Inpainting UI | [GitHub](https://github.com/Sanster/IOPaint) |
| SAM | Segmentation | [GitHub](https://github.com/facebookresearch/segment-anything) |
| Grounded-SAM | Text→Mask | [GitHub](https://github.com/IDEA-Research/Grounded-Segment-Anything) |

## Roadmap

- **Week 1-2:** Core HDR merge + color correction
- **Week 3-4:** Neural HDR + sky replacement
- **Week 5-6:** Day-to-dusk pipeline
- **Week 7-8:** Object removal (LaMa)
- **Week 9-10:** Web UI + API + deployment

## Self-Hosting vs Cloud

### Self-Host (Privacy + Control)
```bash
# Run locally with GPU
docker-compose up -d
```

### Cloud Deploy (Scale)
```bash
# Deploy to RunPod/Lambda
./deploy.sh --provider runpod
```

## Cost Comparison

| Solution | Cost/Image | Speed |
|----------|------------|-------|
| AutoHDR | $0.37-$2.00 | 30 min |
| This (Cloud) | ~$0.01 | 30 sec |
| This (Self) | ~$0.003 | 30 sec |

## License

MIT - Use it however you want.

## Contributing

PRs welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Credits

Research and architecture by Forge + James.

---

*Built with open source. Built for photographers.*
