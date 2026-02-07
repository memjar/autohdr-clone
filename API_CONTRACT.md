# HDRit API Contract

## Endpoints

### POST /process
Process images with specified mode.

**Query Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| mode | string | Yes | `hdr`, `twilight`, or `enhance` |
| brightness | float | No | -2.0 to 2.0, default 0 |
| contrast | float | No | -2.0 to 2.0, default 0 |
| vibrance | float | No | -2.0 to 2.0, default 0 |
| white_balance | float | No | -2.0 to 2.0, default 0 |

**Perfect Edit Mode (`mode=enhance`) Additional Params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| window_pull | bool | true | Balance window exposure |
| sky_enhance | bool | true | Boost blue sky |
| perspective_correct | bool | true | Straighten verticals |
| noise_reduction | bool | true | Reduce grain |
| sharpening | bool | true | Sharpen details |

**Request Body:**
- `multipart/form-data`
- Field: `images` (one or more image files)

**Response:**
- `200 OK`: `image/jpeg` blob
- `400 Bad Request`: `{ "error": "message" }`
- `500 Server Error`: `{ "error": "message" }`

---

### GET /health
Check backend status.

**Response:**
```json
{
  "status": "healthy",
  "pro_processor_available": true,
  "components": {
    "pro_processor": {
      "version": "4.7.0",
      "features": ["window_pull", "sky_enhance", "perspective", "denoise", "sharpen"]
    }
  }
}
```

---

## Mode Behaviors

### HDR Mode (`mode=hdr`)
- **1 image**: Apply single-exposure HDR enhancement
- **2+ images**: Merge as exposure brackets

### Twilight Mode (`mode=twilight`)
- **1 image**: Day-to-dusk conversion

### Enhance Mode (`mode=enhance`)
- **1 image**: Full AI enhancement with selected options
- **2+ images**: Batch process each image

---

## Example Requests

```bash
# Single image enhance
curl -X POST "http://localhost:8000/process?mode=enhance&window_pull=true&sky_enhance=true" \
  -F "images=@photo.jpg" \
  --output result.jpg

# HDR merge
curl -X POST "http://localhost:8000/process?mode=hdr" \
  -F "images=@bracket1.jpg" \
  -F "images=@bracket2.jpg" \
  -F "images=@bracket3.jpg" \
  --output result.jpg
```
