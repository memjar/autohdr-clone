# HDRit Sync Status

## Current State

**Frontend (Laptop):** v2.3.0 ✅
- Perfect Edit mode with AI toggles
- Single image support all modes
- Pro Engine auto-detection
- Ready for backend connection

**Backend (Mac Studio):** ✅ Running
- [x] `mode=enhance` endpoint
- [x] Window pull processing
- [x] Sky enhancement
- [x] Perspective correction
- [x] Noise reduction
- [x] Sharpening
- [x] Cloudflare tunnel running

---

## Communication Protocol

### Frontend → Backend (via git)
- Push UI changes
- Update API_CONTRACT.md if new params needed
- Update this SYNC_STATUS.md

### Backend → Frontend (via git)
- Push API changes
- Update SYNC_STATUS.md with endpoint status
- Share ngrok URL in this file when ready

---

## Active URLs

| Service | URL | Status |
|---------|-----|--------|
| Frontend (Vercel) | https://hdr.it.com | ✅ Live |
| Backend (ngrok) | https://hdr.it.com.ngrok.pro | ✅ Permanent |

---

## Last Updated
- **Frontend:** 2026-02-06 - Perfect Edit v2.3.0 deployed
- **Backend:** 2026-02-06 - Pro Processor v4.7.0 + enhance mode
