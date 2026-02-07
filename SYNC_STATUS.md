# HDRit Sync Status

## Current State

**Frontend (Laptop):** v2.3.0 ✅
- Perfect Edit mode with AI toggles
- Single image support all modes
- Pro Engine auto-detection
- Ready for backend connection

**Backend (Mac Studio):** Pending
- [ ] `mode=enhance` endpoint
- [ ] Window pull processing
- [ ] Sky enhancement
- [ ] Perspective correction
- [ ] Noise reduction
- [ ] Sharpening
- [ ] ngrok tunnel running

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
| Backend (ngrok) | `TBD` | ⏳ Pending |

---

## Last Updated
- **Frontend:** 2026-02-06 - Perfect Edit v2.3.0 deployed
- **Backend:** Awaiting update from Mac Studio
