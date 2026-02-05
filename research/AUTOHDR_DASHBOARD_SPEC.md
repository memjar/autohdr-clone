# AutoHDR Dashboard Specification
## Complete Feature Breakdown for Replication

**Date:** February 5, 2026
**Source:** Direct analysis of autohdr.com dashboard

---

## NAVIGATION STRUCTURE

```
TOP NAV:
├── Pricing          → /pricing
├── Studio           → /playground (photo editing)
├── Listings         → /listings (project management)
├── Contact          → Support
├── [Upload Button]  → Quick upload
└── [Profile Menu]   → Account options, Sign Out
```

---

## 1. LISTINGS PAGE (`/listings`)

**Purpose:** Project management hub - organize shoots by property

### Features
| Feature | Description |
|---------|-------------|
| Search | Find projects by name |
| Listing Cards | Thumbnail previews per project |
| Bulk Actions | Select All, Download, Rate |
| Share Link | Copy shareable project URL |
| Rate | Leave feedback on shoots |
| Bulk Download | Export edited photos |

### Data Model
```typescript
interface Listing {
  id: string
  name: string           // "123 Main St"
  thumbnails: string[]   // Preview images
  photoCount: number
  createdAt: Date
  status: 'processing' | 'complete' | 'partial'
}
```

---

## 2. ACCOUNT PAGE (`/account`)

### A. Style Settings (Photo Editing Preferences)

#### Scene Type Toggle
- Interior mode
- Exterior mode

#### Adjustment Sliders (-2 to +2, default 0)
| Slider | Purpose |
|--------|---------|
| Brightness | Overall luminance |
| Contrast | Tonal range |
| Vibrance | Color intensity |
| White Balance | Color temperature |

#### Styling Options
| Option | Values | Default |
|--------|--------|---------|
| Window Pull Intensity | Natural / Medium / Strong | Natural |
| Cloud Style | Fluffy + Whispy / Dramatic / Clear | Fluffy + Whispy |
| Twilight Style | Pink / Blue / Orange | Pink |
| TV Screen Replacement | None / Custom upload | None |
| Fire in Fireplace | Toggle | OFF |
| Declutter | Toggle | OFF |
| Interior Clouds | Toggle | ON |
| Exterior Clouds | Toggle | ON |
| Deduplication | Toggle | OFF |
| Walkthrough Reorder | Toggle + config | OFF |
| Retain Original Sky | Toggle | OFF |
| Perspective Correction | Toggle | ON |
| Grass Replacement | Toggle | OFF |
| Sign Removal | Toggle | OFF |
| Custom Pin | Customizable | - |
| Filenames | Default / Custom | Default |

#### Actions
- Save Changes
- Reset to Default

### B. Affiliate Program
```
- 10% recurring commission on referrals
- 50% on user referrals
- Referral code gives 25% discount
- Apply Now button
- Join Instagram button
- Copy Referral Code button
```

### C. Human-in-the-Loop
```
- Add human editors for special requests
- Editor queue management
- Add editor by email
```

### D. Balance & Subscription
```
- Current credits display
- Subscription status
- Invoices button
- Credit Log button
- Pay-as-you-go option
```

### E. Dropbox Automation
```
Workflow:
1. Create "AutoHDR" folder
2. Create listing subfolder (e.g., "123 Main St")
3. Create "01-RAW-Photos" subfolder
4. Upload photos → Auto-processed → Final folder

Status: Connected / Not connected
Actions: Connect with Dropbox, Tutorial
```

### F. Team Management
```
- Primary email with notifications toggle
- Add additional team member emails
```

### G. Company Selection
```
- Select which account to use for uploads
- Personal vs Company account
```

---

## 3. STUDIO/PLAYGROUND (`/playground`)

**Purpose:** AI photo editing interface

### Left Sidebar Tools
| Tool | Function |
|------|----------|
| Twilight | Day-to-dusk conversion |
| Auto Remove | Object removal |
| Grass | Grass enhancement/replacement |
| Staging | Virtual furniture staging |

### Upload Options
- Drag and drop
- Browse Files
- Select from Dropbox
- Select from Google Drive

---

## 4. PRICING PAGE (`/pricing`)

### Plans

| Plan | Photos/Month | Price | Per Photo |
|------|-------------|-------|-----------|
| Free | 10 | $0 | - |
| Standard | 500 | $265/mo | $0.53 |
| Enterprise | 5,000 | $2,250/mo | $0.45 |

### Yearly Discount: 20% off

### Enterprise Extras
- Auto TV Blackout
- Auto Add Fire
- Walkthrough Reordering
- Dedicated Slack Channel

### Pay-as-you-go
- Top Up button
- Charged at current per-photo rate

---

## 5. AFFILIATE/INCOME PAGE (`/income`)

```
- 10% recurring commission
- Performance tracking
- Monthly payouts to bank
- Social media sharing tools
```

---

## DATA MODELS

### User Account
```typescript
interface UserAccount {
  id: string
  email: string
  credits: number
  subscription: 'free' | 'standard' | 'enterprise' | null
  teamEmbers: string[]
  dropboxConnected: boolean
  settings: StyleSettings
  affiliateCode: string
  company?: Company
}
```

### Style Settings
```typescript
interface StyleSettings {
  sceneType: 'interior' | 'exterior'
  brightness: number      // -2 to +2
  contrast: number        // -2 to +2
  vibrance: number        // -2 to +2
  whiteBalance: number    // -2 to +2
  windowPullIntensity: 'natural' | 'medium' | 'strong'
  cloudStyle: 'fluffy' | 'dramatic' | 'clear'
  twilightStyle: 'pink' | 'blue' | 'orange'
  tvReplacement: string | null
  fireInFireplace: boolean
  declutter: boolean
  interiorClouds: boolean
  exteriorClouds: boolean
  deduplication: boolean
  walkthroughReorder: boolean
  retainOriginalSky: boolean
  perspectiveCorrection: boolean
  grassReplacement: boolean
  signRemoval: boolean
  customPin: string | null
  filenameFormat: 'default' | 'custom'
}
```

### Listing
```typescript
interface Listing {
  id: string
  userId: string
  name: string
  address?: string
  photos: Photo[]
  createdAt: Date
  updatedAt: Date
  shareableLink: string
  status: 'pending' | 'processing' | 'complete'
}

interface Photo {
  id: string
  listingId: string
  originalUrl: string
  processedUrl?: string
  status: 'uploaded' | 'processing' | 'complete' | 'failed'
  settings: Partial<StyleSettings>
  createdAt: Date
}
```

---

## IMPLEMENTATION PRIORITY

### Phase 1: MVP
1. Upload → Process → Download flow ✅
2. Basic HDR enhancement ✅
3. Listings page (project organization)
4. Account/settings page

### Phase 2: Core Features
5. Style settings (sliders, toggles)
6. Window pull intensity control
7. Perspective correction
8. Twilight conversion

### Phase 3: Advanced
9. Dropbox integration
10. Team management
11. Human-in-the-loop queue
12. Virtual staging

### Phase 4: Monetization
13. Pricing/subscription system
14. Credit system
15. Affiliate program

---

## API ENDPOINTS NEEDED

```
POST   /api/upload              → Upload photos
POST   /api/process             → Process photos
GET    /api/listings            → Get user listings
POST   /api/listings            → Create listing
GET    /api/listings/:id        → Get listing details
DELETE /api/listings/:id        → Delete listing
GET    /api/account             → Get account info
PATCH  /api/account/settings    → Update settings
POST   /api/dropbox/connect     → OAuth connect
POST   /api/team/invite         → Invite team member
GET    /api/credits             → Get credit balance
POST   /api/credits/topup       → Purchase credits
```

---

*Specification for AutoHDR Clone development*
