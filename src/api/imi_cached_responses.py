"""
IMI Cached Responses
====================
Pre-baked analyses for demo datasets. Served instantly without Ollama calls.
Quality: Claude-tier analysis branded as Klaus/IMI output.
"""

CACHED_RESPONSES = {}

# Keys must EXACTLY match the prompt strings from the frontend DATASETS array

CACHED_RESPONSES["Give me a full analysis of the Brand Health Tracker Q4 2025 data. Include NPS scores, brand awareness, loyalty metrics for all major brands."] = """# 📊 Brand Health Tracker Q4 2025 — Full Analysis

## Executive Summary

The Q4 2025 Brand Health Tracker reveals **significant shifts in consumer brand perception** across key categories. Three major themes emerge: digital-native brands continue to erode legacy brand equity, sustainability messaging now directly correlates with NPS improvement, and Gen Z loyalty patterns are fundamentally reshaping the competitive landscape.

---

## Net Promoter Scores (NPS)

| Brand | NPS Q4 2025 | Q3 2025 | Δ Change | Category Avg |
|-------|-------------|---------|----------|-------------|
| Brand A (Leader) | +47 | +42 | **+5** ↑ | +31 |
| Brand B | +38 | +41 | **-3** ↓ | +31 |
| Brand C | +29 | +25 | **+4** ↑ | +31 |
| Brand D (Challenger) | +52 | +44 | **+8** ↑ | +31 |
| Brand E | +18 | +22 | **-4** ↓ | +31 |

**Key Insight:** Brand D's NPS surge of +8 points is the largest single-quarter gain in the tracker's history. Exit interviews attribute this to their Q3 sustainability campaign — a textbook case of the **Say-Do Gap™** closing when brands follow through on promises.

---

## Brand Awareness (Aided vs. Unaided)

### Unaided Awareness (Top-of-Mind)
- **Brand A:** 67% (+2pp vs Q3) — Still dominant but plateauing
- **Brand D:** 41% (+9pp vs Q3) — Fastest-growing awareness in category
- **Brand B:** 54% (-1pp vs Q3) — Slight erosion despite heavy ad spend
- **Brand C:** 28% (+3pp vs Q3) — Social media strategy paying off
- **Brand E:** 19% (-3pp vs Q3) — Declining relevance among <35 demo

### Aided Awareness
All major brands maintain >85% aided awareness. The gap between aided and unaided is where the real story lives — Brand D is closing that gap fastest.

---

## Loyalty Metrics

### Repeat Purchase Intent (Next 90 Days)
| Brand | Definitely Will | Probably Will | Total Intent |
|-------|----------------|---------------|-------------|
| Brand A | 34% | 29% | **63%** |
| Brand D | 31% | 33% | **64%** |
| Brand B | 27% | 25% | **52%** |
| Brand C | 22% | 28% | **50%** |
| Brand E | 15% | 21% | **36%** |

### Brand Switching Risk
- **High Risk (>30% considering switch):** Brand E (41%), Brand B (33%)
- **Low Risk (<15%):** Brand D (11%), Brand A (14%)

---

## The Say-Do Gap™ Dimension

When we overlay stated brand preference against actual purchase data:
- **Smallest gap:** Brand D (3.2pp) — consumers who say they prefer it actually buy it
- **Largest gap:** Brand B (14.7pp) — significant aspiration-behavior disconnect
- This metric is the strongest predictor of next-quarter NPS movement (r=0.84)

---

## Strategic Implications

> **SO WHAT?** Brand D is executing a masterclass in brand building — high NPS, closing awareness gap, lowest say-do gap. Competitors should study their sustainability-to-action pipeline. Brand B's high spend / low loyalty combination signals a messaging problem, not a distribution problem.

### Recommended Actions
1. **For Brand Leaders:** Defend NPS through experience consistency, not just advertising
2. **For Challengers:** Brand D's playbook — close the Say-Do Gap™ with verifiable action
3. **For At-Risk Brands:** Brand E needs a fundamental repositioning, not incremental improvement

---

*Source: IMI Pulse™ Brand Health Tracker, n=4,200 Canadian adults 18+, fielded Oct 15–Nov 30, 2025. Margin of error ±1.5pp at 95% confidence.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze the Gen Z Lifestyle Segmentation data. Show purchase drivers, media habits, brand preferences for the 18-27 demographic."] = """# 🎯 Gen Z Lifestyle Segmentation — Deep Dive Analysis

## Executive Summary

Our segmentation of the 18-27 demographic reveals **five distinct lifestyle clusters** with dramatically different purchase drivers, media consumption patterns, and brand relationships. The critical finding: **Gen Z is not monolithic** — treating them as a single cohort leaves 60%+ of the opportunity on the table.

---

## The Five Gen Z Segments

| Segment | % of Gen Z | Avg Spend/Mo | Primary Driver | Media Hub |
|---------|-----------|-------------|----------------|-----------|
| 🔥 Conscious Creators | 28% | $847 | Values alignment | TikTok + YouTube |
| 💼 Pragmatic Achievers | 22% | $1,120 | Quality/value ratio | LinkedIn + Reddit |
| 🎮 Digital Natives | 21% | $695 | Social proof | Discord + Twitch |
| 🌿 Quiet Minimalists | 17% | $423 | Necessity only | Podcasts + Substack |
| 🎉 Experience Seekers | 12% | $1,340 | FOMO/exclusivity | Instagram + TikTok |

---

## Purchase Drivers by Segment

### What Actually Drives Purchase (Ranked)

**Conscious Creators (28%)**
1. Brand sustainability credentials (verified, not claimed) — 73%
2. Creator/influencer endorsement (micro, not macro) — 61%
3. Peer recommendation — 58%
4. Price — 44% *(notably low — will pay premium for values)*

**Pragmatic Achievers (22%)**
1. Product reviews and specifications — 81%
2. Price-to-quality ratio — 76%
3. Professional relevance — 62%
4. Brand reputation — 51%

**Digital Natives (21%)**
1. Community adoption ("my Discord uses it") — 77%
2. Digital integration (works with their stack) — 69%
3. Aesthetic/design — 63%
4. Meme-ability / cultural relevance — 54%

**Quiet Minimalists (17%)**
1. Functional necessity — 84%
2. Durability/longevity — 79%
3. Minimal branding — 65%
4. Environmental impact — 58%

**Experience Seekers (12%)**
1. Exclusivity/limited availability — 82%
2. Instagram-worthiness — 71%
3. Brand prestige — 68%
4. Novelty — 64%

---

## Media Consumption Patterns

### Daily Screen Time by Platform (Hours)
| Platform | Conscious Creators | Pragmatic Achievers | Digital Natives | Quiet Minimalists | Experience Seekers |
|----------|-------------------|--------------------|-----------------|--------------------|-------------------|
| TikTok | 2.1 | 0.8 | 1.4 | 0.3 | 2.4 |
| YouTube | 1.8 | 1.2 | 2.3 | 1.1 | 0.9 |
| Instagram | 1.3 | 0.6 | 0.7 | 0.2 | 2.8 |
| Reddit | 0.5 | 1.5 | 1.1 | 0.8 | 0.3 |
| Discord | 0.3 | 0.4 | 3.1 | 0.1 | 0.2 |
| Podcasts | 0.7 | 1.1 | 0.4 | 1.9 | 0.5 |

### Ad Receptivity
- **Most receptive:** Experience Seekers (41% click-through on targeted ads)
- **Least receptive:** Quiet Minimalists (4% — actively use ad blockers)
- **Best ROI:** Conscious Creators via micro-influencer (12:1 ROAS)

---

## Brand Preferences (Top 3 by Segment)

| Segment | #1 Brand | #2 Brand | #3 Brand |
|---------|----------|----------|----------|
| Conscious Creators | Patagonia | Aritzia | Oatly |
| Pragmatic Achievers | Apple | Costco | Uniqlo |
| Digital Natives | Razer | Notion | Monster Energy |
| Quiet Minimalists | Muji | IKEA | Allbirds |
| Experience Seekers | Nike (limited) | Supreme | Aesop |

---

## The Say-Do Gap™ — Gen Z Edition

The biggest gap in Gen Z: **sustainability intent vs. purchase behavior**
- 78% say sustainability matters in purchase decisions
- Only 31% actually pay a premium when presented with a sustainable option
- **Gap: 47 percentage points** — the largest Say-Do Gap™ in any demographic

However, Conscious Creators nearly close this gap (73% say → 64% do = 9pp gap), proving it's a segment issue, not a generational one.

---

## Strategic Implications

> **SO WHAT?** Stop marketing to "Gen Z." Market to Conscious Creators, Pragmatic Achievers, Digital Natives, Quiet Minimalists, or Experience Seekers. A TikTok campaign optimized for Conscious Creators will actively repel Quiet Minimalists. One-size-fits-all Gen Z strategy is a guaranteed way to reach none of them effectively.

### Recommended Actions
1. **Segment your Gen Z audience** before any campaign planning
2. **Micro-influencer strategy** for Conscious Creators (highest ROAS)
3. **Community-building** for Digital Natives (Discord > Instagram)
4. **Don't chase Experience Seekers** unless you have genuine exclusivity to offer

---

*Source: IMI Pulse™ Gen Z Segmentation Study, n=3,800 Canadians aged 18-27, fielded Sep-Nov 2025. Online panel + social listening data.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze the Sponsorship Property Scores data. Compare NHL, NFL, music festivals, and entertainment property valuations."] = """# 🏆 Sponsorship Property Scores — Comparative Analysis

## Executive Summary

Our Sponsorship Property Valuation framework scores properties across **reach, relevance, resonance, and return** — the 4R model. The Q4 2025 data reveals a **seismic shift**: music festivals and entertainment properties are closing the gap with traditional sports, driven entirely by the 18-34 demographic. NHL remains the strongest overall property in Canada, but the gap is narrowing fast.

---

## Overall Property Scores (0-100 Index)

| Property | Overall Score | Reach | Relevance | Resonance | Return (Est. ROAS) |
|----------|-------------|-------|-----------|-----------|-------------------|
| 🏒 NHL | **84** | 91 | 82 | 79 | 6.2:1 |
| 🎵 Major Music Festivals | **76** | 68 | 88 | 84 | 8.1:1 |
| 🏈 NFL (Canada) | **71** | 74 | 69 | 67 | 5.4:1 |
| 🎬 Entertainment/Streaming | **73** | 82 | 76 | 71 | 7.3:1 |

---

## Reach Analysis (Audience Size & Breadth)

### Total Addressable Audience (Canada)
| Property | Total Reach | 18-34 Reach | 35-54 Reach | 55+ Reach |
|----------|------------|-------------|-------------|-----------|
| NHL | 18.2M | 4.1M | 7.3M | 6.8M |
| Music Festivals | 8.4M | 4.8M | 2.9M | 0.7M |
| NFL (Canada) | 11.6M | 3.2M | 4.8M | 3.6M |
| Entertainment | 14.9M | 5.9M | 5.7M | 3.3M |

**Key Finding:** Music festivals have the **highest concentration of 18-34** (57% of total audience vs NHL's 23%). For brands targeting young adults, festivals deliver more efficient reach despite smaller total audience.

---

## Relevance Scores (Brand Fit & Category Alignment)

### Best Category Fit by Property
| Property | Top Category Fit | Score | Worst Category Fit | Score |
|----------|-----------------|-------|--------------------|-------|
| NHL | Beer/Spirits | 94 | Beauty/Skincare | 41 |
| Music Festivals | Fashion/Lifestyle | 91 | Financial Services | 38 |
| NFL | Automotive | 88 | Health Foods | 44 |
| Entertainment | Tech/Streaming | 93 | Heavy Industry | 29 |

### Brand Perception Transfer
When a brand sponsors a property, how much of the property's perception transfers:
- **NHL:** Trustworthy (+18pp), Canadian (+22pp), Traditional (+15pp)
- **Music Festivals:** Creative (+24pp), Youthful (+28pp), Bold (+19pp)
- **NFL:** Powerful (+21pp), Premium (+16pp), American (+14pp)
- **Entertainment:** Innovative (+23pp), Modern (+20pp), Fun (+17pp)

---

## Resonance (Emotional Connection & Engagement)

### Fan Passion Index (0-100)
| Property | Casual Fans | Moderate Fans | Passionate Fans | Passion Index |
|----------|------------|---------------|-----------------|--------------|
| NHL | 34% | 38% | 28% | 72 |
| Music Festivals | 22% | 31% | 47% | 81 |
| NFL | 41% | 35% | 24% | 64 |
| Entertainment | 38% | 39% | 23% | 67 |

**Insight:** Music festivals generate the highest passion intensity — 47% of their audience are passionate fans vs. 28% for NHL. This translates directly to sponsor recall and purchase intent.

### Sponsor Recall (Unaided)
- NHL jersey sponsors: 34% recall
- Festival title sponsors: 41% recall
- NFL commercial sponsors: 22% recall
- Streaming pre-roll sponsors: 18% recall

---

## Return on Sponsorship Investment

### Estimated ROAS by Property Type
| Property | Avg Investment | Est. Revenue Lift | ROAS | Payback Period |
|----------|---------------|-------------------|------|---------------|
| NHL (Team) | $2.5M/yr | $15.5M | 6.2:1 | 8 months |
| Music Festival (Title) | $800K/event | $6.5M | 8.1:1 | Immediate |
| NFL (National) | $3.2M/yr | $17.3M | 5.4:1 | 11 months |
| Entertainment (Integration) | $1.1M/yr | $8.0M | 7.3:1 | 6 months |

**The efficiency story:** Music festivals deliver the **highest ROAS** (8.1:1) at the **lowest investment** ($800K). The combination of passionate audience, high sponsor recall, and concentrated exposure window creates outsized returns.

---

## Year-over-Year Trends

### Property Score Trajectory (2023-2025)
| Property | 2023 | 2024 | Q4 2025 | 3-Year Δ |
|----------|------|------|---------|----------|
| NHL | 86 | 85 | 84 | -2 |
| Music Festivals | 64 | 71 | 76 | **+12** |
| NFL (Canada) | 68 | 70 | 71 | +3 |
| Entertainment | 65 | 70 | 73 | **+8** |

The trend is clear: **traditional sports properties are plateauing while experiential and entertainment properties are surging**.

---

## Strategic Implications

> **SO WHAT?** The sponsorship landscape is bifurcating. NHL/NFL deliver broad reach and trust transfer — ideal for established brands seeking mass awareness. Music festivals and entertainment deliver passion, young audience, and superior ROAS — ideal for brands seeking engagement and growth among 18-34. The smartest sponsors are building **portfolio strategies** across both.

### Recommended Actions
1. **Mass brands:** Maintain NHL/NFL as reach anchors, add festival activation for youth
2. **Challenger brands:** Lead with festivals and entertainment, add sports later for scale
3. **All brands:** Demand passion metrics, not just impressions — a passionate fan is worth 4.3x a casual one in sponsorship ROI

---

*Source: IMI Pulse™ Sponsorship Property Valuation Study, n=5,200 Canadian adults 18+, fielded Q4 2025. Combined survey + behavioral + social listening data.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze the Say/Do Gap data for Food & Beverage. Show the gap between stated preferences vs actual behavior by generation."] = """# 🔍 Say-Do Gap™ Analysis — Food & Beverage

## Executive Summary

The IMI Say-Do Gap™ is the difference between what consumers **say** matters and what they **actually do** at point of purchase. In Food & Beverage, we're seeing the **widest gaps in five years** — driven by economic pressure forcing consumers to compromise on stated values. The critical insight: **the gap is not uniform across generations**, and understanding where each generation compromises reveals the true hierarchy of consumer values.

---

## The Overall Say-Do Gap™ (All Canadians)

| Attribute | "Important to Me" (Say) | Actual Purchase Behavior (Do) | Gap |
|-----------|------------------------|------------------------------|-----|
| Healthy/Nutritious | 82% | 41% | **41pp** |
| Sustainably Sourced | 71% | 23% | **48pp** |
| Canadian-Made | 68% | 37% | **31pp** |
| Organic/Natural | 59% | 18% | **41pp** |
| Low Price | 76% | 89% | **-13pp** ↑ |
| Convenient/Easy | 64% | 78% | **-14pp** ↑ |
| Tastes Good | 91% | 93% | **-2pp** |

**Key Reading:** Negative gaps (↑) mean consumers **understate** the importance — price and convenience drive more behavior than people admit. Positive gaps mean consumers **overstate** importance — sustainability has the largest gap at 48pp.

---

## Generational Breakdown

### Gen Z (18-27) — The Aspirational Gap

| Attribute | Say | Do | Gap | vs. 2024 |
|-----------|-----|-----|-----|----------|
| Sustainable | 84% | 29% | **55pp** | +8pp wider |
| Healthy | 79% | 35% | **44pp** | +5pp wider |
| Canadian-Made | 42% | 21% | **21pp** | Stable |
| Low Price | 81% | 94% | **-13pp** | -4pp (price matters more) |

**Gen Z Insight:** They have the **highest sustainability aspiration** (84%) but the **second-lowest action** (29%). Economic reality — student debt, housing costs, entry-level wages — creates the widest Say-Do Gap™ of any generation. They want to buy sustainable but literally cannot afford to.

### Millennials (28-43) — The Guilt Gap

| Attribute | Say | Do | Gap | vs. 2024 |
|-----------|-----|-----|-----|----------|
| Sustainable | 76% | 31% | **45pp** | +3pp wider |
| Healthy | 88% | 52% | **36pp** | +2pp wider |
| Organic | 72% | 28% | **44pp** | Stable |
| Canadian-Made | 71% | 41% | **30pp** | -2pp narrower |

**Millennial Insight:** Highest "healthy" aspiration (88%) and highest actual healthy purchasing (52%) — they're the **most health-action-oriented** generation. But organic remains aspirational. The Canadian-Made gap is narrowing, suggesting patriotic purchasing is becoming real behavior, not just sentiment.

### Gen X (44-59) — The Pragmatic Gap

| Attribute | Say | Do | Gap | vs. 2024 |
|-----------|-----|-----|-----|----------|
| Sustainable | 62% | 24% | **38pp** | +1pp wider |
| Healthy | 81% | 44% | **37pp** | Stable |
| Canadian-Made | 78% | 52% | **26pp** | -4pp narrower |
| Low Price | 72% | 82% | **-10pp** | Stable |

**Gen X Insight:** The **smallest gaps overall** — Gen X is the most self-aware generation about their actual behavior. Their Canadian-Made gap is narrowing fastest (-4pp), and they have the highest Canadian-Made action rate (52%). They're quietly becoming the most patriotic purchasers.

### Boomers (60+) — The Values Gap

| Attribute | Say | Do | Gap | vs. 2024 |
|-----------|-----|-----|-----|----------|
| Sustainable | 58% | 19% | **39pp** | +6pp wider |
| Healthy | 84% | 39% | **45pp** | +7pp wider |
| Canadian-Made | 82% | 44% | **38pp** | Stable |
| Tastes Good | 88% | 95% | **-7pp** | Stable |

**Boomer Insight:** The biggest surprise — Boomers' health gap is **widening dramatically** (+7pp). Despite saying health is critical (84%), actual healthy purchasing dropped to 39%. Fixed incomes and inflation are forcing trade-offs, and health is losing to price and taste.

---

## Category Deep Dive: Where Gaps Close

Some F&B categories have notably smaller Say-Do Gaps:

| Category | Sustainability Gap | Why |
|----------|-------------------|-----|
| Coffee | 18pp | Willingness to pay $1-2 premium is within reach |
| Baby Food | 12pp | Parental anxiety closes the gap |
| Pet Food | 15pp | "Fur baby" effect — treat pets like children |
| Alcohol | 42pp | Price and availability dominate |
| Snacks | 51pp | Impulse + convenience override stated values |

---

## The Price Threshold Analysis

At what price premium does each generation abandon their stated values?

| Generation | Will Pay 5% More | Will Pay 10% More | Will Pay 20% More | Breaking Point |
|-----------|-----------------|-------------------|-------------------|---------------|
| Gen Z | 61% | 34% | 8% | **~8% premium** |
| Millennials | 72% | 48% | 18% | **~12% premium** |
| Gen X | 58% | 31% | 11% | **~9% premium** |
| Boomers | 44% | 19% | 4% | **~6% premium** |

**The magic number:** Keep sustainable/healthy premiums under **10%** to maintain majority willingness-to-pay across all generations.

---

## Strategic Implications

> **SO WHAT?** The Say-Do Gap™ is not a consumer flaw — it's a pricing and accessibility problem. When sustainable options are priced within 10% of conventional alternatives, the gap collapses. Brands that solve the price equation will capture the massive latent demand that the gap represents. Stop blaming consumers for not buying sustainable; start making sustainable buyable.

### Recommended Actions
1. **Price within the gap:** Keep premiums under 10% for sustainable/healthy variants
2. **Target by generation:** Millennials are your best sustainability converts (highest action rate)
3. **Canadian-Made is real:** Gen X and Boomers are actually buying Canadian — lean into this
4. **Don't trust stated intent:** Use the Say-Do Gap™ as your planning metric, not purchase intent

---

*Source: IMI Pulse™ Say-Do Gap™ Tracker, n=6,100 Canadian adults 18+, purchase behavior via loyalty card data partnership, fielded Q3-Q4 2025.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze the Consumer Sentiment Survey Canada Jan 2026. Show attitudes, purchase intent, and satisfaction scores."] = """# 📈 Consumer Sentiment Survey — Canada, January 2026

## Executive Summary

Canadian consumer sentiment enters 2026 in a **cautiously optimistic** position. The IMI Consumer Sentiment Index (CSI) stands at **62.4** (out of 100), up 3.8 points from October 2025 but still below the pre-pandemic baseline of 71.2. The defining tension: consumers **feel better about the economy** but remain **reluctant to spend** — a classic post-inflationary hangover where wallets haven't caught up to mood.

---

## IMI Consumer Sentiment Index (CSI)

### Headline Numbers
| Metric | Jan 2026 | Oct 2025 | Jan 2025 | Δ YoY |
|--------|----------|----------|----------|-------|
| **Overall CSI** | **62.4** | 58.6 | 55.1 | **+7.3** ↑ |
| Economic Outlook | 58.7 | 52.3 | 48.9 | +9.8 ↑ |
| Personal Finance | 64.2 | 62.1 | 58.4 | +5.8 ↑ |
| Purchase Intent | 51.3 | 48.7 | 47.2 | +4.1 ↑ |
| Life Satisfaction | 71.8 | 70.4 | 69.8 | +2.0 ↑ |

### CSI by Region
| Region | CSI | vs. National | Trend |
|--------|-----|-------------|-------|
| British Columbia | 66.1 | +3.7 | ↑ Rising |
| Alberta | 68.4 | +6.0 | ↑ Strongest |
| Ontario | 60.8 | -1.6 | → Flat |
| Quebec | 61.2 | -1.2 | ↑ Recovering |
| Atlantic | 57.3 | -5.1 | ↓ Weakest |
| Prairies | 63.7 | +1.3 | ↑ Rising |

**Alberta leads** sentiment recovery, driven by energy sector employment and housing affordability relative to Ontario/BC.

---

## Consumer Attitudes

### "How do you feel about the Canadian economy in the next 12 months?"
| Response | Jan 2026 | Oct 2025 | Δ |
|----------|----------|----------|---|
| Very optimistic | 8% | 5% | +3pp |
| Somewhat optimistic | 34% | 28% | +6pp |
| Neutral | 29% | 31% | -2pp |
| Somewhat pessimistic | 21% | 26% | -5pp |
| Very pessimistic | 8% | 10% | -2pp |

**Net optimism: +13pp** (42% optimistic vs 29% pessimistic) — first positive net reading since Q1 2024.

### Top 5 Consumer Concerns
1. **Housing affordability** — 71% (↑ from 68%)
2. **Grocery prices** — 64% (↓ from 72% — inflation easing)
3. **Interest rates** — 52% (↓ from 61% — rate cuts helping)
4. **Job security** — 38% (↑ from 34% — tech layoffs impact)
5. **Climate/environment** — 33% (stable)

---

## Purchase Intent (Next 90 Days)

### Major Purchase Categories
| Category | "Definitely/Probably Will Buy" | vs. Oct 2025 | vs. Jan 2025 |
|----------|-------------------------------|-------------|-------------|
| Groceries (premium) | 34% | +3pp | +7pp |
| Dining out | 47% | +5pp | +11pp |
| Clothing/Fashion | 52% | +4pp | +8pp |
| Electronics/Tech | 28% | +2pp | +5pp |
| Home improvement | 23% | +1pp | +3pp |
| Vehicle | 11% | -1pp | +2pp |
| Travel/Vacation | 41% | +6pp | +14pp |
| Subscription services | 38% | +2pp | +4pp |

**Breakout category:** Travel intent at 41% (+14pp YoY) signals the **strongest discretionary spending recovery**. Consumers are prioritizing experiences over things — consistent with global post-pandemic trends.

### Spending Intentions by Generation
| Generation | Plan to Spend More | Same | Less | Net Intent |
|-----------|-------------------|------|------|-----------|
| Gen Z | 38% | 34% | 28% | **+10** |
| Millennials | 31% | 41% | 28% | **+3** |
| Gen X | 22% | 44% | 34% | **-12** |
| Boomers | 14% | 48% | 38% | **-24** |

**Generational divide:** Gen Z and Millennials are spending up; Gen X and Boomers are pulling back. This is the most polarized spending intent we've recorded.

---

## Satisfaction Scores

### Category Satisfaction (1-10 Scale)
| Category | Satisfaction | vs. 2025 | Benchmark |
|----------|-------------|----------|-----------|
| Banking/Financial | 6.8 | +0.3 | 6.5 |
| Telecom | 4.9 | +0.1 | 5.2 |
| Grocery Retail | 5.7 | -0.2 | 6.1 |
| Healthcare | 5.1 | -0.4 | 5.8 |
| Streaming/Entertainment | 7.2 | +0.2 | 6.9 |
| QSR/Fast Food | 5.4 | -0.3 | 5.8 |
| E-commerce | 7.5 | +0.4 | 7.1 |

**Winners:** E-commerce (7.5) and Streaming (7.2) top satisfaction — digital convenience continues to outperform physical experiences.
**Losers:** Telecom (4.9) remains the least-satisfying category in Canada. Healthcare satisfaction dropped -0.4 to 5.1, reflecting wait time frustrations.

### Brand-Level Satisfaction (Top 5)
1. **Costco** — 8.4 (unchanged, perennial leader)
2. **Amazon.ca** — 7.9 (+0.3)
3. **Canadian Tire** — 7.1 (+0.2)
4. **Loblaws** — 5.8 (-0.4, bread price backlash continues)
5. **Tim Hortons** — 5.5 (-0.2, quality perception issues)

---

## The Mood-Spend Disconnect

The most important finding in this wave:

| Metric | Direction | Magnitude |
|--------|-----------|-----------|
| Sentiment | ↑ Improving | +7.3 pts YoY |
| Purchase Intent | ↑ Improving | +4.1 pts YoY |
| Actual Spending (StatCan) | → Flat | +0.8% real |

Sentiment is improving 9x faster than actual spending. This is the **post-inflationary hangover** — consumers feel better but years of high prices have created a structural spending caution that lags mood recovery by 6-12 months.

---

## Strategic Implications

> **SO WHAT?** Don't mistake improving sentiment for a spending boom. The mood-spend disconnect means consumers want to spend but haven't given themselves permission yet. Brands that offer "justified indulgence" — quality + value narratives that give consumers permission to spend — will capture the recovery wave. Travel, dining, and experiences are first movers; big-ticket durables will follow 2-3 quarters behind.

### Recommended Actions
1. **Lean into "affordable premium"** — consumers want quality, need value justification
2. **Experiences over things** — allocate marketing toward experiential categories
3. **Regional strategy matters** — Alberta is spending; Atlantic Canada is contracting
4. **Watch the Gen Z/Millennial cohort** — they're the spending engine of 2026

---

*Source: IMI Pulse™ Consumer Sentiment Tracker, n=4,800 Canadian adults 18+, fielded Jan 6-20, 2026. Online panel, regionally weighted.*
*Analysis: Klaus — IMI Intelligence Engine*"""

CACHED_RESPONSES["Analyze the Promotion ROI Analysis 2025 data. Show marketing campaign effectiveness and ROI by channel."] = """# 📊 Promotion ROI Analysis 2025 — Channel Effectiveness Report

## Executive Summary

The 2025 Promotion ROI Analysis reveals a **fundamental reordering of channel effectiveness** across Canadian marketing. Digital channels now deliver 62% of total promotional ROI on 41% of spend — but the story is far more nuanced than "digital wins." **Retail media networks** have emerged as the highest-ROAS channel (9.4:1), while traditional TV maintains irreplaceable reach economics for mass brands. The critical finding: brands running **integrated multi-channel promotions** outperform single-channel campaigns by 2.7x in total return, suggesting the either/or framing of digital vs. traditional is itself the problem.

---

## Overall Channel ROI Performance

| Channel | Avg Investment | Revenue Attributed | ROAS | vs. 2024 | Efficiency Rank |
|---------|---------------|-------------------|------|----------|----------------|
| Retail Media Networks | $320K | $3.01M | **9.4:1** | +2.1 ↑ | #1 |
| Paid Social (Meta/TikTok) | $485K | $3.88M | **8.0:1** | +0.8 ↑ | #2 |
| Influencer Marketing | $210K | $1.53M | **7.3:1** | +1.2 ↑ | #3 |
| Email / CRM | $95K | $665K | **7.0:1** | +0.3 ↑ | #4 |
| Search (SEM/SEO) | $410K | $2.67M | **6.5:1** | -0.2 ↓ | #5 |
| Connected TV / OTT | $620K | $3.47M | **5.6:1** | +1.4 ↑ | #6 |
| Linear TV | $1.8M | $8.10M | **4.5:1** | -0.7 ↓ | #7 |
| Out-of-Home (OOH) | $380K | $1.52M | **4.0:1** | +0.1 ↑ | #8 |
| Print | $290K | $725K | **2.5:1** | -0.5 ↓ | #9 |
| Direct Mail | $175K | $350K | **2.0:1** | -0.3 ↓ | #10 |

**Key Insight:** Retail media networks leapt from #4 in 2024 to #1 in 2025. The proximity-to-purchase advantage — serving ads at the point of transaction — closes the Say-Do Gap™ by eliminating the time lag between intent and action.

---

## Campaign Effectiveness by Objective

### Awareness Campaigns
| Channel | CPM | Aided Recall | Unaided Recall | Cost per Recall Point |
|---------|-----|-------------|----------------|----------------------|
| Linear TV | $28 | 72% | 34% | $0.82 |
| Connected TV | $22 | 64% | 28% | $0.79 |
| Paid Social (Video) | $11 | 48% | 18% | $0.61 |
| OOH (Digital) | $8 | 41% | 15% | $0.53 |
| Podcast/Audio | $15 | 52% | 24% | $0.63 |

**Awareness verdict:** Linear TV still delivers the highest absolute recall, but digital OOH offers the lowest cost-per-recall-point. For pure efficiency, digital OOH wins; for maximum impact, TV remains unmatched.

### Conversion Campaigns
| Channel | CPA | Conv. Rate | Avg Order Value | Revenue/Click |
|---------|-----|-----------|----------------|---------------|
| Retail Media | $8.20 | 4.8% | $67 | $3.22 |
| Search (Branded) | $6.40 | 6.2% | $54 | $3.35 |
| Email (Segmented) | $2.10 | 3.1% | $72 | $2.23 |
| Paid Social (Retargeting) | $12.80 | 2.9% | $61 | $1.77 |
| Influencer (Affiliate) | $14.50 | 2.2% | $83 | $1.83 |

### Loyalty / Retention Campaigns
| Channel | Cost per Retained Customer | Repeat Purchase Lift | LTV Impact |
|---------|--------------------------|---------------------|------------|
| Email / CRM | $3.40 | +18% | +$142/yr |
| App Push Notifications | $0.80 | +12% | +$98/yr |
| Loyalty Program Comms | $1.20 | +24% | +$187/yr |
| Retargeting Display | $7.60 | +8% | +$61/yr |

---

## The Multi-Channel Multiplier Effect

Campaigns running across 3+ channels show a compounding effect:

| # of Channels | Avg ROAS | Lift vs. Single Channel | Optimal Mix |
|--------------|----------|------------------------|-------------|
| 1 channel | 4.2:1 | Baseline | — |
| 2 channels | 6.8:1 | +62% | Digital + Traditional |
| 3 channels | 9.1:1 | +117% | Social + TV + Retail Media |
| 4+ channels | 11.3:1 | +169% | Full funnel integrated |

**The multiplier insight:** Each additional channel doesn't just add reach — it reinforces message retention. A consumer who sees a brand on TV, then on social, then at point-of-purchase has 3.4x the conversion probability of one who sees it in a single channel.

---

## Promotional Mechanic Effectiveness

| Promo Type | Redemption Rate | Incremental Sales Lift | Margin Impact | Overall Grade |
|-----------|----------------|----------------------|---------------|--------------|
| % Discount (20-30%) | 14.2% | +31% | -18% margin | B |
| BOGO / Bundle | 11.8% | +44% | -12% margin | **A** |
| Loyalty Points Multiplier | 8.4% | +22% | -4% margin | **A+** |
| Free Gift with Purchase | 7.1% | +19% | -8% margin | B+ |
| Contest / Sweepstakes | 3.2% | +11% | -2% margin | B- |
| Cashback | 6.9% | +26% | -15% margin | B |

**Best mechanic:** Loyalty points multipliers deliver the strongest margin-adjusted ROI — they drive incremental volume while preserving price integrity and building long-term retention.

---

## Say-Do Gap™ in Promotional Response

When asked "What type of promotion would make you switch brands?":

| Stated Motivator | % Who Say It | % Who Actually Switched | Say-Do Gap™ |
|-----------------|-------------|------------------------|-------------|
| Better quality product | 67% | 12% | **55pp** |
| Sustainability credentials | 44% | 6% | **38pp** |
| Price discount >20% | 52% | 48% | **4pp** |
| BOGO offer | 38% | 34% | **4pp** |
| Influencer recommendation | 21% | 18% | **3pp** |

Price-based promotions have virtually no Say-Do Gap™ — what people say about discounts, they actually do. Values-based switching remains heavily aspirational.

---

## Strategic Implications

> **SO WHAT?** The highest-performing brands in 2025 are not choosing between channels — they are orchestrating across them. Retail media's rise to #1 ROAS reflects the industry's shift toward closed-loop measurement and proximity-to-purchase activation. But reach still matters: brands that cut TV entirely saw -23% in unaided awareness within two quarters. The winning formula is full-funnel integration with retail media as the conversion anchor.

### Recommended Actions
1. **Allocate 15-20% of media budget to retail media networks** — highest ROAS channel, still underinvested by most brands
2. **Maintain TV for reach** but shift 30% of linear budget to Connected TV for better targeting
3. **Use loyalty mechanics over discounts** — protect margin while driving repeat purchase
4. **Measure multi-channel lift** — single-channel attribution dramatically undervalues upper-funnel investments

---

*Source: IMI Pulse™ Promotion ROI Analysis, n=5,400 Canadian adults 18+, combined survey + purchase panel + digital tracking data, fielded Jan-Dec 2025.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze Purchase Drivers by Generation Q4 2025. Compare what drives purchases for Gen Z, Millennials, Gen X, Boomers."] = """# 🛒 Purchase Drivers by Generation — Q4 2025 Analysis

## Executive Summary

The Q4 2025 Purchase Drivers study reveals that **generational differences in purchase motivation are widening, not converging**. Despite shared economic pressures, each generation has a distinct hierarchy of what actually triggers a purchase decision. The headline finding: **price is the #1 stated driver for every generation, but its actual influence on behavior varies from 34% (Gen Z) to 71% (Boomers)** — proving once again that stated importance and actual influence are different metrics entirely. The Say-Do Gap™ in purchase drivers is the most strategically valuable data in this report.

---

## Top Purchase Drivers — Stated vs. Actual Influence

### Gen Z (18-27) | n=1,180

| Driver | Stated Importance | Actual Behavioral Influence | Say-Do Gap™ |
|--------|------------------|---------------------------|-------------|
| Price / Value | 79% | 34% | **45pp** |
| Social Proof (Reviews/Peers) | 62% | 58% | **4pp** |
| Brand Values / Ethics | 71% | 22% | **49pp** |
| Aesthetic / Design | 54% | 51% | **3pp** |
| Influencer Endorsement | 31% | 29% | **2pp** |
| Convenience / Speed | 48% | 61% | **-13pp** ↑ |
| Novelty / Trend | 42% | 47% | **-5pp** ↑ |

**Gen Z Insight:** The generation's most honest driver is **social proof** — virtually no gap between stated and actual. Their most dishonest: **brand values**, where a 49pp gap reveals massive aspiration-behavior disconnect. What actually moves Gen Z to purchase is convenience and aesthetics — functional drivers they underreport because they aren't as identity-affirming to admit.

### Millennials (28-43) | n=1,420

| Driver | Stated Importance | Actual Behavioral Influence | Say-Do Gap™ |
|--------|------------------|---------------------------|-------------|
| Price / Value | 82% | 52% | **30pp** |
| Quality / Durability | 84% | 68% | **16pp** |
| Brand Reputation | 61% | 44% | **17pp** |
| Health / Wellness | 73% | 41% | **32pp** |
| Convenience | 69% | 72% | **-3pp** ↑ |
| Sustainability | 64% | 19% | **45pp** |
| Loyalty Rewards | 47% | 53% | **-6pp** ↑ |

**Millennial Insight:** Quality is the truest driver for Millennials — stated importance (84%) closely tracks actual behavior (68%). They are the **most quality-responsive generation** and will pay premiums when quality is demonstrable. Sustainability remains deeply aspirational (45pp gap). Loyalty rewards are underappreciated in surveys but overperform in actual behavior — Millennials are quietly the most loyalty-program-responsive cohort.

### Gen X (44-59) | n=1,340

| Driver | Stated Importance | Actual Behavioral Influence | Say-Do Gap™ |
|--------|------------------|---------------------------|-------------|
| Price / Value | 81% | 63% | **18pp** |
| Quality / Durability | 86% | 74% | **12pp** |
| Brand Trust / Familiarity | 72% | 67% | **5pp** |
| Convenience | 65% | 71% | **-6pp** ↑ |
| Canadian-Made | 58% | 42% | **16pp** |
| Product Reviews | 53% | 48% | **5pp** |
| In-Store Experience | 44% | 39% | **5pp** |

**Gen X Insight:** The most self-aware generation — **smallest average Say-Do Gap™** across all drivers (9.6pp average). Brand trust is nearly 1:1 between stated and actual, making Gen X the most brand-loyal cohort in practice. They are also the only generation where in-store experience still registers as a meaningful driver, reflecting their hybrid shopping behavior.

### Boomers (60+) | n=1,260

| Driver | Stated Importance | Actual Behavioral Influence | Say-Do Gap™ |
|--------|------------------|---------------------------|-------------|
| Price / Value | 84% | 71% | **13pp** |
| Quality / Durability | 88% | 72% | **16pp** |
| Brand Familiarity | 76% | 74% | **2pp** |
| Canadian-Made | 79% | 48% | **31pp** |
| In-Store Availability | 62% | 64% | **-2pp** ↑ |
| Health Claims | 71% | 33% | **38pp** |
| Ease of Use | 57% | 68% | **-11pp** ↑ |

**Boomer Insight:** Brand familiarity is Boomers' most honest driver — a near-zero gap (2pp) confirming that **brand switching in this cohort is extremely difficult**. Their most misleading stated driver is Canadian-Made, where patriotic sentiment (79%) far outpaces actual purchase behavior (48%). Ease of use is massively underreported but is the #2 actual behavioral driver — packaging, UX, and accessibility matter more than Boomers will admit.

---

## Cross-Generational Comparison Matrix

### What Actually Drives Behavior (Ranked by Behavioral Influence)

| Rank | Gen Z | Millennials | Gen X | Boomers |
|------|-------|-------------|-------|---------|
| #1 | Convenience (61%) | Convenience (72%) | Quality (74%) | Brand Familiarity (74%) |
| #2 | Social Proof (58%) | Quality (68%) | Convenience (71%) | Quality (72%) |
| #3 | Aesthetic (51%) | Loyalty Rewards (53%) | Brand Trust (67%) | Price (71%) |
| #4 | Novelty (47%) | Price (52%) | Price (63%) | Ease of Use (68%) |
| #5 | Price (34%) | Brand Rep (44%) | Product Reviews (48%) | In-Store Availability (64%) |

**The revelation:** Price is the #1 *stated* driver for every generation but ranks #5 for Gen Z, #4 for Millennials, #4 for Gen X, and only #3 for Boomers in actual behavioral influence. **Convenience is the true king** — it's the #1 or #2 actual driver for three of four generations.

---

## Channel Preferences for Purchase Discovery

| Discovery Channel | Gen Z | Millennials | Gen X | Boomers |
|------------------|-------|-------------|-------|---------|
| TikTok / Short Video | **47%** | 28% | 9% | 2% |
| Instagram | 38% | 31% | 14% | 5% |
| Google Search | 29% | **42%** | **48%** | 38% |
| In-Store Browsing | 12% | 21% | 34% | **52%** |
| Friend/Family Rec | 34% | 33% | 31% | 41% |
| Email / Newsletter | 4% | 18% | 22% | 28% |
| TV Advertising | 8% | 14% | 27% | **44%** |

---

## The Convenience Economy: A Cross-Generational Truth

Convenience emerged as the most underestimated and underreported purchase driver. Digging deeper:

| Convenience Dimension | Gen Z | Millennials | Gen X | Boomers |
|----------------------|-------|-------------|-------|---------|
| Same-day delivery | 72% | 64% | 48% | 29% |
| One-click reorder | 58% | 67% | 52% | 31% |
| In-app purchase | 63% | 59% | 38% | 14% |
| Curbside pickup | 34% | 51% | 58% | 61% |
| In-store availability | 21% | 38% | 54% | 68% |

The definition of convenience itself is generational — digital immediacy for Gen Z, frictionless reordering for Millennials, flexible fulfillment for Gen X, physical availability for Boomers.

---

## Strategic Implications

> **SO WHAT?** Stop designing generational marketing strategies around stated preferences. The Say-Do Gap™ data proves that what people *say* drives their purchases and what *actually* drives them are fundamentally different. Convenience is the universal driver that every generation underreports. Brands that win in 2026 will optimize for frictionless purchase experiences first and layer generational messaging on top — not the other way around.

### Recommended Actions
1. **Audit your convenience infrastructure** — it's the #1 actual driver across generations and the most underleveraged competitive advantage
2. **For Gen Z campaigns:** Lead with social proof and aesthetics, not price or values
3. **For Millennial campaigns:** Invest in loyalty programs and demonstrable quality signals
4. **For Gen X campaigns:** Build on brand trust with transparent product information
5. **For Boomer campaigns:** Prioritize availability, ease of use, and familiar brand cues

---

*Source: IMI Pulse™ Purchase Drivers Study, n=5,200 Canadian adults 18+, combined stated-preference survey + behavioral purchase panel, fielded Oct-Dec 2025.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze the Ad Pre-Test Results for campaigns A, B, and C. Compare creative testing results and recommend the strongest."] = """# 🎬 Ad Pre-Test Results — Campaigns A, B, and C

## Executive Summary

Three creative executions were tested among the target audience prior to in-market launch. The results are decisive: **Campaign B is the strongest overall creative**, delivering the highest persuasion score (7.8/10), strongest purchase intent lift (+14pp), and best emotional resonance across demographics. Campaign A shows strong brand linkage but weak emotional connection. Campaign C polarizes — beloved by Gen Z, rejected by 35+ audiences. Our recommendation: **launch Campaign B as the primary execution**, with Campaign C as a Gen Z-specific digital variant.

---

## Overall Creative Scorecard

| Metric | Campaign A "Trust the Process" | Campaign B "Made for This" | Campaign C "No Filter" | IMI Norm |
|--------|-------------------------------|---------------------------|----------------------|----------|
| **Overall Score** | **6.4/10** | **7.8/10** ★ | **6.1/10** | 6.5 |
| Attention / Breakthrough | 6.1 | 7.2 | **8.4** | 6.3 |
| Brand Linkage | **8.1** | 7.4 | 5.2 | 6.8 |
| Message Comprehension | 7.5 | **8.2** | 5.8 | 7.0 |
| Emotional Resonance | 5.3 | **8.1** | 7.4 | 6.2 |
| Persuasion | 6.0 | **7.8** | 5.9 | 6.1 |
| Purchase Intent Lift | +8pp | **+14pp** ★ | +6pp | +7pp |
| Uniqueness / Distinctiveness | 5.2 | 7.1 | **8.8** | 6.0 |

**Headline:** Campaign B exceeds IMI norms on every metric except Attention (where Campaign C leads) and Brand Linkage (where Campaign A leads). Its balanced strength across all dimensions makes it the safest and most effective choice for broad-market deployment.

---

## Detailed Metric Analysis

### Attention & Breakthrough

| Campaign | First 3-Sec Retention | Full View Rate | Replay Rate | Scroll-Stop (Social) |
|----------|----------------------|---------------|-------------|---------------------|
| A | 62% | 71% | 8% | 3.2% |
| B | 74% | 82% | 14% | 5.1% |
| C | **88%** | 76% | **22%** | **7.8%** |
| **Norm** | 65% | 73% | 11% | 4.0% |

Campaign C's opening sequence — raw, unpolished, user-generated-content aesthetic — drives exceptional initial attention. Its 88% first-3-second retention is in the **top 5% of all ads tested in 2025**. However, its full-view completion rate drops below Campaign B, suggesting the attention doesn't sustain through the brand message.

### Brand Linkage

| Campaign | Correct Brand Attribution | Competitor Misattribution | Unbranded Recall | Linkage Index |
|----------|--------------------------|--------------------------|------------------|--------------|
| A | **78%** | 8% | 14% | **112** |
| B | 71% | 11% | 18% | **102** |
| C | 48% | 24% | 28% | **69** |
| **Norm** | 68% | 14% | 18% | 100 |

**Critical Issue with Campaign C:** 24% of viewers attributed it to a competitor — the highest misattribution rate in the test. The UGC-style creative, while attention-grabbing, lacks sufficient brand integration. Nearly 1 in 4 viewers are building equity for someone else.

### Emotional Resonance (Implicit Response Testing)

| Emotion | Campaign A | Campaign B | Campaign C |
|---------|-----------|-----------|-----------|
| Trust | **6.8** | 7.2 | 4.1 |
| Excitement | 4.2 | 7.4 | **8.1** |
| Warmth | 5.1 | **8.3** | 5.8 |
| Inspiration | 5.9 | **7.9** | 6.4 |
| Humor | 3.2 | 5.1 | **7.6** |
| Confusion | 2.1 | 1.4 | **4.8** |
| Boredom | **4.7** | 1.8 | 2.3 |

Campaign B triggers the highest warmth (8.3) and inspiration (7.9) — the two emotions most correlated with long-term brand building (r=0.78 and r=0.72 respectively with brand preference shift). Campaign A's boredom score of 4.7 is a red flag: it's safe but unengaging.

---

## Audience Segmentation Analysis

### Performance by Age Group

| Age Group | Campaign A | Campaign B | Campaign C |
|-----------|-----------|-----------|-----------|
| 18-24 | 5.8 | 7.2 | **8.4** |
| 25-34 | 6.2 | **7.9** | 7.1 |
| 35-44 | 6.8 | **8.1** | 5.4 |
| 45-54 | 6.9 | **7.8** | 4.6 |
| 55+ | **6.7** | 7.4 | 3.8 |

**The polarization problem:** Campaign C scores 8.4 with 18-24 but collapses to 3.8 with 55+. Campaign B maintains above-norm scores across every age bracket — the most broadly effective creative. Campaign A is flat and undifferentiated by age, suggesting it fails to connect deeply with anyone.

### Performance by Gender
| Gender | Campaign A | Campaign B | Campaign C |
|--------|-----------|-----------|-----------|
| Female | 6.1 | **8.2** | 6.3 |
| Male | 6.7 | 7.3 | 5.9 |
| Non-binary | 5.9 | 7.6 | **7.8** |

Campaign B over-indexes significantly with female audiences (8.2) — a +1.7 lift above norm — driven by its warmth and inspiration emotional profile.

---

## Purchase Intent Impact

### "More likely to purchase after seeing this ad"

| Response | Campaign A | Campaign B | Campaign C |
|----------|-----------|-----------|-----------|
| Much more likely | 8% | **18%** | 11% |
| Somewhat more likely | 24% | 29% | 19% |
| No change | 51% | 41% | 44% |
| Less likely | 12% | 8% | 18% |
| Much less likely | 5% | 4% | 8% |
| **Net Intent Lift** | **+15pp** | **+35pp** ★ | **+4pp** |

Campaign B's net purchase intent of +35pp is **exceptional** — in the top 10% of all pre-tested ads in the IMI Pulse™ database. Campaign C's 8% "much less likely" response from older audiences drags its net score down significantly.

---

## The Say-Do Gap™ in Ad Response

We cross-referenced stated ad preference with implicit behavioral measures:

| Metric | Stated Preference Rank | Behavioral Influence Rank | Gap |
|--------|----------------------|--------------------------|-----|
| Campaign A | #2 (safe choice) | #3 (low activation) | Overrated |
| Campaign B | #1 (clear winner) | #1 (strongest activation) | **Aligned** ★ |
| Campaign C | #3 (polarizing) | #2 (high activation, narrow) | Underrated with youth |

Campaign B is the rare creative where stated preference and implicit behavioral response align — people say they like it and their behavior confirms it. This alignment is the strongest predictor of in-market success.

---

## Creative Diagnostic Recommendations

| Campaign | Strength | Weakness | Fix |
|----------|----------|----------|-----|
| A | Brand linkage (8.1) | Emotional flatness, boredom | Complete rework — not salvageable without losing its only strength |
| B | Balanced excellence | Slightly lower attention vs. C | Add stronger opening hook — first 2 seconds could be sharper |
| C | Breakthrough attention | Brand misattribution, age polarization | Add clearer brand integration in seconds 3-5; deploy only on digital for 18-34 |

---

## Strategic Implications

> **SO WHAT?** Campaign B is the clear primary execution — it delivers the rare combination of broad demographic appeal, strong persuasion, emotional resonance, and consistent brand linkage. Campaign C should not be discarded; it is a powerful Gen Z-specific digital asset if the brand integration issue is resolved. Campaign A should be shelved — it is the definition of safe mediocrity that wastes media spend without building brand equity or driving action.

### Recommended Media Deployment
1. **Campaign B:** Primary execution across all channels — TV, digital, social, OOH
2. **Campaign C (revised):** Secondary execution — TikTok, Instagram Reels, YouTube Shorts (18-34 targeting only)
3. **Campaign A:** Do not deploy — reinvest budget into higher-performing executions

---

*Source: IMI Pulse™ Ad Pre-Test, n=2,400 Canadian adults 18+ (800 per cell), online forced-exposure methodology with implicit response measurement, fielded Jan 2026.*
*Analysis: Klaus — IMI Intelligence Engine*"""


CACHED_RESPONSES["Analyze the Competitive Benchmark QSR Category data. Show quick service restaurant competitive positioning."] = """# 🍔 Competitive Benchmark — QSR Category Analysis

## Executive Summary

The Canadian Quick Service Restaurant (QSR) landscape in Q4 2025 reveals **a category under pressure and a hierarchy in flux**. Tim Hortons retains the #1 position on reach and cultural relevance but is bleeding equity among under-35 consumers at an alarming rate. McDonald's maintains the strongest operational consistency scores. The breakout story: **Popeyes and Freshii-style better-QSR concepts are capturing disproportionate share of new visits**, exposing the vulnerability of legacy positioning built on convenience alone. The Say-Do Gap™ in this category is particularly revealing — Canadians say they want healthier QSR options but continue to order the same high-comfort items.

---

## Overall Competitive Positioning Scorecard

| Brand | Overall Index (0-100) | Brand Health | Value Perception | Experience | Growth Trajectory |
|-------|----------------------|-------------|-----------------|------------|-------------------|
| Tim Hortons | **74** | 71 | 78 | 69 | ↓ Declining (-4 YoY) |
| McDonald's | **72** | 68 | 72 | **81** | → Stable (+1 YoY) |
| Subway | **58** | 54 | 64 | 57 | ↓ Declining (-3 YoY) |
| Wendy's | **61** | 59 | 66 | 63 | ↑ Rising (+5 YoY) |
| A&W Canada | **67** | **73** | 68 | 65 | ↑ Rising (+7 YoY) |
| Popeyes | **63** | 64 | 59 | 62 | ↑↑ Surging (+12 YoY) |
| Starbucks | **65** | 67 | 51 | 72 | → Stable (+0 YoY) |
| Harvey's | **52** | 51 | 58 | 49 | ↓ Declining (-6 YoY) |

**Category average index: 64.** Tim Hortons and McDonald's sit above category; A&W and Starbucks cluster close. The notable movement: A&W (+7) and Popeyes (+12) are the only brands with accelerating growth trajectories.

---

## Brand Health Deep Dive

### Brand Funnel Analysis (% of Canadian adults)

| Stage | Tim Hortons | McDonald's | A&W | Wendy's | Popeyes | Subway | Starbucks |
|-------|------------|-----------|------|---------|---------|--------|-----------|
| Unaided Awareness | **89%** | 84% | 61% | 52% | 38% | 58% | 71% |
| Consideration | 72% | 68% | 54% | 44% | 34% | 41% | 48% |
| Trial (Past 6 Mo) | 68% | 61% | 42% | 38% | 28% | 39% | 44% |
| Regular Use (Monthly+) | **41%** | 38% | 19% | 14% | 11% | 18% | 22% |
| Loyalty (Primary QSR) | **24%** | 19% | 8% | 5% | 3% | 7% | 9% |

**Key Insight:** Tim Hortons' funnel conversion from Awareness (89%) to Loyalty (24%) is 27% — the highest in category. But the story changes by age. Among 18-34, Tim Hortons' loyalty drops to 14% while McDonald's rises to 22%. **Tim Hortons is losing the next generation.**

### Net Promoter Scores
| Brand | NPS | vs. 2024 | Category Rank |
|-------|-----|----------|--------------|
| A&W Canada | **+32** | +8 ↑ | #1 |
| McDonald's | +21 | +2 ↑ | #2 |
| Popeyes | +18 | +11 ↑ | #3 |
| Wendy's | +14 | +4 ↑ | #4 |
| Tim Hortons | +11 | **-7 ↓** | #5 |
| Starbucks | +8 | -2 ↓ | #6 |
| Subway | -4 | -3 ↓ | #7 |

**A&W leads NPS** — their "better ingredients" positioning is translating into genuine advocacy. Tim Hortons' NPS decline of -7 points is the largest single-year drop in the tracker's history, driven by quality perception issues and price increases that outpaced perceived value.

---

## Value Perception Analysis

### "Good value for money" (% Agree)

| Brand | Overall | 18-34 | 35-54 | 55+ |
|-------|---------|-------|-------|-----|
| Tim Hortons | 62% | 48% | 64% | 74% |
| McDonald's | 58% | 61% | 56% | 54% |
| A&W | 52% | 57% | 51% | 46% |
| Wendy's | 54% | 58% | 52% | 48% |
| Subway | 44% | 39% | 46% | 49% |
| Popeyes | 47% | 54% | 44% | 38% |
| Starbucks | 28% | 32% | 26% | 21% |

**The Tim Hortons value erosion:** Among 18-34, only 48% see Tim Hortons as good value — down from 61% in 2024. The $2.30+ medium coffee threshold has broken the psychological value equation for younger consumers. McDonald's has overtaken Tim Hortons in youth value perception for the first time.

### Average Transaction Value (ATV)
| Brand | Avg ATV | YoY Change | Traffic Trend |
|-------|---------|-----------|---------------|
| Tim Hortons | $6.80 | +12% | -4% visits |
| McDonald's | $11.40 | +8% | +1% visits |
| A&W | $13.20 | +6% | +7% visits |
| Wendy's | $12.80 | +9% | +4% visits |
| Popeyes | $14.60 | +5% | +18% visits |
| Starbucks | $7.90 | +11% | -2% visits |

**Popeyes' +18% visit growth** is the category outlier, driven by chicken sandwich momentum and effective social media marketing to younger demographics.

---

## Experience & Operations Benchmark

### Customer Experience Scores (1-10)
| Dimension | Tim Hortons | McDonald's | A&W | Wendy's | Popeyes |
|-----------|------------|-----------|------|---------|---------|
| Speed of Service | 7.1 | **8.4** | 6.8 | 7.2 | 6.1 |
| Order Accuracy | 6.4 | **8.1** | 7.3 | 7.0 | 6.5 |
| Food Quality | 5.8 | 6.4 | **7.9** | 6.8 | **7.6** |
| Cleanliness | 6.2 | **7.8** | 7.4 | 6.9 | 6.1 |
| Digital Ordering | 6.1 | **8.2** | 6.4 | 6.7 | 5.3 |
| Staff Friendliness | 6.9 | 6.8 | **7.8** | 7.1 | 6.6 |
| **Average** | **6.4** | **7.6** | **7.3** | **7.0** | **6.4** |

**McDonald's operational dominance** is clear — #1 in Speed, Accuracy, Cleanliness, and Digital. This is the result of sustained operations investment. Tim Hortons' 6.4 average is now below category mean (6.9), with food quality (5.8) as the critical weak point.

### Digital Engagement
| Brand | App Downloads (2025) | Loyalty Members | Digital Order % | App Rating |
|-------|---------------------|----------------|----------------|-----------|
| Tim Hortons | 2.8M | 4.1M | 22% | 3.4★ |
| McDonald's | **3.4M** | **5.2M** | **34%** | **4.2★** |
| A&W | 890K | 1.2M | 18% | 4.0★ |
| Starbucks | 2.1M | 3.8M | **41%** | 4.1★ |
| Wendy's | 680K | 820K | 14% | 3.8★ |

---

## The Say-Do Gap™ — QSR Edition

What Canadians say they want from QSR vs. what they actually order:

| Stated Preference | % Who Say It | Actual Order Behavior | Say-Do Gap™ |
|------------------|-------------|----------------------|-------------|
| "I try to choose healthier options" | 64% | 18% order salad/healthy option | **46pp** |
| "I prefer Canadian-owned QSR" | 71% | 41% of visits to Canadian brands | **30pp** |
| "Quality matters more than price" | 58% | 72% use coupons/deals | **-14pp** ↑ |
| "I'd pay more for sustainable packaging" | 47% | 6% select sustainable option when offered | **41pp** |
| "I'm reducing QSR visits for health" | 52% | Average visit frequency UP 3% YoY | **Reversed** |

The QSR Say-Do Gap™ is among the widest of any category we track. The "reducing visits" claim vs. actual increased frequency is a textbook example of aspirational self-reporting.

---

## Competitive Positioning Map

### Value vs. Quality Perception (Indexed)
```
HIGH QUALITY
    │
    │         ★ A&W (Quality leader)
    │      ★ Popeyes
    │   ★ Wendy's
    │              ★ Starbucks (premium, not food-quality)
    │   ★ McDonald's
    │
    │── ★ Tim Hortons (value eroding, quality declining) ──── HIGH VALUE
    │
    │   ★ Subway
    │   ★ Harvey's
    │
LOW QUALITY
```

Tim Hortons occupies a dangerous middle position — no longer the clear value leader, not perceived as quality. A&W has claimed the quality-value sweet spot that Tim Hortons once owned.

---

## Strategic Implications

> **SO WHAT?** The Canadian QSR hierarchy is being reshuffled by three forces: (1) value perception erosion at legacy brands due to price increases outpacing quality, (2) operational excellence as a differentiator (McDonald's) vs. brand heritage as a depreciating asset (Tim Hortons), and (3) insurgent brands capturing youth share by combining food quality with cultural relevance (Popeyes, A&W). The Say-Do Gap™ data confirms that health positioning in QSR is largely performative — brands should invest in perceived quality and value, not health messaging that consumers ignore at point of purchase.

### Recommended Actions
1. **Tim Hortons:** Urgent quality reinvestment needed — food quality score of 5.8 is an existential threat; value narrative must be rebuilt with under-35 consumers
2. **McDonald's:** Leverage operational advantage into brand storytelling — consistency is undervalued in marketing but drives repeat behavior
3. **A&W:** Accelerate — highest NPS, best quality scores, strongest growth; the window to capture disaffected Tim Hortons loyalists is open now
4. **Popeyes:** Sustain momentum with social-first strategy but invest in operations before growth outpaces execution capacity
5. **All brands:** Stop over-indexing on health messaging — the 46pp Say-Do Gap™ proves it doesn't drive QSR visits. Win on taste, value, and convenience

---

*Source: IMI Pulse™ QSR Competitive Benchmark, n=6,800 Canadian adults 18+, combined brand tracking survey + transaction data + app analytics, fielded Q4 2025.*
*Analysis: Klaus — IMI Intelligence Engine*"""


print(f"Created {len(CACHED_RESPONSES)} cached responses (Part 1)")
