"""
Case Study Management for Klaus IMI Platform.
Stores, searches, and manages IMI marketing research case studies.
"""

import json
import logging
import os
import re
from typing import Optional

from config import VECTOR_STORE_PATH

logger = logging.getLogger("imi_case_studies")

CASE_STUDIES_PATH = os.path.join(os.path.dirname(VECTOR_STORE_PATH), "case_studies.json")
os.makedirs(os.path.dirname(CASE_STUDIES_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def load_studies() -> list[dict]:
    """Load all case studies from disk."""
    if not os.path.exists(CASE_STUDIES_PATH):
        studies = _seed_studies()
        save_studies(studies)
        return studies
    try:
        with open(CASE_STUDIES_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to load case studies: %s", e)
        return []


def save_studies(studies: list[dict]) -> None:
    """Persist case studies to disk."""
    with open(CASE_STUDIES_PATH, "w") as f:
        json.dump(studies, f, indent=2)
    logger.info("Saved %d case studies to %s", len(studies), CASE_STUDIES_PATH)


def get_study(study_id: str) -> Optional[dict]:
    """Return a single study by ID, or None."""
    for s in load_studies():
        if s["id"] == study_id:
            return s
    return None


def add_studies(new_studies: list[dict]) -> int:
    """Append new studies, auto-assigning IDs. Returns count added."""
    existing = load_studies()
    max_id = 0
    for s in existing:
        try:
            num = int(s["id"].replace("CS-", ""))
            if num > max_id:
                max_id = num
        except (ValueError, KeyError):
            pass
    for i, s in enumerate(new_studies, start=1):
        if "id" not in s or not s["id"]:
            s["id"] = f"CS-{max_id + i:04d}"
    existing.extend(new_studies)
    save_studies(existing)
    return len(new_studies)


def search_studies(query: str, n: int = 5) -> list[dict]:
    """Keyword search across title, summary, industry, methodology, pillar."""
    studies = load_studies()
    tokens = [t.lower() for t in re.split(r"\W+", query) if len(t) > 1]
    if not tokens:
        return studies[:n]

    scored = []
    for s in studies:
        blob = " ".join([
            s.get("title", ""),
            s.get("summary", ""),
            s.get("industry", ""),
            s.get("methodology", ""),
            s.get("pillar", ""),
            s.get("client", ""),
            " ".join(s.get("key_findings", [])),
        ]).lower()
        score = sum(blob.count(t) for t in tokens)
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n]]


# ---------------------------------------------------------------------------
# Seed data — 50 representative IMI-style case studies
# ---------------------------------------------------------------------------

def _seed_studies() -> list[dict]:
    return [
        {
            "id": "CS-0001",
            "title": "QSR Brand Health Tracker: Measuring Consideration Drivers",
            "client": "Major QSR Chain",
            "industry": "QSR",
            "methodology": "brand health",
            "year": 2024,
            "pillar": "discover",
            "summary": "Longitudinal brand health tracking study across 12 markets measuring aided and unaided awareness, consideration, preference, and loyalty metrics for a top-5 QSR brand. Identified value perception as the primary driver of consideration decline among 18-34 year-olds, leading to a repositioning of the value menu architecture.",
            "key_findings": [
                "Value perception dropped 14 points among 18-34 year-olds over 6 months",
                "Unaided awareness remained stable at 87% but consideration fell to 34%",
                "Competitive set expanded as fast-casual brands entered value positioning",
                "Menu innovation was the #1 driver of re-trial among lapsed customers"
            ]
        },
        {
            "id": "CS-0002",
            "title": "CPG Segmentation: Identifying High-Value Snacking Occasions",
            "client": "Global Snack Manufacturer",
            "industry": "CPG",
            "methodology": "segmentation",
            "year": 2023,
            "pillar": "discover",
            "summary": "Occasion-based segmentation study combining survey data with purchase panel analytics to identify underserved snacking occasions. Revealed that 'better-for-you evening snacking' represented a $2.1B whitespace opportunity, with working parents aged 30-45 as the primary target.",
            "key_findings": [
                "Six distinct snacking occasions identified through latent class analysis",
                "Evening 'permissible indulgence' occasion was 3x larger than brand assumed",
                "Health-conscious snackers willing to pay 22% premium for clean-label products",
                "Cross-category competition from yogurt and fruit most pronounced at 8-10pm"
            ]
        },
        {
            "id": "CS-0003",
            "title": "Financial Services Campaign Evaluation: Digital-First Mortgage Launch",
            "client": "National Bank",
            "industry": "Financial Services",
            "methodology": "campaign evaluation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Pre/post campaign evaluation of a digital-first mortgage product launch measuring awareness lift, message comprehension, and intent to apply. The campaign achieved 18-point awareness lift in target markets but message recall around digital convenience was diluted by rate-focused creative executions.",
            "key_findings": [
                "Awareness lift of 18 points in heavy-up markets vs. 3 points in control",
                "Digital convenience message recalled by only 29% of aware respondents",
                "Rate-focused creative drove 2.4x more click-through but lower brand attribution",
                "Social media outperformed display by 67% on cost-per-qualified-lead"
            ]
        },
        {
            "id": "CS-0004",
            "title": "Telecom Say/Do Gap Analysis: Churn Prediction vs. Stated Intent",
            "client": "Tier-1 Wireless Carrier",
            "industry": "Telecom",
            "methodology": "say/do gap",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Integrated study linking stated satisfaction and switching intent survey data with actual churn behavior over 12 months. Found that 41% of customers who stated they were 'very unlikely to switch' churned within 9 months, revealing a critical say/do gap driven by competitive promotional offers.",
            "key_findings": [
                "41% of self-reported 'loyal' customers churned within 9 months",
                "Price sensitivity was understated by 2.3x in survey vs. behavioral data",
                "NPS alone predicted only 18% of actual churn variance",
                "Adding behavioral signals improved churn prediction accuracy to 73%"
            ]
        },
        {
            "id": "CS-0005",
            "title": "Automotive Sponsorship Valuation: Motorsport Portfolio ROI",
            "client": "Premium Auto Manufacturer",
            "industry": "Automotive",
            "methodology": "sponsorship valuation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Comprehensive sponsorship valuation study quantifying the brand equity and sales impact of a motorsport sponsorship portfolio worth $45M annually. Used media equivalency modeling combined with brand lift measurement to demonstrate a 3.2x ROI, with the strongest returns among male enthusiasts aged 25-44.",
            "key_findings": [
                "Total sponsorship portfolio delivered $144M in equivalent media value",
                "Brand consideration among motorsport fans was 2.1x higher than non-fans",
                "Hospitality activations drove the highest per-contact brand lift",
                "Digital extensions of sponsorship outperformed traditional signage 4:1 on engagement"
            ]
        },
        {
            "id": "CS-0006",
            "title": "Retail Brand Architecture Study: Private Label Positioning",
            "client": "National Grocery Retailer",
            "industry": "Retail",
            "methodology": "brand health",
            "year": 2023,
            "pillar": "discover",
            "summary": "Brand architecture evaluation of a three-tier private label strategy measuring cannibalization risk, quality perceptions, and price-value trade-offs across 14 categories. Found that the premium private label tier was perceived as equivalent to national brands on quality but lacked differentiated positioning.",
            "key_findings": [
                "Premium private label achieved quality parity with national brands in 9 of 14 categories",
                "Mid-tier cannibalized premium tier by 23% when shelf placement was adjacent",
                "Value-tier shoppers showed zero trade-up propensity without trial incentives",
                "Brand architecture clarity scored 34/100 among regular shoppers"
            ]
        },
        {
            "id": "CS-0007",
            "title": "QSR Segmentation: Breakfast Daypart Consumer Typology",
            "client": "Quick-Service Breakfast Chain",
            "industry": "QSR",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Needs-based segmentation study focused on the breakfast daypart, identifying five distinct consumer segments based on morning routines, nutritional priorities, and convenience needs. The 'Grab & Go Optimizer' segment represented 28% of the addressable market and was dramatically underserved by current menu offerings.",
            "key_findings": [
                "Five segments identified: Routine Loyalists, Grab & Go Optimizers, Health Seekers, Indulgers, Skippers",
                "Grab & Go Optimizers drove 34% of breakfast revenue but had lowest satisfaction",
                "Drive-through speed was 4x more important than menu variety for this segment",
                "Protein-forward menu items increased average check by $1.40 among Health Seekers"
            ]
        },
        {
            "id": "CS-0008",
            "title": "CPG Campaign Evaluation: Plant-Based Product Launch",
            "client": "Multinational Food Company",
            "industry": "CPG",
            "methodology": "campaign evaluation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Multi-wave campaign evaluation for a plant-based product line extension, measuring awareness funnel progression, trial barriers, and repeat purchase intent across three creative strategies. Influencer-led creative outperformed traditional advertising by 2.8x on trial intent among flexitarians.",
            "key_findings": [
                "Influencer creative drove 31% trial intent vs. 11% for traditional TV",
                "Taste skepticism remained the #1 trial barrier at 58% of non-triers",
                "In-store sampling converted 44% of skeptics to first purchase",
                "Repeat purchase intent was 67% among triers, suggesting strong product-market fit"
            ]
        },
        {
            "id": "CS-0009",
            "title": "Financial Services Brand Health: Trust Rebuilding Post-Crisis",
            "client": "Regional Banking Group",
            "industry": "Financial Services",
            "methodology": "brand health",
            "year": 2023,
            "pillar": "optimize",
            "summary": "Ongoing brand health monitor tracking trust, consideration, and switching intent following a data breach incident. Measured the effectiveness of remediation communications over 18 months. Trust metrics recovered to 80% of pre-crisis levels after 14 months, with transparent communication being the single most impactful recovery lever.",
            "key_findings": [
                "Trust dropped 42 points immediately post-crisis and recovered 34 points over 14 months",
                "Proactive customer outreach was 3x more effective than mass media at trust recovery",
                "Customers who received personal apology communication showed 89% retention rate",
                "Competitive consideration spiked 27 points at crisis peak but normalized within 6 months"
            ]
        },
        {
            "id": "CS-0010",
            "title": "Telecom Segmentation: 5G Adoption Readiness",
            "client": "National Wireless Provider",
            "industry": "Telecom",
            "methodology": "segmentation",
            "year": 2023,
            "pillar": "discover",
            "summary": "Technology adoption segmentation study classifying the wireless customer base into readiness tiers for 5G upgrade marketing. Combined survey-based tech affinity scores with actual device upgrade cycles and data usage patterns to build a five-segment model predicting upgrade propensity.",
            "key_findings": [
                "Only 12% of base qualified as 'Early Enthusiasts' ready for premium 5G plans",
                "44% were 'Pragmatic Upgraders' who needed tangible use-case demonstrations",
                "Speed alone was insufficient motivation; streaming quality and gaming drove interest",
                "Price sensitivity for 5G premium ranged from $5/mo (Enthusiasts) to $0 (Resistors)"
            ]
        },
        {
            "id": "CS-0011",
            "title": "Automotive Brand Health: EV Transition Perception Study",
            "client": "Legacy Auto Manufacturer",
            "industry": "Automotive",
            "methodology": "brand health",
            "year": 2024,
            "pillar": "discover",
            "summary": "Brand health tracking study measuring how a legacy manufacturer's brand equity transferred to their new EV lineup. Found significant perception gaps between ICE and EV brand associations, with reliability and craftsmanship carrying over but innovation and sustainability lagging Tesla and newer EV-native brands.",
            "key_findings": [
                "Reliability perception carried over at 78% strength from ICE to EV models",
                "Innovation perception for EV lineup was 31 points behind EV-native competitors",
                "Brand loyalists showed 3.2x higher EV consideration than the general market",
                "Dealer experience was the #1 barrier cited by EV-intenders who chose competitors"
            ]
        },
        {
            "id": "CS-0012",
            "title": "Retail Campaign Evaluation: Holiday Loyalty Program Relaunch",
            "client": "Department Store Chain",
            "industry": "Retail",
            "methodology": "campaign evaluation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Campaign effectiveness study for a holiday season loyalty program relaunch, measuring enrollment lift, incremental spend, and program awareness. The omnichannel creative approach drove a 156% increase in new enrollments versus prior year, with mobile-first touchpoints delivering the highest conversion rates.",
            "key_findings": [
                "New loyalty enrollments increased 156% YoY during the campaign period",
                "Loyalty members spent $47 more per transaction than non-members on average",
                "Mobile push notifications drove 38% of all enrollments during the campaign",
                "Email creative with personalized offers outperformed generic messaging by 4.1x"
            ]
        },
        {
            "id": "CS-0013",
            "title": "QSR Say/Do Gap: Health Claims vs. Actual Ordering Behavior",
            "client": "National Burger Chain",
            "industry": "QSR",
            "methodology": "say/do gap",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Study linking stated health-consciousness in survey responses with POS transaction data across 2,400 locations. Revealed that 62% of consumers who self-identified as 'health-conscious' regularly ordered items exceeding 1,200 calories, demonstrating a massive say/do gap in QSR health positioning.",
            "key_findings": [
                "62% of self-described health-conscious consumers ordered high-calorie meals regularly",
                "Salad menu items were ordered by only 7% of health-conscious respondents",
                "Portion size reduction was preferred 2:1 over menu substitution among health-seekers",
                "Calorie labeling had measurable impact for only 18% of the health-conscious segment"
            ]
        },
        {
            "id": "CS-0014",
            "title": "CPG Sponsorship Valuation: Music Festival Portfolio Assessment",
            "client": "Global Beverage Brand",
            "industry": "CPG",
            "methodology": "sponsorship valuation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Valuation of a $22M music festival sponsorship portfolio spanning six major festivals. Measured brand visibility, social amplification, sampling effectiveness, and long-term brand affinity impact. Found that experiential activations at festivals generated 5.7x more brand recall than passive signage.",
            "key_findings": [
                "Experiential activations delivered 5.7x higher brand recall than passive signage",
                "Social sharing from festival attendees generated $8.3M in earned media value",
                "On-site sampling drove 41% trial among non-users with 28% conversion to regular purchase",
                "Festival sponsorship lifted brand 'coolness' score by 19 points among Gen Z"
            ]
        },
        {
            "id": "CS-0015",
            "title": "Financial Services Segmentation: Wealth Management Prospect Typology",
            "client": "Investment Management Firm",
            "industry": "Financial Services",
            "methodology": "segmentation",
            "year": 2023,
            "pillar": "discover",
            "summary": "Psychographic and behavioral segmentation of high-net-worth individuals ($1M+ investable assets) to optimize wealth management client acquisition. Identified four distinct investor personas with different advisory needs, fee sensitivity, and digital engagement preferences.",
            "key_findings": [
                "Four segments: Delegators (31%), Validators (27%), Self-Directors (24%), Hybrid (18%)",
                "Delegators had 3x higher lifetime value but required 2x more advisor touchpoints",
                "Self-Directors were most responsive to fintech-style digital tools and reporting",
                "Fee transparency was the #1 acquisition driver across all segments"
            ]
        },
        {
            "id": "CS-0016",
            "title": "Telecom Campaign Evaluation: Family Plan Bundle Launch",
            "client": "Regional Wireless Carrier",
            "industry": "Telecom",
            "methodology": "campaign evaluation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Pre/post evaluation of a family plan bundle campaign targeting households with 2+ lines. Measured message cut-through, competitive consideration impact, and subscriber acquisition cost. The campaign reduced cost-per-acquisition by 34% versus the previous quarter through improved targeting of family decision-makers.",
            "key_findings": [
                "Campaign reduced CPA by 34% through refined household-level targeting",
                "Female household decision-makers drove 61% of family plan switches",
                "Parental controls messaging increased consideration by 23 points among parents",
                "Competitive win-rate improved from 18% to 29% during the campaign flight"
            ]
        },
        {
            "id": "CS-0017",
            "title": "Automotive Segmentation: Luxury SUV Buyer Journey Mapping",
            "client": "European Luxury Brand",
            "industry": "Automotive",
            "methodology": "segmentation",
            "year": 2023,
            "pillar": "discover",
            "summary": "Decision journey segmentation for the luxury SUV category combining survey-based need states with digital behavior tracking and dealer visit data. Mapped the average 4.2-month purchase journey and identified three critical decision inflection points where competitive defection was most likely.",
            "key_findings": [
                "Average luxury SUV purchase journey lasted 4.2 months with 19 digital touchpoints",
                "Three critical inflection points: initial research, test drive, and negotiation",
                "62% of defections occurred at the test drive stage due to competitor experience",
                "Online configurator engagement predicted dealer visit within 14 days with 71% accuracy"
            ]
        },
        {
            "id": "CS-0018",
            "title": "Retail Say/Do Gap: Sustainability Claims vs. Purchase Behavior",
            "client": "Fast Fashion Retailer",
            "industry": "Retail",
            "methodology": "say/do gap",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Study comparing stated sustainability preferences with actual purchase data across 180 stores and e-commerce. Found that while 73% of shoppers claimed sustainability influenced their purchases, sustainable product lines accounted for only 8% of revenue, revealing one of the largest say/do gaps observed in retail research.",
            "key_findings": [
                "73% claimed sustainability influence but sustainable lines were only 8% of revenue",
                "Price premium tolerance for sustainable items was stated at 25% but actual was 7%",
                "Sustainable product discovery was the primary barrier, not willingness to pay",
                "In-store sustainability signage increased sustainable line sales by 31%"
            ]
        },
        {
            "id": "CS-0019",
            "title": "QSR Brand Health: Drive-Through Experience Benchmarking",
            "client": "Pizza Delivery Chain",
            "industry": "QSR",
            "methodology": "brand health",
            "year": 2023,
            "pillar": "optimize",
            "summary": "Competitive benchmarking study measuring drive-through and delivery experience metrics across 8 major QSR brands. Combined mystery shopping, customer satisfaction surveys, and operational data to identify experience gaps. Speed-to-order was the strongest predictor of overall satisfaction, outweighing food quality in importance.",
            "key_findings": [
                "Speed-to-order explained 34% of overall satisfaction variance, more than food quality",
                "Order accuracy above 95% was table stakes; below that, NPS dropped precipitously",
                "Digital ordering customers reported 18% higher satisfaction than phone orderers",
                "Delivery time expectations tightened to 25 minutes, down from 35 minutes two years prior"
            ]
        },
        {
            "id": "CS-0020",
            "title": "CPG Brand Health: Beverage Category Competitive Landscape",
            "client": "Sports Drink Brand",
            "industry": "CPG",
            "methodology": "brand health",
            "year": 2024,
            "pillar": "discover",
            "summary": "Comprehensive competitive brand health study across the sports and functional beverage category measuring 14 KPIs for 12 brands. Mapped the evolving competitive landscape as energy drink brands encroached on hydration positioning and new functional ingredients disrupted traditional sports drink differentiation.",
            "key_findings": [
                "Energy drinks captured 19% of traditional sports drink occasions among 18-24s",
                "Electrolyte awareness increased 340% YoY driven by social media health influencers",
                "Natural ingredient sourcing overtook athletic endorsements as the top consideration driver",
                "Sugar content was now the #1 rejection reason, cited by 47% of category rejecters"
            ]
        },
        {
            "id": "CS-0021",
            "title": "Financial Services Say/Do Gap: Digital Banking Adoption Intent",
            "client": "Digital-First Bank",
            "industry": "Financial Services",
            "methodology": "say/do gap",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Matched panel study comparing stated digital banking feature adoption intent with actual usage data over 6 months post-launch. Found that mobile check deposit had 89% stated intent but only 23% actual adoption, while peer-to-peer payments had 45% stated intent but 61% actual usage, revealing asymmetric say/do patterns.",
            "key_findings": [
                "Mobile check deposit: 89% stated intent, 23% actual usage — massive overprediction",
                "P2P payments: 45% stated intent, 61% actual usage — rare underprediction",
                "Feature discoverability explained 72% of adoption gap for underused features",
                "Push notification onboarding increased feature adoption by 3.4x in first 30 days"
            ]
        },
        {
            "id": "CS-0022",
            "title": "Telecom Sponsorship Valuation: Esports Partnership ROI",
            "client": "Mobile Carrier",
            "industry": "Telecom",
            "methodology": "sponsorship valuation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "ROI analysis of a $15M esports sponsorship portfolio including team sponsorships, tournament naming rights, and streamer partnerships. Measured brand awareness, consideration lift, and subscriber acquisition among Gen Z and millennial gamers. Found that streamer integrations delivered 7.2x higher engagement-per-dollar than tournament signage.",
            "key_findings": [
                "Streamer integrations delivered 7.2x higher engagement-per-dollar than signage",
                "Brand awareness among 16-24 gamers increased 22 points over 12 months",
                "Esports-sourced subscribers had 14% lower churn than average new subscribers",
                "Authenticity perception required minimum 18-month sponsorship commitment to build"
            ]
        },
        {
            "id": "CS-0023",
            "title": "Automotive Campaign Evaluation: EV Test Drive Program",
            "client": "Mass-Market Auto Brand",
            "industry": "Automotive",
            "methodology": "campaign evaluation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Effectiveness evaluation of an experiential EV test drive campaign deployed across 50 markets. Measured awareness-to-test-drive conversion, test-drive-to-purchase conversion, and the impact of ride-and-drive events versus dealer test drives. Pop-up experiential events in non-automotive venues converted at 2.3x the rate of traditional dealer visits.",
            "key_findings": [
                "Pop-up events converted awareness to test drive at 2.3x dealer rate",
                "Test drive to purchase conversion was 31% for experiential vs. 22% for dealer",
                "Range anxiety decreased 44% among consumers who completed a 30+ minute test drive",
                "Peer referral from test drive attendees generated 18% of all subsequent bookings"
            ]
        },
        {
            "id": "CS-0024",
            "title": "Retail Segmentation: Omnichannel Shopping Behavior Clusters",
            "client": "Multi-Format Retailer",
            "industry": "Retail",
            "methodology": "segmentation",
            "year": 2023,
            "pillar": "discover",
            "summary": "Behavioral segmentation combining online browsing, mobile app engagement, in-store traffic, and transaction data to classify shoppers by omnichannel engagement pattern. Identified that 'Channel Fluid' shoppers (23% of base) generated 47% of total revenue and required integrated marketing strategies that traditional channel-based planning could not deliver.",
            "key_findings": [
                "Channel Fluid shoppers (23% of base) generated 47% of total revenue",
                "Average touchpoints before purchase: 6.3 for Channel Fluid vs. 2.1 for Channel Loyal",
                "BOPIS (buy online, pick up in store) users had 34% higher annual spend",
                "Mobile app users who also shopped in-store had 2.8x higher lifetime value"
            ]
        },
        {
            "id": "CS-0025",
            "title": "QSR Campaign Evaluation: Limited-Time Offer Strategy Optimization",
            "client": "Chicken QSR Brand",
            "industry": "QSR",
            "methodology": "campaign evaluation",
            "year": 2023,
            "pillar": "optimize",
            "summary": "Multi-cell campaign test evaluating four creative strategies for limited-time offer promotions across 120 DMAs. Measured awareness velocity, traffic lift, and cannibalization of core menu items. Scarcity-framed messaging ('while supplies last') outperformed discount-framed messaging by 41% on traffic lift while protecting average check size.",
            "key_findings": [
                "Scarcity framing drove 41% higher traffic lift than discount framing",
                "Average check was $0.87 higher with scarcity vs. discount messaging",
                "Optimal LTO duration was 6-8 weeks; shorter created frustration, longer reduced urgency",
                "Social media buzz peaked in week 2 and drove 26% of incremental visits"
            ]
        },
        {
            "id": "CS-0026",
            "title": "CPG Segmentation: Pet Food Premium Buyer Profiles",
            "client": "Premium Pet Food Manufacturer",
            "industry": "CPG",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Attitudinal and behavioral segmentation of premium pet food buyers ($3+/lb) identifying key purchase drivers, information sources, and brand switching triggers. Humanization of pets was the dominant attitudinal driver, with 'Pet Parents' (42% of premium segment) treating food selection with grocery-shopping-level scrutiny.",
            "key_findings": [
                "Pet Parents (42%) read ingredient lists as carefully as their own food purchases",
                "Vet recommendation was the #1 brand selection driver, cited by 56% of premium buyers",
                "Subscription delivery increased retention by 47% vs. retail-only purchasers",
                "Ingredient transparency and sourcing claims drove 3.1x more engagement than taste claims"
            ]
        },
        {
            "id": "CS-0027",
            "title": "Financial Services Campaign Evaluation: Retirement Planning Awareness",
            "client": "Insurance & Retirement Provider",
            "industry": "Financial Services",
            "methodology": "campaign evaluation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Campaign evaluation for a retirement readiness awareness initiative targeting pre-retirees aged 50-65. Measured awareness of retirement income gap, consideration for advisory services, and lead generation quality. Emotionally resonant creative depicting aspirational retirement lifestyles outperformed fear-based 'savings gap' messaging on every measured KPI.",
            "key_findings": [
                "Aspirational creative outperformed fear-based messaging by 2.7x on lead generation",
                "Advisory consultation requests increased 89% during campaign flight",
                "Digital retargeting of site visitors drove 52% of all qualified leads",
                "Consideration for advisory services increased 16 points among the target audience"
            ]
        },
        {
            "id": "CS-0028",
            "title": "Telecom Brand Health: Post-Merger Brand Integration Tracking",
            "client": "Merged Wireless Carrier",
            "industry": "Telecom",
            "methodology": "brand health",
            "year": 2023,
            "pillar": "optimize",
            "summary": "18-month brand health tracking study monitoring brand equity transfer during a major carrier merger and rebrand. Tracked awareness, familiarity, favorability, and consideration for both legacy brands and the new unified brand. Legacy brand equity was successfully transferred at 71% efficiency after 12 months of integrated marketing.",
            "key_findings": [
                "Legacy brand equity transferred at 71% efficiency after 12 months",
                "Customer confusion about brand identity peaked at month 4 and resolved by month 10",
                "Favorability for the merged brand exceeded both legacy brands by month 14",
                "Churn among legacy brand loyalists was 2.3x higher during months 3-6 of transition"
            ]
        },
        {
            "id": "CS-0029",
            "title": "Automotive Say/Do Gap: Willingness to Pay for ADAS Features",
            "client": "Japanese Auto Manufacturer",
            "industry": "Automotive",
            "methodology": "say/do gap",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Conjoint-based willingness to pay study for advanced driver assistance systems (ADAS) validated against actual trim-level sales data. Found systematic overstatement of WTP for safety features in survey contexts, with respondents claiming $2,800 average WTP but actual take-rate data implying $1,100 true valuation.",
            "key_findings": [
                "Stated WTP for ADAS package was $2,800 but revealed preference showed $1,100",
                "Bundling ADAS with comfort features increased actual take-rate by 38%",
                "Parents of teenage drivers showed the smallest say/do gap at 1.4x overstatement",
                "Adaptive cruise control was the single highest-valued feature in both stated and revealed"
            ]
        },
        {
            "id": "CS-0030",
            "title": "Retail Sponsorship Valuation: Professional Sports Stadium Naming Rights",
            "client": "Home Improvement Retailer",
            "industry": "Retail",
            "methodology": "sponsorship valuation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Valuation of a $200M, 20-year stadium naming rights deal measuring annual brand exposure value, local market consideration lift, and incremental foot traffic to stores within the stadium's DMA. Found that naming rights delivered $18M in annual equivalent media value but the real impact was a sustained 8-point consideration advantage in the local market.",
            "key_findings": [
                "Annual equivalent media value of naming rights was $18M against $10M annual cost",
                "Local market consideration was 8 points higher than comparable non-sponsored markets",
                "Game-day foot traffic to nearby stores increased 22% during home games",
                "Community association scores were 34 points higher in the stadium DMA"
            ]
        },
        {
            "id": "CS-0031",
            "title": "QSR Segmentation: Late-Night Daypart Opportunity Sizing",
            "client": "Taco QSR Chain",
            "industry": "QSR",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Occasion and needs-based segmentation for the late-night QSR daypart (10pm-2am) across 18 DMAs. Combined mobile location data, transaction timestamps, and survey-based occasion mapping to quantify the $4.7B late-night QSR market and identify the three primary consumer need states driving late-night visits.",
            "key_findings": [
                "Late-night QSR market sized at $4.7B with 11% annual growth rate",
                "Three need states: Social Fuel (43%), Comfort Craving (31%), Shift Workers (26%)",
                "Social Fuel segment was 78% aged 18-29 with average group size of 3.2 people",
                "Drive-through availability was the #1 brand selection criterion, ahead of menu and price"
            ]
        },
        {
            "id": "CS-0032",
            "title": "CPG Brand Health: Household Cleaning Category Disruption",
            "client": "Legacy Cleaning Brand",
            "industry": "CPG",
            "methodology": "brand health",
            "year": 2023,
            "pillar": "optimize",
            "summary": "Competitive brand health assessment as DTC cleaning brands disrupted the traditional household cleaning category. Tracked how legacy brand perceptions shifted as new entrants emphasized eco-friendly formulations and Instagram-worthy packaging. Found that legacy brands retained trust on efficacy but lost significantly on modernity and environmental perception.",
            "key_findings": [
                "Legacy brand led on efficacy trust by 28 points but trailed on modernity by 41 points",
                "DTC brands captured 14% of category consideration within 18 months of market entry",
                "Sustainability perception gap between legacy and DTC was 52 points among under-35s",
                "Reformulation announcements closed the sustainability gap by only 8 points without proof"
            ]
        },
        {
            "id": "CS-0033",
            "title": "Financial Services Segmentation: Small Business Banking Needs",
            "client": "Commercial Bank",
            "industry": "Financial Services",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Needs-based segmentation of small businesses (1-50 employees) for banking product development and marketing. Combined firmographic data, banking behavior, and decision-maker surveys to identify four segments with distinct product needs, advisory preferences, and digital banking requirements.",
            "key_findings": [
                "Four segments: Growth Seekers (22%), Simplifiers (34%), Relationship-Driven (28%), Cost-Minimizers (16%)",
                "Growth Seekers valued lending speed above all; 48-hour approval was the key threshold",
                "Simplifiers would pay 15% more for integrated accounting-banking platforms",
                "Relationship-Driven segment showed 4.2x higher product adoption with dedicated banker"
            ]
        },
        {
            "id": "CS-0034",
            "title": "Telecom Say/Do Gap: Customer Service Channel Preferences",
            "client": "Cable & Internet Provider",
            "industry": "Telecom",
            "methodology": "say/do gap",
            "year": 2023,
            "pillar": "optimize",
            "summary": "Study comparing stated customer service channel preferences with actual channel usage and resolution data across chat, phone, app, store, and social media. Found that while 67% of customers stated preference for digital self-service, phone remained the most-used channel at 54% of contacts, primarily because digital channels failed to resolve complex issues.",
            "key_findings": [
                "67% stated preference for digital self-service but 54% of contacts were still phone",
                "Digital channel first-contact resolution was 41% vs. 72% for phone",
                "Customers who failed digital self-service and escalated to phone had 2.1x lower CSAT",
                "Chat satisfaction equaled phone when agents had full account context (achieved in only 31% of chats)"
            ]
        },
        {
            "id": "CS-0035",
            "title": "Automotive Brand Health: Certified Pre-Owned Perception Tracking",
            "client": "Luxury Auto Brand",
            "industry": "Automotive",
            "methodology": "brand health",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Brand perception study for a certified pre-owned (CPO) program measuring quality perceptions, value propositions, and brand halo effects. Compared CPO buyer perceptions against new vehicle buyers and independent used car buyers across trust, quality, and value metrics. CPO buyers demonstrated 91% of the brand attachment of new buyers at 60% of the price point.",
            "key_findings": [
                "CPO buyers showed 91% of new-buyer brand attachment scores",
                "Warranty coverage was the #1 CPO driver, cited by 71% of buyers",
                "CPO-to-new trade-up rate was 44% within 3 years, higher than any other entry path",
                "Dealer CPO experience quality was the strongest predictor of eventual new-car purchase"
            ]
        },
        {
            "id": "CS-0036",
            "title": "Retail Campaign Evaluation: Back-to-School Media Mix Optimization",
            "client": "Mass Merchandise Retailer",
            "industry": "Retail",
            "methodology": "campaign evaluation",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Multi-touch attribution study for a $40M back-to-school campaign evaluating the incremental contribution of each media channel. Used marketing mix modeling combined with geo-matched test/control to isolate channel-level ROI. Found that connected TV and retail media networks delivered the highest incremental ROAS, while traditional print circular was the least efficient at $0.62 per dollar spent.",
            "key_findings": [
                "Connected TV delivered $4.20 incremental ROAS, highest of all channels",
                "Retail media networks drove $3.80 ROAS with strongest bottom-funnel attribution",
                "Print circular delivered $0.62 ROAS but had legacy stakeholder attachment",
                "Shifting 30% of print budget to CTV and RMN projected 18% total campaign ROI improvement"
            ]
        },
        {
            "id": "CS-0037",
            "title": "QSR Say/Do Gap: Loyalty Program Engagement Claims",
            "client": "Coffee QSR Chain",
            "industry": "QSR",
            "methodology": "say/do gap",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Analysis comparing stated loyalty program engagement and satisfaction with actual app usage, redemption patterns, and visit frequency. Found that 83% of enrolled members claimed to 'actively use' the program, but only 34% had redeemed a reward in the past 90 days. Identified 'passive loyalists' as a critical re-engagement opportunity.",
            "key_findings": [
                "83% claimed active usage but only 34% redeemed rewards in past 90 days",
                "Passive loyalists (enrolled, non-redeeming) represented 49% of program membership",
                "Push notification re-engagement campaigns activated 22% of passive members",
                "Point expiration urgency messaging was 3.2x more effective than bonus point offers"
            ]
        },
        {
            "id": "CS-0038",
            "title": "CPG Say/Do Gap: Organic Product Purchase Intent vs. Behavior",
            "client": "Organic Food Brand",
            "industry": "CPG",
            "methodology": "say/do gap",
            "year": 2023,
            "pillar": "optimize",
            "summary": "Panel-based study matching organic food purchase intent surveys with 12 months of grocery purchase data across 8,000 households. Quantified the organic say/do gap at 3.1x (stated organic purchase frequency was 3.1x higher than actual). Income was a weaker predictor of the gap than expected; availability and habitual purchasing explained more variance.",
            "key_findings": [
                "Organic purchase intent overstated by 3.1x vs. actual purchase frequency",
                "Availability (store selection) explained 38% of the gap; income explained only 12%",
                "Habitual brand loyalty to conventional products was the strongest behavioral barrier",
                "Organic conversion was highest in produce (28% of occasions) and lowest in pantry staples (6%)"
            ]
        },
        {
            "id": "CS-0039",
            "title": "Financial Services Sponsorship Valuation: Golf Tournament Partnership",
            "client": "Wealth Management Firm",
            "industry": "Financial Services",
            "methodology": "sponsorship valuation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "ROI assessment of a $8M annual PGA Tour tournament sponsorship focused on high-net-worth audience reach, client entertainment value, and brand prestige metrics. Integrated broadcast exposure measurement, on-site intercepts, and post-event client surveys to quantify the sponsorship's contribution to new client acquisition and existing client retention.",
            "key_findings": [
                "Tournament generated $12.4M in equivalent media value across broadcast and digital",
                "Client hospitality attendees showed 94% retention rate vs. 81% for non-attendees",
                "Brand prestige scores among HNW targets were 27 points higher in tournament DMA",
                "New prospect pipeline from hospitality events valued at $340M in potential AUM"
            ]
        },
        {
            "id": "CS-0040",
            "title": "Telecom Campaign Evaluation: Broadband Speed Upgrade Messaging",
            "client": "Regional ISP",
            "industry": "Telecom",
            "methodology": "campaign evaluation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "A/B/C creative test for broadband speed upgrade campaigns testing three message strategies: speed-focused, reliability-focused, and family-activity-focused. Measured click-through, upgrade conversion, and subscriber value across 800K targeted households. Family-activity messaging outperformed speed claims by 67% on actual upgrade conversion.",
            "key_findings": [
                "Family-activity creative drove 67% higher upgrade conversion than speed claims",
                "Reliability messaging attracted highest-value subscribers with lowest 12-month churn",
                "Speed-focused ads had highest CTR but lowest upgrade conversion, suggesting curiosity clicks",
                "Households with 3+ connected devices showed 2.4x higher upgrade propensity"
            ]
        },
        {
            "id": "CS-0041",
            "title": "Automotive Sponsorship Valuation: Olympic Games Brand Impact",
            "client": "Global Auto Brand",
            "industry": "Automotive",
            "methodology": "sponsorship valuation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Comprehensive valuation of an Olympic Games TOP sponsorship measuring global brand awareness lift, market-by-market consideration impact, and the activation multiplier effect. The $200M+ investment delivered measurable brand lift in 28 of 32 measured markets, with emerging markets showing the strongest relative gains.",
            "key_findings": [
                "Global awareness lifted 7 points; emerging markets saw 14-point average lift",
                "Olympic association increased brand trust scores by 11% globally",
                "Activation spending at 2:1 ratio to rights fees maximized ROI",
                "Digital and social activation delivered 3.8x higher engagement than broadcast alone"
            ]
        },
        {
            "id": "CS-0042",
            "title": "Retail Brand Health: E-Commerce Marketplace Trust Benchmarking",
            "client": "E-Commerce Marketplace",
            "industry": "Retail",
            "methodology": "brand health",
            "year": 2024,
            "pillar": "discover",
            "summary": "Trust and satisfaction benchmarking study comparing the client's marketplace against Amazon, Walmart.com, and category specialists across dimensions of product authenticity, seller reliability, delivery speed, and return ease. Identified that product authenticity concerns were the single largest barrier to category expansion into premium goods.",
            "key_findings": [
                "Product authenticity concern was the #1 barrier for premium goods, cited by 61%",
                "Delivery speed expectations narrowed to 2-day standard, regardless of seller type",
                "Return ease was the strongest driver of repeat purchase across all marketplaces",
                "Seller rating systems were trusted by only 44% of consumers due to review manipulation fears"
            ]
        },
        {
            "id": "CS-0043",
            "title": "QSR Sponsorship Valuation: Sports League Official Partner Assessment",
            "client": "Pizza QSR Brand",
            "industry": "QSR",
            "methodology": "sponsorship valuation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Valuation of 'Official Pizza of the NFL' partnership measuring game-day ordering lift, brand association with football viewing occasions, and promotional activation effectiveness. Found that the partnership drove measurable 14% ordering lift on game days but that the association was strongest among heavy viewers (3+ games/week).",
            "key_findings": [
                "Game-day ordering lift of 14% across digital and phone channels",
                "Heavy viewers (3+ games/week) showed 2.8x stronger brand-occasion association",
                "Co-branded promotional items (game box deals) drove 41% of incremental game-day revenue",
                "Fantasy football integration created year-round brand engagement beyond game days"
            ]
        },
        {
            "id": "CS-0044",
            "title": "CPG Campaign Evaluation: DTC Launch for Legacy CPG Brand",
            "client": "Personal Care Conglomerate",
            "industry": "CPG",
            "methodology": "campaign evaluation",
            "year": 2024,
            "pillar": "confirm",
            "summary": "Launch campaign evaluation for a legacy CPG brand's first direct-to-consumer channel, measuring awareness of the DTC offering, trial driver effectiveness, and channel conflict with retail partners. Found that DTC launch increased total brand sales by 7% with only 2% retail cannibalization, as the DTC channel attracted previously unreachable consumers.",
            "key_findings": [
                "DTC channel drove 7% total brand sales lift with only 2% retail cannibalization",
                "DTC attracted 34% of customers who had never purchased the brand in retail",
                "Subscription model achieved 71% 6-month retention rate among DTC enrollees",
                "Personalization and exclusive products were the top two DTC value propositions"
            ]
        },
        {
            "id": "CS-0045",
            "title": "Financial Services Brand Health: Neobank vs. Traditional Bank Perception",
            "client": "Traditional National Bank",
            "industry": "Financial Services",
            "methodology": "brand health",
            "year": 2024,
            "pillar": "discover",
            "summary": "Competitive brand health tracking study comparing perceptions of traditional banks versus neobanks (Chime, SoFi, etc.) across trust, innovation, fees, and customer experience. Found that traditional banks retained a significant trust advantage among consumers over 40 but were losing the innovation and fee transparency perception battle with younger consumers.",
            "key_findings": [
                "Trust advantage for traditional banks was 34 points among 40+ but only 6 points among 18-29",
                "Neobanks led on fee transparency (+41 points) and innovation (+38 points) overall",
                "Branch access remained important to 62% of consumers despite declining visit frequency",
                "Multi-banked consumers (using both traditional and neo) grew from 18% to 31% in two years"
            ]
        },
        {
            "id": "CS-0046",
            "title": "Telecom Segmentation: Cord-Cutting Journey and Re-bundling Opportunities",
            "client": "Cable & Streaming Provider",
            "industry": "Telecom",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Behavioral segmentation of cord-cutting consumers mapping the unbundling journey and identifying re-bundling opportunities for a converged cable/streaming provider. Found that 'Subscription Fatigued' cord-cutters (37% of segment) were open to re-bundling if pricing was transparent and no-contract.",
            "key_findings": [
                "37% of cord-cutters experienced subscription fatigue managing 4+ streaming services",
                "Average monthly streaming spend reached $61, approaching prior cable bills",
                "Re-bundling consideration was 54% among fatigued cutters if no-contract and transparent",
                "Live sports remained the #1 driver of cable retention and the top re-bundling attraction"
            ]
        },
        {
            "id": "CS-0047",
            "title": "Automotive Campaign Evaluation: Safety-First Brand Repositioning",
            "client": "Scandinavian Auto Brand",
            "industry": "Automotive",
            "methodology": "campaign evaluation",
            "year": 2023,
            "pillar": "confirm",
            "summary": "Campaign effectiveness study for a safety-focused brand repositioning campaign emphasizing advanced safety technology beyond the brand's historical safety reputation. Measured whether consumers could perceive the evolution from 'passive safety' (crash protection) to 'active safety' (crash prevention). Found that tech-forward creative successfully shifted perceptions among younger prospects.",
            "key_findings": [
                "Active safety perception increased 24 points among 25-39 year-olds after campaign",
                "Brand modernity scores improved 18 points without sacrificing core safety equity",
                "Younger prospects were 2.1x more responsive to tech demo video creative than testimonials",
                "Safety remained the #1 brand association but 'innovative safety' replaced 'traditional safety'"
            ]
        },
        {
            "id": "CS-0048",
            "title": "Retail Segmentation: Grocery Delivery vs. In-Store Shopper Profiles",
            "client": "Grocery Chain with Delivery Service",
            "industry": "Retail",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Segmentation comparing grocery delivery adopters, hybrid shoppers, and in-store-only shoppers across demographics, basket composition, price sensitivity, and loyalty. Found that delivery-only shoppers had 23% larger average baskets but 31% lower store brand penetration, representing a significant private label growth opportunity in the digital channel.",
            "key_findings": [
                "Delivery-only baskets were 23% larger but had 31% lower private label penetration",
                "Hybrid shoppers (both delivery and in-store) had 2.4x higher annual spend",
                "Delivery substitution acceptance was the #1 friction point, cited by 48% of users",
                "Store brand recommendations in digital channel increased private label trial by 19%"
            ]
        },
        {
            "id": "CS-0049",
            "title": "CPG Segmentation: Functional Wellness Consumer Taxonomy",
            "client": "Vitamin & Supplement Brand",
            "industry": "CPG",
            "methodology": "segmentation",
            "year": 2024,
            "pillar": "discover",
            "summary": "Comprehensive segmentation of the functional wellness consumer market (vitamins, supplements, functional foods/beverages) identifying six segments based on health motivation, information source trust, and product format preferences. The 'Evidence-Based Optimizer' segment (18% of market) showed the highest LTV and lowest price sensitivity but required clinical-grade messaging.",
            "key_findings": [
                "Six segments identified spanning casual to clinical wellness engagement levels",
                "Evidence-Based Optimizers (18%) had 3.7x higher LTV than average supplement buyer",
                "Social media-influenced buyers (28%) had highest trial but lowest retention",
                "Personalized supplement regimens increased retention by 56% vs. single-product buyers"
            ]
        },
        {
            "id": "CS-0050",
            "title": "QSR Brand Health: Digital Ordering Experience Impact on Brand Perception",
            "client": "Multi-Brand QSR Corporation",
            "industry": "QSR",
            "methodology": "brand health",
            "year": 2024,
            "pillar": "optimize",
            "summary": "Study measuring how digital ordering experience quality (app, web, kiosk, third-party) influenced overall brand health metrics across five QSR brands within a corporate portfolio. Found that app experience accounted for 28% of total brand perception among digital-primary customers, making it the second most influential touchpoint after food quality itself.",
            "key_findings": [
                "App experience explained 28% of brand perception among digital-primary customers",
                "Order accuracy through proprietary app was 97% vs. 88% through third-party aggregators",
                "Third-party ordering eroded brand attribution by 43% compared to owned channels",
                "Kiosk ordering in-store increased average check by 12% through upsell prompts"
            ]
        },
    ]
