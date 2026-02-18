"""
IMI Report Templates — pre-built structures for common IMI deliverables.
Each template defines sections, prompts, and data requirements.
"""

TEMPLATES = {
    "brand_health": {
        "name": "Brand Health Report",
        "description": "Comprehensive brand health analysis: funnel metrics, NPS, competitive positioning",
        "sections": [
            {"title": "Executive Summary", "prompt": "Provide a 2-3 sentence executive summary of the brand's overall health based on the data. Lead with the most surprising finding."},
            {"title": "Brand Health Funnel", "prompt": "Analyze the brand health funnel: awareness → familiarity → consideration → purchase → recommendation. Where is the biggest drop-off? What does it mean?"},
            {"title": "NPS & Loyalty Analysis", "prompt": "Deep dive into NPS scores and loyalty metrics. How does this brand compare to competitors? What's driving promoters vs detractors?"},
            {"title": "Competitive Positioning", "prompt": "Map this brand's position relative to competitors on key metrics. Who is the biggest threat and why?"},
            {"title": "SO WHAT — Recommendations", "prompt": "Provide 3-5 specific, actionable recommendations tied to IMI's 12 Growth Levers. Each should follow: 'If you do X, the data suggests Y because Z.'"}
        ],
        "datasets": ["brand_health_tracker_Q4_2025.csv", "competitive_benchmark_overall.csv"],
        "rag_query": "brand health NPS awareness loyalty trust consideration purchase"
    },
    "campaign_evaluation": {
        "name": "Campaign Evaluation Report",
        "description": "Pre/post campaign metrics, ROI analysis, awareness lift measurement",
        "sections": [
            {"title": "Executive Summary", "prompt": "Summarize campaign performance in 2-3 sentences. Was it effective?"},
            {"title": "Awareness & Recall", "prompt": "Analyze unaided and aided awareness changes. What was the awareness lift?"},
            {"title": "Engagement Metrics", "prompt": "Analyze engagement, consideration, and purchase intent changes attributable to the campaign."},
            {"title": "ROI Analysis", "prompt": "Calculate and interpret the campaign ROI. Compare to industry benchmarks and IMI's normative database."},
            {"title": "SO WHAT — Optimization", "prompt": "What should be optimized for the next campaign? Provide 3 specific recommendations with expected impact."}
        ],
        "datasets": ["promotion_roi_analysis_2025.csv", "brand_health_tracker_Q4_2025.csv"],
        "rag_query": "campaign ROI promotion effectiveness awareness lift"
    },
    "segmentation": {
        "name": "Segmentation Report",
        "description": "Consumer segment profiles, sizing, value assessment, targeting recommendations",
        "sections": [
            {"title": "Executive Summary", "prompt": "Summarize the key segments identified and the single most important targeting insight."},
            {"title": "Segment Profiles", "prompt": "Profile each segment: demographics, behaviors, attitudes, media consumption. Use tables for clarity."},
            {"title": "Segment Sizing & Value", "prompt": "Size each segment (% of population, estimated revenue potential). Which segments are most valuable?"},
            {"title": "Purchase Drivers by Segment", "prompt": "What drives purchase decisions in each segment? How do drivers differ across segments?"},
            {"title": "SO WHAT — Targeting Strategy", "prompt": "Which segments should be prioritized and why? Provide specific targeting recommendations with channel suggestions."}
        ],
        "datasets": ["genz_lifestyle_segmentation.csv", "purchase_drivers_by_generation_Q4_2025.csv", "consumer_sentiment_survey_canada_jan2026.csv"],
        "rag_query": "segmentation demographics generation purchase drivers lifestyle"
    },
    "sponsorship_valuation": {
        "name": "Sponsorship Valuation Report",
        "description": "Property scores, ROI analysis, media equivalency, sponsorship recommendations",
        "sections": [
            {"title": "Executive Summary", "prompt": "Summarize the sponsorship portfolio's overall ROI and the single highest-value property."},
            {"title": "Property Scorecard", "prompt": "Score each sponsorship property on: awareness lift, brand association, purchase intent, media value. Use a table."},
            {"title": "ROI by Property", "prompt": "Calculate ROI for each sponsorship. Which properties deliver the best return per dollar?"},
            {"title": "Audience Alignment", "prompt": "How well does each property's audience align with the brand's target segments?"},
            {"title": "SO WHAT — Portfolio Optimization", "prompt": "Which sponsorships to keep, drop, or increase investment in? Apply IMI's experiential marketing 10x multiplier where relevant."}
        ],
        "datasets": ["sponsorship_property_scores.csv", "sports_viewership_crosstab_2025.csv"],
        "rag_query": "sponsorship property ROI sports viewership media value"
    },
    "competitive_landscape": {
        "name": "Competitive Landscape Report",
        "description": "Market share analysis, brand positioning, competitive threats, strategic implications",
        "sections": [
            {"title": "Executive Summary", "prompt": "Summarize the competitive landscape in 2-3 sentences. Who is winning and why?"},
            {"title": "Market Share Analysis", "prompt": "Analyze market share data with YoY changes. Who is gaining, who is losing? Show in a table."},
            {"title": "Brand Positioning Map", "prompt": "Map brands on key dimensions (awareness vs trust, NPS vs market share). Identify positioning gaps and opportunities."},
            {"title": "Generational Differences", "prompt": "How does competitive positioning differ across generations? Where are the biggest generational gaps?"},
            {"title": "SO WHAT — Strategic Response", "prompt": "What are the top 3 competitive threats and how should the brand respond? Tie to specific Growth Levers."}
        ],
        "datasets": ["competitive_benchmark_overall.csv", "competitive_benchmark_by_generation.csv", "competitive_benchmark_market_share.csv"],
        "rag_query": "competitive market share brand positioning awareness NPS generation"
    },
    "saydo_gap": {
        "name": "Say/Do Gap Analysis",
        "description": "IMI's signature analysis: stated intentions vs actual behavior, gap magnitude, implications",
        "sections": [
            {"title": "Executive Summary", "prompt": "Summarize the biggest Say/Do gaps found. What's the single most surprising discrepancy between what consumers say and what they do?"},
            {"title": "Gap Magnitude Table", "prompt": "Present a table of stated intention vs actual behavior for each measured category. Calculate the gap percentage. Bold the largest gaps."},
            {"title": "Category Deep Dive", "prompt": "For the top 3 largest gaps: why does the gap exist? What psychological, economic, or situational factors explain it?"},
            {"title": "Implications for Messaging", "prompt": "How should marketing messaging change based on Say/Do gap insights? What claims should brands stop making?"},
            {"title": "SO WHAT — Closing the Gap", "prompt": "Provide 3 specific strategies to close the Say/Do gap, each tied to a Growth Lever. Focus on proof-based messaging over aspirational."}
        ],
        "datasets": ["say_do_gap_food_beverage.csv"],
        "rag_query": "say do gap behavioral stated intention actual behavior"
    }
}


def get_template(template_type: str) -> dict | None:
    """Get a report template by type."""
    return TEMPLATES.get(template_type)


def list_templates() -> list[dict]:
    """List all available templates with name and description."""
    return [
        {"type": k, "name": v["name"], "description": v["description"]}
        for k, v in TEMPLATES.items()
    ]
