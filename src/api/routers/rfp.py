"""
RFP Automation for Klaus IMI Platform.
Generate structured RFP responses and evaluate incoming RFPs.
"""

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from config import OLLAMA_BASE, IMI_MODEL

import httpx

logger = logging.getLogger("imi_rfp")

router = APIRouter(prefix="/klaus/imi", tags=["imi-rfp"])

# ---------------------------------------------------------------------------
# Ollama helper (sync, matching imi_agents pattern)
# ---------------------------------------------------------------------------


def _call_ollama(system: str, user: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Synchronous Ollama call for RFP generation."""
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": IMI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens},
        },
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = """You are an RFP response writer for IMI International — a 54-year-old global marketing research & consulting firm (founded 1971, HQ Toronto, operating in 45+ countries with 50,000+ completed case studies).

IMI's methodology is built on the Three-Pillar Service Model — IMI Pinpoint:
1. DISCOVER — Foundational insight (segmentation, messaging, channel optimization) powered by IMI Pulse™
2. CONFIRM — Validation & quantification (concept testing, campaign evaluation, sponsorship valuation) benchmarked against normative databases
3. OPTIMIZE — Performance measurement & ROI (campaign tracking, cost effectiveness, competitive benchmarking)

Generate a structured RFP response with these exact sections. Output valid JSON only with this structure:
{
  "overview": "Company overview section text",
  "methodology": "Methodology approach section text",
  "experience": "Relevant experience section text",
  "team": "Team structure section text",
  "timeline": "Timeline & deliverables section text",
  "investment": "Investment section text"
}

Guidelines:
- Leverage IMI's 54-year track record and global presence
- Reference relevant Three-Pillar capabilities based on the project requirements
- Cite normative databases and 50,000+ case studies as differentiators
- Propose realistic timelines based on methodology complexity
- Keep each section 2-4 paragraphs, professional and compelling
- Investment section should outline fee structure categories without specific dollar amounts unless requirements dictate otherwise"""

_EVALUATE_SYSTEM = """You are an RFP analyst for IMI International — a 54-year-old global marketing research & consulting firm (founded 1971, HQ Toronto, 45+ countries, 50,000+ case studies).

IMI's Three-Pillar Service Model — IMI Pinpoint:
1. DISCOVER — Foundational insight (segmentation, messaging, channel optimization) powered by IMI Pulse™
2. CONFIRM — Validation & quantification (concept testing, campaign evaluation, sponsorship valuation)
3. OPTIMIZE — Performance measurement & ROI (campaign tracking, cost effectiveness, competitive benchmarking)

Analyze the incoming RFP text and extract structured intelligence. Output valid JSON only:
{
  "requirements": ["requirement 1", "requirement 2", ...],
  "suggested_methodology": "Recommended approach based on IMI capabilities",
  "relevant_pillars": ["discover", "confirm", and/or "optimize"],
  "estimated_scope": "Small / Medium / Large / Enterprise with brief justification"
}

Guidelines:
- Extract every explicit and implicit requirement from the RFP
- Map requirements to IMI's Three-Pillar model
- Suggest methodology that leverages IMI's normative databases and proprietary tools
- Scope estimation should consider timeline, geography, and complexity"""

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RFPGenerateRequest(BaseModel):
    requirements: str
    client_name: str
    industry: str


class RFPSections(BaseModel):
    overview: str
    methodology: str
    experience: str
    team: str
    timeline: str
    investment: str


class RFPGenerateResponse(BaseModel):
    sections: RFPSections
    full_text: str
    error: str | None = None


class RFPEvaluateRequest(BaseModel):
    rfp_text: str


class RFPEvaluateResponse(BaseModel):
    requirements: list[str]
    suggested_methodology: str
    relevant_pillars: list[str]
    estimated_scope: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/rfp/generate", response_model=RFPGenerateResponse)
def rfp_generate(req: RFPGenerateRequest):
    """Generate a structured RFP response for a prospective client."""
    user_prompt = (
        f"Client: {req.client_name}\n"
        f"Industry: {req.industry}\n"
        f"Requirements:\n{req.requirements}\n\n"
        "Generate the RFP response JSON now."
    )

    try:
        raw = _call_ollama(_GENERATE_SYSTEM, user_prompt, temperature=0.3, max_tokens=4096)
    except Exception as e:
        logger.error("Ollama error in rfp/generate: %s", e)
        return RFPGenerateResponse(
            sections=RFPSections(overview="", methodology="", experience="", team="", timeline="", investment=""),
            full_text="",
            error=f"LLM error: {e}",
        )

    # Parse JSON from response
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON object found in response")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("JSON parse error in rfp/generate: %s", e)
        return RFPGenerateResponse(
            sections=RFPSections(overview="", methodology="", experience="", team="", timeline="", investment=""),
            full_text=raw,
            error=f"Parse error: {e}",
        )

    sections = RFPSections(
        overview=data.get("overview", ""),
        methodology=data.get("methodology", ""),
        experience=data.get("experience", ""),
        team=data.get("team", ""),
        timeline=data.get("timeline", ""),
        investment=data.get("investment", ""),
    )

    full_text = (
        f"# RFP Response — {req.client_name}\n\n"
        f"## Company Overview\n{sections.overview}\n\n"
        f"## Methodology Approach\n{sections.methodology}\n\n"
        f"## Relevant Experience\n{sections.experience}\n\n"
        f"## Team Structure\n{sections.team}\n\n"
        f"## Timeline & Deliverables\n{sections.timeline}\n\n"
        f"## Investment\n{sections.investment}"
    )

    return RFPGenerateResponse(sections=sections, full_text=full_text)


@router.post("/rfp/evaluate", response_model=RFPEvaluateResponse)
def rfp_evaluate(req: RFPEvaluateRequest):
    """Analyze an incoming RFP and extract requirements with suggested approach."""
    user_prompt = f"Analyze this RFP:\n\n{req.rfp_text}\n\nExtract requirements and suggest approach as JSON."

    try:
        raw = _call_ollama(_EVALUATE_SYSTEM, user_prompt, temperature=0.1, max_tokens=2048)
    except Exception as e:
        logger.error("Ollama error in rfp/evaluate: %s", e)
        return RFPEvaluateResponse(
            requirements=[], suggested_methodology="", relevant_pillars=[], estimated_scope="", error=f"LLM error: {e}"
        )

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON object found in response")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("JSON parse error in rfp/evaluate: %s", e)
        return RFPEvaluateResponse(
            requirements=[], suggested_methodology="", relevant_pillars=[], estimated_scope="", error=f"Parse error: {e}"
        )

    return RFPEvaluateResponse(
        requirements=data.get("requirements", []),
        suggested_methodology=data.get("suggested_methodology", ""),
        relevant_pillars=data.get("relevant_pillars", []),
        estimated_scope=data.get("estimated_scope", ""),
    )
