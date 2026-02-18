"""
Klaus x IMI — Local Narrative Polisher (Qwen via Ollama)
=========================================================
Takes the deterministic output from survey_bridge.py and uses
Qwen 2.5 32B locally to sharpen narrative headlines, SO WHAT
callouts, and strategic recommendations.

The analytical decisions are ALREADY MADE by the bridge.
Qwen only rewrites text fields — it never touches the numbers.

100% local. Zero API cost.

Usage:
    from survey_bridge import survey_to_slides
    from narrative_polisher import polish_configs
    from deck_renderer import build_deck

    configs = survey_to_slides(survey_data)       # Deterministic analysis
    configs = polish_configs(configs, survey_data) # Qwen sharpens text
    build_deck(configs, "output.pptx")             # Render slides
"""

import json
import httpx
from typing import Optional

OLLAMA_URL = "http://localhost:11434"
MODEL = "klaus-imi"  # Your Modelfile-based model, or "qwen2.5:32b"


def _ask_qwen(prompt: str, temperature: float = 0.4, timeout: float = 60.0) -> str:
    """Single synchronous call to Qwen via Ollama."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{OLLAMA_URL}/api/generate", json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 8192}
            })
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        print(f"[WARN] Qwen call failed: {e}")
    return ""


def _polish_headline(draft_headline: str, slide_context: str) -> str:
    """
    Ask Qwen to rewrite a single headline. Constrained task.
    If Qwen fails or returns garbage, keep the draft.
    """
    prompt = f"""Rewrite this slide headline to be more compelling. It should read like a senior consultant's insight, not a data label.

RULES:
- Keep it under 15 words
- It must be an INSIGHT, not a description
- Use one of these patterns: tension ("X leads — but Y"), quantified gap ("Xpt gap separates A from B"), strategic label ("The [Name]: what it means"), or action ("For [audience], X is the play")
- Do NOT change any numbers
- Return ONLY the rewritten headline, nothing else

CONTEXT: {slide_context}

DRAFT HEADLINE: {draft_headline}

REWRITTEN HEADLINE:"""

    result = _ask_qwen(prompt, temperature=0.5)

    # Validation: must be reasonable length, not empty, not a refusal
    if result and 3 < len(result) < 120 and not result.lower().startswith(("i ", "sure", "here")):
        # Strip quotes if Qwen wrapped it
        result = result.strip('"\'')
        return result

    return draft_headline  # Keep draft if Qwen returns garbage


def _polish_so_what(draft: str, data_context: str) -> str:
    """Sharpen a SO WHAT callout."""
    if not draft:
        return draft

    prompt = f"""Rewrite this strategic insight to be sharper and more actionable. Write as if advising a CMO who needs to make a sponsorship decision this week.

RULES:
- Keep it under 40 words
- Must connect the data to a specific business action
- Do NOT change any numbers
- Return ONLY the rewritten text

DATA CONTEXT: {data_context}

DRAFT: {draft}

REWRITTEN:"""

    result = _ask_qwen(prompt, temperature=0.4)
    if result and 10 < len(result) < 300 and not result.lower().startswith(("i ", "sure", "here")):
        return result.strip('"\'')
    return draft


def _polish_recommendations(recs: list, survey_summary: str) -> list:
    """Sharpen the 4 strategic recommendation cards."""
    if not recs:
        return recs

    prompt = f"""You are a senior IMI consultant. Rewrite these 4 strategic recommendations to be sharper, more specific, and more actionable. Each recommendation must reference specific numbers from the data.

RULES:
- Each title should be 2-5 words, action-oriented
- Each body should be 2-3 sentences max
- Reference specific percentages and gaps from the data
- Tell the client exactly what to DO
- Return ONLY a JSON array of 4 objects with "title" and "body" fields
- No markdown, no explanation

DATA SUMMARY: {survey_summary}

CURRENT RECOMMENDATIONS:
{json.dumps(recs, indent=2)}

REWRITTEN JSON ARRAY:"""

    result = _ask_qwen(prompt, temperature=0.4, timeout=90.0)

    try:
        # Try to parse as JSON
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

        parsed = json.loads(result)
        if isinstance(parsed, list) and len(parsed) >= 4:
            # Validate each rec has title and body
            valid = all("title" in r and "body" in r for r in parsed[:4])
            if valid:
                return parsed[:4]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return recs  # Keep originals if parsing fails


def _build_survey_summary(survey_data: dict, configs: dict) -> str:
    """Build a brief text summary for context in Qwen prompts."""
    stats = configs.get("slide_2", {}).get("stats", [])
    stat_text = "; ".join(f"{s['number']}: {s['description'][:80]}" for s in stats[:3])

    national = configs.get("slide_3", {}).get("data", {})
    top_3 = list(national.items())[:3]
    top_text = ", ".join(f"{k} at {v}%" for k, v in top_3)

    return f"Survey: {survey_data.get('survey_name', 'Unknown')} (n={survey_data.get('total_n', 0)}). Top 3 nationally: {top_text}. Key stats: {stat_text}"


# ═══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════

def polish_configs(
    configs: dict,
    survey_data: dict,
    skip_if_offline: bool = True
) -> dict:
    """
    Take slide configs from survey_bridge and polish narrative fields
    using local Qwen. Falls back gracefully if Ollama is unreachable.

    Polishes: slide titles, SO WHAT callouts, recommendations.
    Never touches: data values, chart configs, source lines.
    """

    # Quick connectivity check
    if skip_if_offline:
        try:
            with httpx.Client(timeout=3.0) as c:
                c.get(f"{OLLAMA_URL}/api/tags")
        except Exception:
            print("[INFO] Ollama not reachable — skipping narrative polish")
            return configs

    summary = _build_survey_summary(survey_data, configs)

    # ─── Polish slide headlines ───────────────────────────────

    headline_slides = {
        "slide_3": "National results ranking",
        "slide_4": "Combined/second preference",
        "slide_5": "Biggest demographic divide",
        "slide_6": "Age/generational differences",
        "slide_7": "Gender and regional patterns",
        "slide_8": "Aspirational/dream preference",
        "slide_9": "Income-based segmentation",
    }

    for slide_key, context in headline_slides.items():
        if slide_key in configs and "title" in configs[slide_key]:
            original = configs[slide_key]["title"]
            # Don't polish all-caps titles (THE NATIVITY DIVIDE, STRATEGIC IMPLICATIONS)
            if original.isupper():
                continue
            data_hint = json.dumps(configs[slide_key].get("data", {}))[:200]
            full_context = f"{context}. {summary}. Top items: {data_hint}"
            polished = _polish_headline(original, full_context)
            if polished != original:
                configs[slide_key]["title"] = polished
                print(f"  [POLISHED] {slide_key}: '{original}' → '{polished}'")

    # ─── Polish SO WHAT callouts ──────────────────────────────

    for slide_key in ["slide_5", "slide_7"]:
        if slide_key in configs and "so_what" in configs[slide_key]:
            original = configs[slide_key]["so_what"]
            data_hint = json.dumps(configs[slide_key].get("left_data", configs[slide_key].get("data", {})))[:200]
            polished = _polish_so_what(original, f"{summary}. {data_hint}")
            if polished != original:
                configs[slide_key]["so_what"] = polished

    # ─── Polish strategic recommendations ─────────────────────

    if "slide_10" in configs and "recommendations" in configs["slide_10"]:
        original_recs = configs["slide_10"]["recommendations"]
        polished_recs = _polish_recommendations(original_recs, summary)
        if polished_recs != original_recs:
            configs["slide_10"]["recommendations"] = polished_recs
            print(f"  [POLISHED] slide_10: recommendations sharpened")

    # ─── Polish exec summary subtitle ─────────────────────────

    if "slide_2" in configs:
        original_sub = configs["slide_2"].get("exec_subtitle", "The Essential Insight")
        prompt = f"""Write a 3-5 word thematic label that captures the main story of this survey data. Examples: "The Nativity Imperative", "A Contested Landscape", "Two Canadas, One Sport". Return ONLY the label.

DATA: {summary}

LABEL:"""
        result = _ask_qwen(prompt, temperature=0.5)
        if result and 2 < len(result.split()) < 8:
            configs["slide_2"]["exec_subtitle"] = result.strip('"\'')

    return configs


# ═══════════════════════════════════════════════════════════════
# COMPLETE PIPELINE (convenience function)
# ═══════════════════════════════════════════════════════════════

def generate_polished_deck(survey_data: dict, output_path: str = "output.pptx") -> str:
    """
    Full pipeline: parse → analyze → polish → render.
    100% local. Zero API cost.
    """
    from survey_bridge import survey_to_slides
    from deck_renderer import build_deck

    print("[1/3] Analyzing survey data...")
    configs = survey_to_slides(survey_data)

    print("[2/3] Polishing narratives with Qwen...")
    configs = polish_configs(configs, survey_data)

    print("[3/3] Rendering deck...")
    build_deck(configs, output_path)

    print(f"Done → {output_path}")
    return output_path
