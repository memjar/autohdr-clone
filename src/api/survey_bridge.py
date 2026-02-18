"""
Klaus x IMI — Survey-to-Slides Bridge
======================================
Parses survey data (dict from pandas) and auto-generates the
slide configuration JSON that feeds into deck_renderer.build_deck().

This is the INTELLIGENCE layer — it decides what goes on each slide
based on what the data contains, following the IMI analytical framework.

Usage:
    from survey_bridge import survey_to_slides
    from deck_renderer import build_deck

    configs = survey_to_slides(survey_data)
    build_deck(configs, "output.pptx")
"""

import json
import re
from typing import Optional


# ─── Label Cleaning ─────────────────────────────────────────
# Raw xlsx labels often have case issues, extra whitespace, or artifacts.

_LABEL_FIXES = {
    "stanley cup playoffs - nhl": "Stanley Cup NHL",
    "stanley cup - nhl": "Stanley Cup NHL",
    "stanley cup nhl": "Stanley Cup NHL",
    "fifa world cup": "FIFA World Cup",
    "winter olympics": "Winter Olympics",
    "summer olympics": "Summer Olympics",
    "nba finals": "NBA Finals",
    "nfl super bowl": "NFL Super Bowl",
    "nascar": "NASCAR",
    "cfl grey cup": "CFL Grey Cup",
}


def _clean_label(raw: str) -> str:
    """Normalize a survey option label — fix case, trim whitespace, apply known fixes."""
    cleaned = re.sub(r'\s+', ' ', raw.strip())
    lookup = cleaned.lower()
    if lookup in _LABEL_FIXES:
        return _LABEL_FIXES[lookup]
    # Fix common patterns: "NhL" → "NHL", "Nba" → "NBA"
    cleaned = re.sub(r'\bNhL\b', 'NHL', cleaned)
    cleaned = re.sub(r'\bNba\b', 'NBA', cleaned)
    cleaned = re.sub(r'\bNfl\b', 'NFL', cleaned)
    cleaned = re.sub(r'\bNascar\b', 'NASCAR', cleaned)
    cleaned = re.sub(r'\bCfl\b', 'CFL', cleaned)
    cleaned = re.sub(r'\bFifa\b', 'FIFA', cleaned)
    return cleaned


def _detect_survey_name(survey_data: dict) -> str:
    """
    Infer a proper survey name from the data content rather than relying on filename.
    Falls back to the provided survey_name if no topic can be detected.
    """
    raw_name = survey_data.get("survey_name", "")
    questions = survey_data.get("questions", [])

    # Collect all option labels across questions
    all_labels = []
    for q in questions:
        for opt in q.get("options", []):
            all_labels.append(opt.get("label", "").lower())
    label_text = " ".join(all_labels)

    # Detect topic from option content
    topic = None
    if any(kw in label_text for kw in ["fifa", "nhl", "olympics", "nba", "nfl", "stanley cup", "nascar"]):
        topic = "Sports Fandom"
    elif any(kw in label_text for kw in ["brand", "purchase", "buy", "prefer"]):
        topic = "Brand Preference"
    elif any(kw in label_text for kw in ["vote", "party", "liberal", "conservative", "ndp"]):
        topic = "Political Sentiment"
    elif any(kw in label_text for kw in ["climate", "environment", "carbon"]):
        topic = "Environmental Attitudes"

    if topic:
        # Detect country from segments
        segments = survey_data.get("segments", [])
        seg_text = " ".join(s.lower() for s in segments)
        if "canada" in seg_text or "ontario" in seg_text or "quebec" in seg_text:
            country = "Canadian"
        elif "australia" in seg_text or "nsw" in seg_text:
            country = "Australian"
        else:
            country = ""

        return f"{country} {topic} Pulse 2025".strip()

    # Fallback: clean up the filename-based name
    cleaned = raw_name.replace("-", " ").replace("_", " ").strip()
    # Title-case it if it looks like a slug
    if cleaned == cleaned.lower():
        cleaned = cleaned.title()
    return cleaned if cleaned else "IMI Pulse™ Survey"


def survey_to_slides(survey_data: dict) -> dict:
    """
    Convert parsed survey JSON into slide configs for deck_renderer.

    survey_data format (from parse_crosstab):
    {
        "survey_name": "...",
        "total_n": 503,
        "questions": [
            {
                "id": "Q36",
                "text": "If you could only watch ONE...",
                "options": [
                    {"label": "FIFA World Cup", "values": {"Canada": 27, "Male": 28, ...}},
                    ...
                ],
                "base_sizes": {"Canada": 503, "Male": 241, ...}
            }
        ],
        "segments": ["Canada", "Male", "Female", ...]
    }
    """
    # Clean labels in-place before processing
    for q in survey_data.get("questions", []):
        for opt in q.get("options", []):
            opt["label"] = _clean_label(opt.get("label", ""))

    questions = survey_data.get("questions", [])
    name = _detect_survey_name(survey_data)
    total_n = survey_data.get("total_n", 0)
    source = f"Source: IMI Pulse™ | {name} | n = {total_n}"

    q_main = questions[0] if len(questions) > 0 else None
    q_second = questions[1] if len(questions) > 1 else None
    q_dream = questions[2] if len(questions) > 2 else None

    # ─── Helpers ─────────────────────────────────────────────

    def _fmt(v):
        """Format a value: 27.0 → 27, 27.5 → 27.5"""
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v

    def _national_sorted(q):
        """Get national results sorted descending."""
        if not q:
            return {}
        results = {}
        for opt in q.get("options", []):
            vals = opt.get("values", {})
            nat = None
            for k in ["Canada", "Total", "National", "NET"]:
                if k in vals:
                    nat = vals[k]; break
            if nat is None and vals:
                nat = list(vals.values())[0]
            if nat is not None:
                results[opt["label"]] = _fmt(round(float(nat), 1))
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def _seg_val(q, option_label, segment_key):
        """Get value for a specific option × segment."""
        if not q:
            return 0
        for opt in q.get("options", []):
            if opt["label"] == option_label:
                vals = opt.get("values", {})
                # First try exact match
                for k, v in vals.items():
                    if k.lower() == segment_key.lower():
                        return _fmt(round(float(v), 1))
                # Then try starts-with match (for "Born in Canada" matching "born in")
                for k, v in vals.items():
                    if k.lower().startswith(segment_key.lower()):
                        return _fmt(round(float(v), 1))
                # Finally try contains, but with word boundary awareness
                # Avoid "male" matching "female" by checking it's not preceded by 'fe'
                sk = segment_key.lower()
                for k, v in vals.items():
                    kl = k.lower()
                    idx = kl.find(sk)
                    if idx >= 0:
                        # Check it's not a substring of another word
                        if idx == 0 or not kl[idx-1].isalpha():
                            return _fmt(round(float(v), 1))
        return 0

    def _seg_vals_for_option(q, option_label, segment_keys):
        """Get {display_label: value} for multiple segments of one option."""
        result = {}
        if not q:
            return result
        for opt in q.get("options", []):
            if opt["label"] == option_label:
                for seg_key, display in segment_keys:
                    for k, v in opt.get("values", {}).items():
                        if seg_key.lower() in k.lower():
                            result[display] = round(float(v), 1)
                            break
        return result

    def _base(q, segment_key):
        """Get base size for a segment."""
        if not q:
            return 0
        for k, v in q.get("base_sizes", {}).items():
            if segment_key.lower() in k.lower():
                return int(v)
        return 0

    def _detect_segments(q, keywords):
        """Auto-detect which segment keys exist in the data."""
        if not q or not q.get("options"):
            return []
        sample_vals = q["options"][0].get("values", {})
        found = []
        for kw, display in keywords:
            for k in sample_vals.keys():
                if kw.lower() in k.lower():
                    found.append((kw, display, k))
                    break
        return found

    # ─── Core data extraction ────────────────────────────────

    national = _national_sorted(q_main)
    top_items = list(national.items())

    # Detect available segment types
    age_segs = _detect_segments(q_main, [
        ("18 to 34", "18-34"), ("18-34", "18-34"),
        ("35 to 54", "35-54"), ("35-54", "35-54"),
        ("55 and", "55+"), ("55+", "55+"), ("55-64", "55-64"),
    ])
    gender_segs = _detect_segments(q_main, [
        ("female", "Female"), ("male", "Male"),
    ])
    nativity_segs = _detect_segments(q_main, [
        ("born in", "Born in Canada"), ("born outside", "Born Outside"),
    ])
    income_segs = _detect_segments(q_main, [
        ("under $50", "<$50K"), ("under $", "<$50K"),
        ("$50", "$50-125K"), ("$125", "$125K+"),
    ])
    region_segs = _detect_segments(q_main, [
        ("west", "West"), ("ontario", "Ontario"),
        ("quebec", "Quebec"), ("atlantic", "Atlantic"),
        ("british", "BC"), ("alberta", "Alberta"),
        ("prairies", "Prairies"),
    ])

    configs = {}

    # ═══════════════════════════════════════════════════════════
    # SLIDE 1: TITLE
    # ═══════════════════════════════════════════════════════════

    # Auto-detect topic from option labels
    topic_hint = "Consumer Intelligence Insights"
    option_labels = [o["label"] for o in (q_main or {}).get("options", [])]
    sport_keywords = ["FIFA", "NHL", "Olympics", "NBA", "NFL", "NASCAR", "CFL"]
    if any(kw.lower() in " ".join(option_labels).lower() for kw in sport_keywords):
        topic_hint = "What Canadians Watch — And What It Means for Sponsors"

    seg_list = []
    if age_segs: seg_list.append("age")
    if gender_segs: seg_list.append("gender")
    if region_segs: seg_list.append("region")
    if income_segs: seg_list.append("income")
    if nativity_segs: seg_list.append("nativity")

    configs["slide_1"] = {
        "title": name,
        "subtitle": topic_hint,
        "methodology": f"n = {total_n} | Cross-tabulated by {', '.join(seg_list)}" if seg_list else f"n = {total_n}"
    }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 2: EXECUTIVE SUMMARY (3 stat cards)
    # ═══════════════════════════════════════════════════════════

    stats = []

    # Stat 1: Combined preference for #1 (if Q2 exists) or standalone #1
    if q_second and top_items:
        national_q2 = _national_sorted(q_second)
        combined_top_val = round(top_items[0][1] + national_q2.get(top_items[0][0], 0), 1)
        stats.append({
            "number": f"{_fmt(combined_top_val)}%",
            "description": f"combined 1st + 2nd choice for {top_items[0][0]} — the broadest reach play in the data."
        })
    elif top_items:
        stats.append({
            "number": f"{top_items[0][1]}%",
            "description": f"of Canadians choose {top_items[0][0]} as #1 — leading the field nationally."
        })

    # Stat 2: Biggest nativity/demographic gap (foreign-born stat if available)
    if q_main:
        # Try to find foreign-born standout first
        foreign_stat_found = False
        for opt in q_main.get("options", []):
            vals = opt.get("values", {})
            for k, v in vals.items():
                if "born outside" in k.lower() or "foreign" in k.lower():
                    if isinstance(v, (int, float)) and v > 40:
                        stats.append({
                            "number": f"{_fmt(round(float(v)))}%",
                            "description": f"of foreign-born Canadians choose {opt['label']} — revealing the deepest cultural passion point in the data."
                        })
                        foreign_stat_found = True
                        break
            if foreign_stat_found:
                break

        if not foreign_stat_found:
            # Fallback: biggest gap
            max_gap = 0
            max_gap_desc = ""
            for opt in q_main.get("options", []):
                label = opt["label"]
                vals = opt.get("values", {})
                all_vals = [v for v in vals.values() if isinstance(v, (int, float)) and v > 0]
                if len(all_vals) >= 2:
                    gap = max(all_vals) - min(all_vals)
                    if gap > max_gap:
                        max_gap = gap
                        max_seg = max(vals.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0)
                        max_gap_desc = f"gap on {label}: {round(max_seg[1])}% ({max_seg[0]}). The single most dramatic divide in the data."
            if max_gap > 0:
                stats.append({
                    "number": f"{round(max_gap)}pt",
                    "description": max_gap_desc
                })

    # Stat 3: Championship / head-to-head stat
    if len(top_items) >= 2:
        gap_1_2 = round(top_items[0][1] - top_items[1][1])
        if gap_1_2 <= 10 and len(top_items) >= 3:
            spread = round(top_items[0][1] - top_items[2][1])
            stats.append({
                "number": f"{top_items[1][1]}%",
                "description": f"choose {top_items[1][0]} as runner-up — only {gap_1_2}pts behind. The top 3 are within {spread}pts, making this a contested landscape."
            })
        else:
            stats.append({
                "number": f"{top_items[1][1]}%",
                "description": f"choose {top_items[1][0]} — {gap_1_2}pts behind {top_items[0][0]}. The gap signals a clear leader but competitive second tier."
            })

    configs["slide_2"] = {
        "stats": stats[:3],
        "source": source
    }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 3: NATIONAL RESULTS (bar chart)
    # ═══════════════════════════════════════════════════════════

    headline_3 = f"{top_items[0][0]} Leads" if top_items else "National Results"
    if len(top_items) >= 3:
        spread = round(top_items[0][1] - top_items[2][1])
        if spread <= 10:
            headline_3 += f" — But the Race Is Tighter Than You Think"
        else:
            headline_3 += f" — With a Commanding {round(top_items[0][1] - top_items[1][1])}pt Lead"

    configs["slide_3"] = {
        "title": headline_3,
        "subtitle": f"{q_main.get('text', '')} | National (n={total_n})" if q_main else "",
        "data": national,
        "insight": f"The top three are within {round(top_items[0][1] - top_items[2][1]) if len(top_items) >= 3 else '—'}pts. No single event commands majority preference." if len(top_items) >= 3 else "",
        "source": source
    }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 4: COMBINED PREFERENCE (if second question exists)
    # ═══════════════════════════════════════════════════════════

    if q_second:
        national_q2 = _national_sorted(q_second)
        combined = {}
        for label, val in national.items():
            combined[label] = round(val + national_q2.get(label, 0), 1)
        combined = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
        combined_top = list(combined.items())

        reshuffle = list(combined.keys())[:3] != list(national.keys())[:3]
        headline_4 = "Combined Reach Reshuffles the Rankings" if reshuffle else "Combined Preference Confirms the Pattern"

        configs["slide_4"] = {
            "title": headline_4,
            "subtitle": "1st + 2nd choice combined preference | National",
            "data": combined,
            "insight": "Reach and passion are different metrics. The events with broadest total interest may not be the ones driving deepest engagement.",
            "source": source
        }
    else:
        # No second question — show detailed breakdown of Q1
        configs["slide_4"] = {
            "title": "Preference Distribution — The Full Picture",
            "subtitle": f"National results (n={total_n})",
            "data": national,
            "source": source
        }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 5: NATIVITY DIVIDE (if nativity segments exist)
    # ═══════════════════════════════════════════════════════════

    if nativity_segs and q_main:
        left_data = {}
        right_data = {}
        born_in_key = None
        born_out_key = None

        for kw, display, actual_key in nativity_segs:
            if "born in" in kw.lower() or "canadian" in kw.lower():
                born_in_key = kw
            elif "born outside" in kw.lower() or "foreign" in kw.lower() or "immigrant" in kw.lower():
                born_out_key = kw

        if born_in_key and born_out_key:
            # Use combined 1st+2nd choice if second question exists (matches Claude reference)
            for opt in q_main.get("options", []):
                label = opt["label"]
                bi = _seg_val(q_main, label, born_in_key)
                bo = _seg_val(q_main, label, born_out_key)
                if q_second:
                    bi2 = _seg_val(q_second, label, born_in_key)
                    bo2 = _seg_val(q_second, label, born_out_key)
                    bi = round(bi + bi2, 1) if bi or bi2 else 0
                    bo = round(bo + bo2, 1) if bo or bo2 else 0
                if bi: left_data[label] = _fmt(bi)
                if bo: right_data[label] = _fmt(bo)

        left_data = dict(sorted(left_data.items(), key=lambda x: x[1], reverse=True))
        right_data = dict(sorted(right_data.items(), key=lambda x: x[1], reverse=True))

        # Find biggest nativity gap
        nat_gaps = []
        for label in set(list(left_data.keys()) + list(right_data.keys())):
            bi = left_data.get(label, 0)
            bo = right_data.get(label, 0)
            if bi and bo:
                nat_gaps.append((label, bo, bi, abs(bo - bi)))
        nat_gaps.sort(key=lambda x: x[3], reverse=True)

        so_what_5 = ""
        if nat_gaps:
            top_gap = nat_gaps[0]
            index_val = round((top_gap[1] / national.get(top_gap[0], 1)) * 100) if national.get(top_gap[0], 0) > 0 else 0
            so_what_5 = (f"{top_gap[0]} at {top_gap[1]}% for foreign-born vs {top_gap[2]}% for Canadian-born "
                        f"(index {index_val}). Canada's growing immigrant population is reshaping the landscape.")

        nat_subtitle = "Combined 1st + 2nd choice by place of birth" if q_second else "Where you were born determines what you choose"
        configs["slide_5"] = {
            "title": "THE NATIVITY DIVIDE",
            "subtitle": nat_subtitle,
            "left_label": "BORN IN CANADA",
            "left_n": _base(q_main, born_in_key) if born_in_key else "",
            "left_data": dict(list(left_data.items())[:6]),
            "right_label": "BORN OUTSIDE CANADA",
            "right_n": _base(q_main, born_out_key) if born_out_key else "",
            "right_data": dict(list(right_data.items())[:6]),
            "so_what": so_what_5
        }
    else:
        # No nativity data — use biggest available split instead
        configs["slide_5"] = _build_biggest_divide(q_main, national, source)

    # ═══════════════════════════════════════════════════════════
    # SLIDE 6: AGE BREAKDOWN
    # ═══════════════════════════════════════════════════════════

    if age_segs and q_main:
        age_data = {}
        age_pairs = [(kw, display) for kw, display, actual in age_segs]
        for opt in q_main.get("options", []):
            label = opt["label"]
            vals_by_age = {}
            for seg_key, display in age_pairs:
                v = _seg_val(q_main, label, seg_key)
                if v: vals_by_age[display] = v
            if vals_by_age:
                age_data[label] = vals_by_age
        age_data = dict(list(age_data.items())[:5])

        configs["slide_6"] = {
            "title": "Younger Canadians Diverge Sharply From Older Generations",
            "subtitle": "% preference by age cohort",
            "data": age_data,
            "colors": ["E8651A", "00A3A1", "0B1D3A"],
            "insight": "The generational divide is stark. What dominates with 55+ may barely register with 18-34.",
            "source": source
        }
    else:
        configs["slide_6"] = {"title": "Demographic Breakdown", "data": {}, "source": source}

    # ═══════════════════════════════════════════════════════════
    # SLIDE 7: GENDER & REGIONAL
    # ═══════════════════════════════════════════════════════════

    gender_rows = []
    if gender_segs and q_main:
        for opt in list(q_main.get("options", []))[:6]:
            label = opt["label"]
            f_val = _seg_val(q_main, label, "female")
            m_val = _seg_val(q_main, label, "male")
            if f_val and m_val:
                gap = round(f_val - m_val, 1)
                gap_str = f"+{gap}F" if gap > 0 else f"+{abs(gap)}M"
                gender_rows.append([label, f"{f_val}%", f"{m_val}%", gap_str])

    region_cards = []
    if region_segs and q_main:
        for kw, display, actual in region_segs[:4]:
            top_label = ""
            top_val = 0
            for opt in q_main.get("options", []):
                v = _seg_val(q_main, opt["label"], kw)
                if v > top_val:
                    top_val = v
                    top_label = opt["label"]
            if top_label:
                region_cards.append({
                    "name": display,
                    "insight": f"#1: {top_label} at {top_val}%"
                })

    configs["slide_7"] = {
        "title": "Gender & Regional Lens",
        "gender_data": gender_rows,
        "regions": region_cards,
        "so_what": "Gender and regional differences demand targeted activation. A national campaign misses the nuance.",
        "source": source
    }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 8: ASPIRATIONAL / DREAM QUESTION
    # ═══════════════════════════════════════════════════════════

    dream_national = _national_sorted(q_dream) if q_dream else national
    q_for_subtitle = q_dream or q_main or {}
    configs["slide_8"] = {
        "title": "The Dream Choice" if q_dream else "Preference Landscape",
        "subtitle": f"{q_for_subtitle.get('text', '')} | National (n={total_n})",
        "data": dream_national,
        "insight": "Aspirational preference reveals emotional investment beyond habitual viewership.",
        "source": source
    }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 9: INCOME ANALYSIS
    # ═══════════════════════════════════════════════════════════

    if income_segs and q_main:
        income_data = {}
        inc_pairs = [(kw, display) for kw, display, actual in income_segs]
        for opt in q_main.get("options", []):
            label = opt["label"]
            vals_by_inc = {}
            for seg_key, display in inc_pairs:
                v = _seg_val(q_main, label, seg_key)
                if v: vals_by_inc[display] = v
            if vals_by_inc:
                income_data[label] = vals_by_inc
        income_data = dict(list(income_data.items())[:5])

        configs["slide_9"] = {
            "title": "Income Shapes Preference — But Not How You'd Expect",
            "subtitle": "% preference by household income",
            "data": income_data,
            "colors": ["D4A843", "E8651A", "00A3A1"],
            "insight": "Income-based targeting has limits. The biggest differences are cultural, not economic.",
            "source": source
        }
    else:
        configs["slide_9"] = {"title": "Additional Segmentation", "data": {}, "source": source}

    # ═══════════════════════════════════════════════════════════
    # SLIDE 10: STRATEGIC IMPLICATIONS
    # ═══════════════════════════════════════════════════════════

    recs = _build_recommendations(national, top_items, nativity_segs, age_segs, gender_rows)
    configs["slide_10"] = {
        "title": "STRATEGIC IMPLICATIONS",
        "recommendations": recs,
        "source": source
    }

    # ═══════════════════════════════════════════════════════════
    # SLIDE 11: CLOSING
    # ═══════════════════════════════════════════════════════════

    configs["slide_11"] = {
        "cta_title": "Let's Unlock the Value of Your Data",
        "cta_subtitle": "For a deeper dive into the insight and what it means for your brand, reach out to our team."
    }

    return configs


# ─── Supporting functions ────────────────────────────────────

def _build_biggest_divide(q_main, national, source):
    """Fallback for slide 5 when nativity data isn't available."""
    # Find the two segments with biggest gap on any option
    if not q_main:
        return {"title": "KEY SEGMENT DIVIDE", "left_data": {}, "right_data": {}, "source": source}

    biggest_gap = 0
    best_split = None

    for opt in q_main.get("options", []):
        vals = opt.get("values", {})
        numeric_vals = {k: float(v) for k, v in vals.items() if isinstance(v, (int, float)) and v > 0}
        if len(numeric_vals) >= 2:
            sorted_segs = sorted(numeric_vals.items(), key=lambda x: x[1], reverse=True)
            high_seg, high_val = sorted_segs[0]
            low_seg, low_val = sorted_segs[-1]
            gap = high_val - low_val
            if gap > biggest_gap:
                biggest_gap = gap
                best_split = (high_seg, low_seg)

    if not best_split:
        return {"title": "KEY SEGMENT DIVIDE", "left_data": {}, "right_data": {}, "source": source}

    left_data = {}
    right_data = {}
    for opt in q_main.get("options", []):
        vals = opt.get("values", {})
        l_val = vals.get(best_split[0], 0)
        r_val = vals.get(best_split[1], 0)
        if l_val: left_data[opt["label"]] = round(float(l_val), 1)
        if r_val: right_data[opt["label"]] = round(float(r_val), 1)

    left_data = dict(sorted(left_data.items(), key=lambda x: x[1], reverse=True))
    right_data = dict(sorted(right_data.items(), key=lambda x: x[1], reverse=True))

    return {
        "title": "THE KEY DIVIDE",
        "subtitle": "The most dramatic segment split in the data",
        "left_label": best_split[0].upper(),
        "left_n": "",
        "left_data": dict(list(left_data.items())[:6]),
        "right_label": best_split[1].upper(),
        "right_n": "",
        "right_data": dict(list(right_data.items())[:6]),
        "so_what": f"A {round(biggest_gap)}pt gap between {best_split[0]} and {best_split[1]} — one-size-fits-all targeting misses this divide."
    }


def _build_recommendations(national, top_items, nativity_segs, age_segs, gender_rows):
    """Auto-generate 4 strategic recommendations based on available data."""
    recs = []

    # Rec 1: Always — the #1 finding
    if top_items:
        recs.append({
            "title": f"Lead with {top_items[0][0]}",
            "body": f"At {top_items[0][1]}% national preference, {top_items[0][0]} represents the broadest reach opportunity. Build primary sponsorship and media strategy around this property."
        })

    # Rec 2: Multicultural play (if nativity data exists)
    if nativity_segs:
        recs.append({
            "title": "Activate the Multicultural Opportunity",
            "body": "The nativity divide reveals two distinct audiences with different passion points. Sponsors targeting Canada's growing immigrant population need a dedicated multicultural strategy."
        })
    elif age_segs:
        recs.append({
            "title": "Bridge the Generation Gap",
            "body": "Young and older Canadians prefer fundamentally different properties. Brands targeting 18-34 need different sponsorship assets than those targeting 55+."
        })

    # Rec 3: Gender-based
    if gender_rows:
        # Find biggest female over-index
        recs.append({
            "title": "Don't Ignore Gender Dynamics",
            "body": "Gender gaps create both risk and opportunity. Properties with strong female over-index are undervalued in sponsorship markets dominated by male-skewing sports."
        })

    # Rec 4: Portfolio approach
    if len(top_items) >= 3:
        spread = round(top_items[0][1] - top_items[2][1])
        if spread <= 10:
            recs.append({
                "title": "Build a Portfolio, Not a Bet",
                "body": f"With only {spread}pts separating the top 3, no single property delivers majority reach. A diversified sponsorship portfolio across 2-3 top properties maximizes coverage."
            })
        else:
            recs.append({
                "title": "Concentrate Where It Counts",
                "body": f"{top_items[0][0]} leads by {spread}pts — a clear market leader. Concentrate primary investment here, with targeted plays on secondary properties for specific segments."
            })

    # Pad to 4 if needed
    while len(recs) < 4:
        recs.append({
            "title": "Commission Deeper Research",
            "body": "This data reveals surface-level preferences. Qualitative research can uncover the WHY behind these patterns — motivation, emotion, and activation triggers."
        })

    return recs[:4]


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate IMI insight deck from survey data")
    parser.add_argument("--data", required=True, help="Path to parsed survey JSON")
    parser.add_argument("--out", default="output.pptx", help="Output .pptx path")
    args = parser.parse_args()

    with open(args.data) as f:
        survey_data = json.load(f)

    configs = survey_to_slides(survey_data)

    # Import renderer and build
    from deck_renderer import build_deck
    build_deck(configs, args.out)
    print(f"Deck saved to {args.out}")
