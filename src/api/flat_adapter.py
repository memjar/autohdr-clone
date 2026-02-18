"""
Klaus x IMI — Flat-Format Survey Adapter
=========================================
Converts flat/tabular xlsx data (brand trackers, sponsorship scores,
say-do gap tables, etc.) into the structured survey format expected
by survey_bridge.survey_to_slides().

Supports auto-detection of these IMI data types:
  - Brand Health Tracker (columns: Brand, Awareness, NPS, etc.)
  - Sponsorship Property Scores (columns: Property, Awareness, Engagement, etc.)
  - Say-Do Gap (columns: Category, Segment, Stated_Purchase_Intent, etc.)
  - Purchase Drivers (columns: Generation, Purchase_Driver, Rank, etc.)
  - Competitive Benchmark (columns: Brand, Awareness, Consideration, etc.)
"""

import pandas as pd
import io
from typing import Optional


def flat_to_structured(file_bytes: bytes, filename: str) -> Optional[dict]:
    """
    Attempt to convert a flat xlsx/csv into structured survey format.
    Returns None if the format isn't recognized.
    """
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception:
        return None

    cols = [c.lower().replace(" ", "_") for c in df.columns]
    col_map = {c: orig for c, orig in zip(cols, df.columns)}

    # Detect format and dispatch
    if "brand" in cols and "nps" in cols:
        return _parse_brand_health(df, filename)
    if "property" in cols and "awareness" in cols:
        return _parse_sponsorship(df, filename)
    if "say_do_gap" in cols or "stated_purchase_intent" in cols:
        return _parse_say_do(df, filename)
    if "purchase_driver" in cols or "importance_score" in cols:
        return _parse_purchase_drivers(df, filename)
    if "brand" in cols and "consideration" in cols and "trial" in cols:
        return _parse_competitive(df, filename)

    return None


def _clean_name(filename: str) -> str:
    return filename.replace(".xlsx", "").replace(".csv", "").replace("_", " ").replace("-", " ").strip().title()


def _pct(v):
    """Convert 0.xx to xx%, leave integers alone."""
    if isinstance(v, float) and 0 < v < 1:
        return round(v * 100, 1)
    return round(float(v), 1) if isinstance(v, (int, float)) else 0


# ─── Brand Health Tracker ──────────────────────────────────

def _parse_brand_health(df: pd.DataFrame, filename: str) -> dict:
    """Brand × metric grid → structured questions per metric."""
    name = _clean_name(filename)
    brands = df.iloc[:, 0].tolist()
    total_n = int(df["Sample Size"].max()) if "Sample Size" in df.columns else 0

    metrics = [
        ("Unaided Awareness", "unaided_awareness"),
        ("Aided Awareness", "aided_awareness"),
        ("Consideration", "consideration"),
        ("Purchase Intent", "purchase_intent"),
        ("NPS", "nps"),
        ("Trust Score", "trust_score"),
        ("Quality Perception", "quality_perception"),
        ("Value Perception", "value_perception"),
    ]

    questions = []
    for display_name, _ in metrics:
        if display_name not in df.columns:
            continue
        options = []
        for _, row in df.iterrows():
            brand = str(row.iloc[0])
            val = _pct(row[display_name])
            options.append({
                "label": brand,
                "values": {"Total": val}
            })
        # Sort by value descending
        options.sort(key=lambda o: o["values"].get("Total", 0), reverse=True)
        questions.append({
            "id": display_name.replace(" ", "_"),
            "text": f"{display_name} by Brand",
            "options": options,
            "base_sizes": {"Total": total_n}
        })

    return {
        "survey_name": name,
        "total_n": total_n,
        "segments": ["Total"],
        "questions": questions,
        "data_type": "brand_health"
    }


# ─── Sponsorship Property Scores ───────────────────────────

def _parse_sponsorship(df: pd.DataFrame, filename: str) -> dict:
    name = _clean_name(filename)
    total_n = int(df["Sample_Size"].max()) if "Sample_Size" in df.columns else 0

    # Build generation segments from pct columns
    gen_cols = [c for c in df.columns if c.startswith("Pct_Gen") or c.startswith("Pct_Millennial") or c.startswith("Pct_Boomer")]

    # Main question: Opportunity Score ranking
    opp_options = []
    for _, row in df.iterrows():
        prop = str(row["Property"])
        vals = {"Total": float(row.get("Opportunity_Score", 0))}
        # Add generation breakdown
        for gc in gen_cols:
            gen_label = gc.replace("Pct_", "").replace("_", " ")
            vals[gen_label] = _pct(row.get(gc, 0))
        opp_options.append({"label": prop, "values": vals})
    opp_options.sort(key=lambda o: o["values"].get("Total", 0), reverse=True)

    # Secondary metrics as additional questions
    metric_qs = []
    for metric in ["Awareness", "Favorability", "Engagement", "Emotional_Connection", "Purchase_Impact"]:
        if metric not in df.columns:
            continue
        opts = []
        for _, row in df.iterrows():
            vals = {"Total": _pct(row[metric])}
            if "Pct_Female" in df.columns:
                vals["Female"] = _pct(row["Pct_Female"])
                vals["Male"] = round(100 - _pct(row["Pct_Female"]), 1)
            opts.append({"label": str(row["Property"]), "values": vals})
        opts.sort(key=lambda o: o["values"].get("Total", 0), reverse=True)
        metric_qs.append({
            "id": metric,
            "text": f"{metric.replace('_', ' ')} by Property",
            "options": opts,
            "base_sizes": {"Total": total_n}
        })

    questions = [{
        "id": "Opportunity_Score",
        "text": "Sponsorship Opportunity Score by Property",
        "options": opp_options,
        "base_sizes": {"Total": total_n}
    }] + metric_qs

    segments = ["Total"] + [gc.replace("Pct_", "").replace("_", " ") for gc in gen_cols]
    if "Pct_Female" in df.columns:
        segments += ["Female", "Male"]

    return {
        "survey_name": name,
        "total_n": total_n,
        "segments": segments,
        "questions": questions,
        "data_type": "sponsorship"
    }


# ─── Say-Do Gap ────────────────────────────────────────────

def _parse_say_do(df: pd.DataFrame, filename: str) -> dict:
    name = _clean_name(filename)

    # Get unique categories and segments
    categories = df["Category"].unique().tolist() if "Category" in df.columns else []
    segments = df["Segment"].unique().tolist() if "Segment" in df.columns else ["Total"]
    total_n = int(df["Sample_Size_Stated"].max()) if "Sample_Size_Stated" in df.columns else 0

    # Q1: Stated Purchase Intent by category, segmented
    q_stated = []
    for cat in categories:
        cat_rows = df[df["Category"] == cat]
        vals = {}
        for _, row in cat_rows.iterrows():
            seg = str(row.get("Segment", "Total"))
            vals[seg] = _pct(row.get("Stated_Purchase_Intent", 0))
        q_stated.append({"label": cat, "values": vals})
    q_stated.sort(key=lambda o: max(o["values"].values()) if o["values"] else 0, reverse=True)

    # Q2: Actual Purchase
    q_actual = []
    for cat in categories:
        cat_rows = df[df["Category"] == cat]
        vals = {}
        for _, row in cat_rows.iterrows():
            seg = str(row.get("Segment", "Total"))
            vals[seg] = _pct(row.get("Actual_Purchase_3mo", 0))
        q_actual.append({"label": cat, "values": vals})
    q_actual.sort(key=lambda o: max(o["values"].values()) if o["values"] else 0, reverse=True)

    # Q3: Say-Do Gap
    q_gap = []
    for cat in categories:
        cat_rows = df[df["Category"] == cat]
        vals = {}
        for _, row in cat_rows.iterrows():
            seg = str(row.get("Segment", "Total"))
            vals[seg] = _pct(row.get("Say_Do_Gap", 0))
        q_gap.append({"label": cat, "values": vals})
    q_gap.sort(key=lambda o: max(o["values"].values()) if o["values"] else 0, reverse=True)

    base_sizes = {seg: total_n for seg in segments}

    return {
        "survey_name": name,
        "total_n": total_n,
        "segments": segments,
        "questions": [
            {"id": "Stated_Intent", "text": "Stated Purchase Intent by Category", "options": q_stated, "base_sizes": base_sizes},
            {"id": "Actual_Purchase", "text": "Actual Purchase (3mo) by Category", "options": q_actual, "base_sizes": base_sizes},
            {"id": "Say_Do_Gap", "text": "Say-Do Gap™ by Category — Intent vs Reality", "options": q_gap, "base_sizes": base_sizes},
        ],
        "data_type": "say_do_gap"
    }


# ─── Purchase Drivers ──────────────────────────────────────

def _parse_purchase_drivers(df: pd.DataFrame, filename: str) -> dict:
    name = _clean_name(filename)
    generations = df["Generation"].unique().tolist() if "Generation" in df.columns else ["Total"]
    total_n = int(df["Sample_Size"].max()) if "Sample_Size" in df.columns else 0
    drivers = df["Purchase_Driver"].unique().tolist() if "Purchase_Driver" in df.columns else []

    # Q1: Importance Score by driver, segmented by generation
    q_importance = []
    for driver in drivers:
        drv_rows = df[df["Purchase_Driver"] == driver]
        vals = {}
        for _, row in drv_rows.iterrows():
            gen = str(row.get("Generation", "Total"))
            vals[gen] = round(float(row.get("Importance_Score", 0)), 1)
        q_importance.append({"label": driver, "values": vals})
    q_importance.sort(key=lambda o: max(o["values"].values()) if o["values"] else 0, reverse=True)

    # Q2: % Selected Top 3
    q_top3 = []
    for driver in drivers:
        drv_rows = df[df["Purchase_Driver"] == driver]
        vals = {}
        for _, row in drv_rows.iterrows():
            gen = str(row.get("Generation", "Total"))
            vals[gen] = _pct(row.get("Pct_Selected_Top3", 0))
        q_top3.append({"label": driver, "values": vals})
    q_top3.sort(key=lambda o: max(o["values"].values()) if o["values"] else 0, reverse=True)

    base_sizes = {gen: total_n for gen in generations}

    return {
        "survey_name": name,
        "total_n": total_n,
        "segments": generations,
        "questions": [
            {"id": "Importance", "text": "Purchase Driver Importance Score by Generation", "options": q_importance, "base_sizes": base_sizes},
            {"id": "Top3_Selected", "text": "% Selected in Top 3 Purchase Drivers", "options": q_top3, "base_sizes": base_sizes},
        ],
        "data_type": "purchase_drivers"
    }


# ─── Competitive Benchmark ─────────────────────────────────

def _parse_competitive(df: pd.DataFrame, filename: str) -> dict:
    name = _clean_name(filename)
    brands = df.iloc[:, 0].tolist()

    metrics = [c for c in df.columns if c != df.columns[0]]
    questions = []
    for metric in metrics:
        opts = []
        for _, row in df.iterrows():
            brand = str(row.iloc[0])
            val = _pct(row[metric]) if metric != "NPS" else round(float(row[metric]), 1)
            opts.append({"label": brand, "values": {"Total": val}})
        opts.sort(key=lambda o: o["values"].get("Total", 0), reverse=True)
        questions.append({
            "id": metric,
            "text": f"{metric.replace('_', ' ')} by Brand",
            "options": opts,
            "base_sizes": {"Total": len(brands)}
        })

    return {
        "survey_name": name,
        "total_n": len(brands),
        "segments": ["Total"],
        "questions": questions,
        "data_type": "competitive_benchmark"
    }
