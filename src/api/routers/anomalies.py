"""
Anomaly Detection endpoints for Klaus IMI.
Surfaces statistical outliers across all IMI CSV datasets.
"""

import logging

from fastapi import APIRouter

from config import IMI_DATA_DIR
from imi_anomaly import detect_anomalies, detect_anomalies_for_dataset, anomaly_summary

logger = logging.getLogger("imi_anomalies")

router = APIRouter(prefix="/klaus/imi", tags=["imi-anomalies"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/anomalies/summary")
def get_anomaly_summary():
    """Count by severity, top anomalous datasets."""
    return anomaly_summary(IMI_DATA_DIR)


@router.get("/anomalies/{dataset}")
def get_dataset_anomalies(dataset: str):
    """Anomalies for a specific dataset."""
    results = detect_anomalies_for_dataset(dataset, IMI_DATA_DIR)
    return {"dataset": dataset, "count": len(results), "anomalies": results}


@router.get("/anomalies")
def get_all_anomalies():
    """All current anomalies across all datasets."""
    results = detect_anomalies(IMI_DATA_DIR)
    return {"count": len(results), "anomalies": results}
