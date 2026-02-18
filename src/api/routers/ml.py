"""
ML Router — Predictive analytics endpoints for Klaus IMI.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import imi_ml

logger = logging.getLogger("imi-ml")

router = APIRouter(prefix="/klaus/imi", tags=["imi-ml"])


# ─── Request Models ───────────────────────────────────────


class ClusterRequest(BaseModel):
    dataset: str
    columns: list[str]
    n_clusters: int = Field(default=3, ge=2, le=20)


class PredictRequest(BaseModel):
    dataset: str
    target_column: str
    feature_columns: list[str]


class SegmentRequest(BaseModel):
    dataset: str
    columns: Optional[list[str]] = None


class TrendRequest(BaseModel):
    dataset: str
    column: str
    periods: int = Field(default=4, ge=1, le=52)


class ScenarioRequest(BaseModel):
    dataset: str
    column: str
    change_pct: float


# ─── Endpoints ────────────────────────────────────────────


@router.get("/ml/datasets")
async def get_datasets():
    """List available CSV datasets with their numeric columns."""
    return {"datasets": imi_ml.list_datasets()}


@router.post("/ml/cluster")
async def cluster(req: ClusterRequest):
    """KMeans clustering on specified columns."""
    try:
        result = imi_ml.cluster_data(req.dataset, req.columns, req.n_clusters)
        return result
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Cluster error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/predict")
async def predict(req: PredictRequest):
    """Prediction via Linear Regression or RandomForest."""
    try:
        result = imi_ml.predict(req.dataset, req.target_column, req.feature_columns)
        return result
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Predict error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/segment")
async def segment(req: SegmentRequest):
    """Auto-segmentation with optimal k via silhouette score."""
    try:
        result = imi_ml.segment(req.dataset, req.columns)
        return result
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Segment error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/trend")
async def trend_forecast(req: TrendRequest):
    """Linear trend extrapolation forecast."""
    try:
        result = imi_ml.trend_forecast(req.dataset, req.column, req.periods)
        return result
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Trend forecast error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/scenario")
async def scenario_analysis(req: ScenarioRequest):
    """What-if scenario: apply % change and show correlated impacts."""
    try:
        result = imi_ml.scenario_analysis(req.dataset, req.column, req.change_pct)
        return result
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Scenario analysis error")
        raise HTTPException(status_code=500, detail=str(e))
