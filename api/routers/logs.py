from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from src.logger import logger

router = APIRouter()

@router.get("/")
def get_logs(limit: int = 100, level: str = "ALL"):
    df = logger.get_logs(limit=limit)
    if df.empty:
        return []

    if level != "ALL":
        df = df[df['level'] == level]

    df = df.sort_values('timestamp', ascending=False)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df.to_dict(orient="records")

@router.delete("/")
def clear_logs():
    logger.clear()
    logger.log("API", "INFO", "Logs cleared via API")
    return {"status": "success"}
