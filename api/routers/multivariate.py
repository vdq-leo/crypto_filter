from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.data import DataManager
from ml_engine.analysis.multivariate import MatrixEngine, DecompositionEngine
from src.config import MANDATORY_CRYPTO
from src.shared_state import get_manager, get_engine

router = APIRouter()
manager = get_manager()

class MatrixRequest(BaseModel):
    symbols: List[str]
    interval: str = "1h"
    structure: str = "Correlation" # Correlation, Covariance, Arbitrage
    data_source: str = "return" # price, return
    data_structure: str = "raw" # raw, ranking, sign
    method: Optional[str] = "pearson" # pearson, spearman, kendall; or arb_method for Arbitrage
    window: int = 300
    mean_reversion: bool = True

class DecompRequest(BaseModel):
    symbols: List[str]
    interval: str = "1h"
    method: str = "eigen" # eigen, rmt, cluster, mst
    data_source: str = "return"
    data_structure: str = "raw"
    window: int = 100
    n_components: int = 5
    linkage_method: str = "complete"
    matrix_structure: str = "Correlation"
    matrix_method: Optional[str] = "pearson"
    mean_reversion: bool = True

def _load_data_concurrently(symbols, interval, load_price=False):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    data_map = {}
    
    def fetch_sym(sym):
        try:
            df = manager.load_data(sym, interval, auto_sync=False)
            if df is not None and not df.empty:
                close = df.set_index("open_time")["close"]
                if load_price:
                    return sym, close
                ret_series = np.log(close / close.shift(1))
                return sym, ret_series.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            pass
        return sym, None

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {executor.submit(fetch_sym, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            sym, result = future.result()
            if result is not None:
                data_map[sym] = result
    return data_map

@router.post("/matrix")
def generate_matrix(req: MatrixRequest):
    try:
        load_price = (req.structure == "Arbitrage" or req.data_source == "price" or req.structure == "Correlation" or req.structure == "Covariance" or req.structure == "Partial Correlation")
        data_map = _load_data_concurrently(req.symbols, req.interval, load_price=True) # Always load price for MatrixEngine to handle
        
        if not data_map:
            raise HTTPException(status_code=400, detail="Failed to load data")

        if req.structure == "Correlation":
            raw_matrix = MatrixEngine.calculate_matrix(
                data_map, method=req.method, window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )
        elif req.structure == "Covariance":
            raw_matrix = MatrixEngine.calculate_covariance(
                data_map, window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )
        elif req.structure == "Partial Correlation":
            raw_matrix = MatrixEngine.calculate_partial_correlation(
                data_map, window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )
        elif req.structure == "Arbitrage":
            price_df = pd.DataFrame(data_map)
            method = req.method
            if method == "cointegration":
                raw_matrix = MatrixEngine.calculate_coint_matrix(price_df, req.window)
            elif method == "zscore":
                raw_matrix = MatrixEngine.calculate_zscore_matrix(price_df, req.window)
            elif method == "halflife":
                raw_matrix = MatrixEngine.calculate_halflife_matrix(price_df, req.window)
            elif method == "vol_ratio":
                raw_matrix = MatrixEngine.calculate_vol_ratio_matrix(price_df, req.window)
            elif method == "arbitrage_score":
                raw_matrix = MatrixEngine.calculate_arbitrage_score_matrix(price_df, req.window, req.mean_reversion)
            else:
                raise HTTPException(status_code=400, detail="Invalid arbitrage method")
        else:
            raise HTTPException(status_code=400, detail="Invalid structure")

        filtered_matrix, _ = MatrixEngine.filter_blanks(raw_matrix)
        
        # Format for JSON
        filtered_matrix = filtered_matrix.dropna(axis=1, how="all").dropna(axis=0, how="all")
        filtered_matrix = filtered_matrix.replace([np.inf, -np.inf], None).where(pd.notnull(filtered_matrix), None)
        
        return {
            "columns": list(filtered_matrix.columns),
            "index": list(filtered_matrix.index),
            "data": filtered_matrix.values.tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decomposition")
def run_decomposition(req: DecompRequest):
    try:
        load_price = (req.matrix_structure == "Arbitrage" or req.data_source == "price" or req.matrix_structure == "Correlation" or req.matrix_structure == "Covariance" or req.matrix_structure == "Partial Correlation")
        data_map = _load_data_concurrently(req.symbols, req.interval, load_price=True)
        
        if not data_map:
            raise HTTPException(status_code=400, detail="Failed to load data")

        if req.matrix_structure == "Correlation":
            raw_matrix = MatrixEngine.calculate_matrix(
                data_map, method=req.matrix_method, window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )
        elif req.matrix_structure == "Covariance":
            raw_matrix = MatrixEngine.calculate_covariance(
                data_map, window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )
        elif req.matrix_structure == "Partial Correlation":
            raw_matrix = MatrixEngine.calculate_partial_correlation(
                data_map, window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )
        elif req.matrix_structure == "Arbitrage":
            price_df = pd.DataFrame(data_map)
            method = req.matrix_method
            if method == "cointegration":
                raw_matrix = MatrixEngine.calculate_coint_matrix(price_df, req.window)
            elif method == "zscore":
                raw_matrix = MatrixEngine.calculate_zscore_matrix(price_df, req.window)
            elif method == "halflife":
                raw_matrix = MatrixEngine.calculate_halflife_matrix(price_df, req.window)
            elif method == "vol_ratio":
                raw_matrix = MatrixEngine.calculate_vol_ratio_matrix(price_df, req.window)
            elif method == "arbitrage_score":
                raw_matrix = MatrixEngine.calculate_arbitrage_score_matrix(price_df, req.window, req.mean_reversion)
            else:
                raise HTTPException(status_code=400, detail="Invalid arbitrage method")
        else:
            raw_matrix = MatrixEngine.calculate_matrix(
                data_map, method="pearson", window_size=req.window,
                data_source=req.data_source, data_structure=req.data_structure
            )

        corr_clean, _ = MatrixEngine.filter_blanks(raw_matrix)
        
        aligned_price_map = {s: data_map[s] for s in corr_clean.columns if s in data_map}
        wide_df_raw = pd.DataFrame(aligned_price_map).iloc[-req.window:].dropna()
        wide_df = MatrixEngine._prepare_data(wide_df_raw, data_source=req.data_source, data_structure=req.data_structure)
        
        if wide_df.empty:
            raise HTTPException(status_code=400, detail="Not enough aligned data")
            
        T = len(wide_df)
        N = len(wide_df.columns)
        
        if req.method == "eigen":
            res = DecompositionEngine.eigen_decomposition(corr_clean, n_components=req.n_components)
            return {"method": "eigen", "components": res['components'].to_dict(), "variance": res['variance_explained'].to_dict()}
        elif req.method == "rmt":
            res = DecompositionEngine.rmt_filter(corr_clean, T, N)
            filtered = res['filtered_matrix'].replace([np.inf, -np.inf], None).where(pd.notnull(res['filtered_matrix']), None)
            return {"method": "rmt", "filtered_matrix": filtered.to_dict()}
        elif req.method == "cluster":
            dist = DecompositionEngine.distance_matrix(corr_clean.fillna(0))
            res = DecompositionEngine.hierarchical_cluster(dist, method=req.linkage_method)
            return {"method": "cluster", "linkage": res['linkage_matrix'].tolist(), "labels": list(dist.index)}
        elif req.method == "mst":
            res = DecompositionEngine.mst_network(corr_clean)
            return {"method": "mst", "edges": res['edges'].to_dict(orient="records")}
        else:
            raise HTTPException(status_code=400, detail="Invalid method")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
