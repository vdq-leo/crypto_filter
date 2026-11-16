import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import data, market_radar, diagnostics, predictive, multivariate, pair_radar, logs, portfolio
from src.logger import logger
from src.shared_state import get_manager
from src.config import MANDATORY_CRYPTO

SYNC_INTERVALS = ["15m", "1h", "4h", "1d"]

def _sync_mandatory_batch():
    try:
        manager = get_manager()
        symbols = list(MANDATORY_CRYPTO)
        try:
            top_syms = manager.get_universe(top_n=50)
            if top_syms:
                for s in top_syms:
                    if s not in symbols:
                        symbols.append(s)
        except Exception:
            pass

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for sym in symbols[:40]:
                for interval in SYNC_INTERVALS:
                    futures.append(pool.submit(manager.load_data, sym, interval, auto_sync=True))
            for f in futures:
                try:
                    f.result()
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Error in _sync_mandatory_batch: {e}")

async def continuous_data_sync_daemon():
    logger.info("Continuous market data ingestion daemon started.")
    await asyncio.sleep(5)
    while True:
        try:
            await asyncio.to_thread(_sync_mandatory_batch)
            logger.info("Continuous data ingestion: Synced mandatory symbols cache.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in continuous data ingestion daemon: {e}")
        await asyncio.sleep(120)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_task = asyncio.create_task(continuous_data_sync_daemon())
    yield
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Crypto Filter API",
    description="Backend API for quantitative symbol diagnostics and predictive analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Setup CORS for the Next.js frontend and other consumers
cors_env = os.getenv("CORS_ORIGINS", "*")
if cors_env == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_env.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def read_root():
    return {"status": "success", "message": "Welcome to Crypto Filter API"}

# Include routers here as we implement them
app.include_router(data.router, prefix="/api/data", tags=["Data Processing"])
app.include_router(market_radar.router, prefix="/api/market-radar", tags=["Market Radar"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["Diagnostics"])
app.include_router(predictive.router, prefix="/api/predictive", tags=["Predictive Analytics"])
app.include_router(multivariate.router, prefix="/api/multivariate", tags=["Multivariate"])
app.include_router(pair_radar.router, prefix="/api/pair-radar", tags=["Pair Radar"])
app.include_router(pair_radar.router, prefix="/api/pair_radar", tags=["Pair Radar"])
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
