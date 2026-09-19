from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import data, market_radar, diagnostics, predictive, multivariate, pair_radar, logs

app = FastAPI(
    title="Crypto Filter API",
    description="Backend API for quantitative symbol diagnostics and predictive analytics",
    version="1.0.0"
)

# Setup CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default ports
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
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
