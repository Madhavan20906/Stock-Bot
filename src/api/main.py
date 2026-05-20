"""
FastAPI REST API for StockBot inference endpoints.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("StockBot API starting up...")
    yield
    logger.info("StockBot API shutting down.")


app = FastAPI(
    title="StockBot API",
    description="AI Stock Market Prediction & Trading Signal API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


def start():
    import os
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )


if __name__ == "__main__":
    start()
