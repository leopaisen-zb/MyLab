# main.py
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
import uvicorn

app = FastAPI(
    title="MatGen-Eq API",
    description="四层微服务架构的材料发现系统 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "MatGen-Eq",
        "version": "1.0.0",
        "architecture": "四层微服务架构",
        "layers": ["交互层 (API)", "任务编排层 (Orchestrator)", "算法适配层 (Adapters)", "数据持久化层 (Persistence)"]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
