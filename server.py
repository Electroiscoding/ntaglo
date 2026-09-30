import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bootstrap import load_algo

engine, api = load_algo()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("ranking")

API_KEY = os.getenv("API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

state = {}


@asynccontextmanager
async def lifespan(app):
    service = api.RankingService()
    warmup = api.RankRequest.model_validate({
        "user": {"userId": "warmup"},
        "posts": [{"postId": "w", "text": "warm up the embedding model"}],
    })
    service.rank(warmup)
    state["service"] = service
    log.info("ranking service ready")
    yield


app = FastAPI(
    title="Feed Ranking API",
    version="1.0.0",
    description="Ranks a list of posts for one user using embeddings, engagement, recency, roadmap and taste signals.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def require_key(x_api_key: str = Header(default="")):
    if not API_KEY:
        return
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
def health():
    return {"status": "ok", "ready": "service" in state}


@app.get("/v1/stats", dependencies=[Depends(require_key)])
def stats():
    return state["service"].stats()


@app.post("/v1/rank", response_model=api.RankResponse, dependencies=[Depends(require_key)])
def rank(request: api.RankRequest):
    return state["service"].rank(request)
