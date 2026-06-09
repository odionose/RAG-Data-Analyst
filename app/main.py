from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from agent.graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Data Analyst API",
    description="LangGraph ReAct Agent with Gemini, Qdrant vector search, and SQLite",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    cost_summary: dict
    latency_seconds: float

@app.get("/")
def root():
    return {
        "app": "RAG Data Analyst",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"[REQUEST] query='{request.query[:80]}'")
    start = time.time()

    try:
        result = run_agent(request.query)
    except Exception as e:
        logger.error(f"[ERROR] Agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = round(time.time() - start, 3)

    # FinOps log — visible in Cloud Run log viewer
    logger.info(
        f"[FINOPS] tokens_in={result['cost']['total_input_tokens']} "
        f"tokens_out={result['cost']['total_output_tokens']} "
        f"cost_usd={result['cost']['total_cost_usd']:.6f} "
        f"latency_s={latency} "
        f"tier={result['cost']['tier']}"
    )

    return QueryResponse(
        answer=result["answer"],
        cost_summary=result["cost"],
        latency_seconds=latency,
    )