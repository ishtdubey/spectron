"""
Streaming HTTP wrapper around GroqDeepResearchPipeline (research_agent.py).

Run:
    cd backend
    pip install fastapi uvicorn pydantic openai python-dotenv tavily-python
    uvicorn server:app --reload --port 8000

Set MOCK=1 to run without API keys.
"""
import asyncio
import json
import os
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

MOCK = os.getenv("MOCK") == "1"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

app = FastAPI(title="Research UI backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def ev(**kw) -> str:
    return json.dumps(kw) + "\n"


async def run_mock(query: str) -> AsyncIterator[str]:
    complex_q = len(query.split()) > 8
    yield ev(type="stage", stage="classify")
    await asyncio.sleep(0.4)
    yield ev(type="classified", query_type="COMPLEX" if complex_q else "SIMPLE",
             reasoning="Mock: long open-ended query." if complex_q else "Mock: short fact lookup.")
    perspectives = []
    if complex_q:
        yield ev(type="stage", stage="plan")
        await asyncio.sleep(0.4)
        perspectives = ["Operational", "Cost", "Risk"]
        yield ev(type="perspectives", perspectives=perspectives)
    yield ev(type="stage", stage="collect")
    await asyncio.sleep(0.4)
    evidence = [
        {
            "id": f"doc_{i}",
            "title": f"Sample Source {i}",
            "source": f"Research Repository {i}",
            "url": f"https://example.com/source-{i}",
            "perspective": "General",
            "content": f"Sample evidence {i} about '{query}'.",
            "relevance_score": 0.95 - (i * 0.05)
        }
        for i in range(1, 5)
    ]
    yield ev(type="stage", stage="critique")
    await asyncio.sleep(0.4)
    yield ev(type="stage", stage="synthesize")
    await asyncio.sleep(0.4)
    yield ev(type="stage", stage="verify")
    await asyncio.sleep(0.4)
    yield ev(type="stage", stage="epistemic")
    await asyncio.sleep(0.4)
    
    valid_ids = [e["id"] for e in evidence[:3]]
    yield ev(
        type="result",
        query_type="COMPLEX" if complex_q else "SIMPLE",
        perspectives=perspectives,
        evidence=evidence,
        validated_evidence_ids=valid_ids,
        conflicts=["Mock: sources disagree on migration cost."] if complex_q else [],
        weak_evidence=["Mock: one source is thin."] if complex_q else [],
        answer=(f"This is a mock answer to: {query}\n\n"
                f"First point is supported [{valid_ids[0]}].\n\nSecond point [{valid_ids[1]}]."),
        limitations=["Mock: evidence is placeholder data.", "Mock: one claim is disputed."],
        claim_verification={
            "total_citations_found": 2,
            "valid_citations": valid_ids[:2],
            "invalid_citations": [],
            "verification_passed": True,
            "verification_notes": "All 2 citations verified against grounded evidence."
        },
        epistemic={
            "known_facts": ["Mock fact A", "Mock fact B"],
            "uncertainties": ["Mock uncertainty"],
            "disputed_claims": ["Mock disputed claim"],
            "unresolved_questions": ["Mock open question?"],
        },
    )


async def run_real(query: str) -> AsyncIterator[str]:
    from research_agent import GroqDeepResearchPipeline

    p = GroqDeepResearchPipeline(model_name=MODEL)

    # Stage 1: Classify
    yield ev(type="stage", stage="classify")
    c = await p.classify_query(query)
    yield ev(type="classified", query_type=c.query_type, reasoning=c.reasoning)

    raw_evidence = []
    perspective_names = []
    if c.query_type == "SIMPLE":
        yield ev(type="stage", stage="collect")
        raw_evidence = await p.collect_evidence(
            query=query, 
            perspective_name="General", 
            search_queries=[query]
        )
    else:
        # Stage 2: Perspective Planning
        yield ev(type="stage", stage="plan")
        plan = await p.plan_perspectives(query)
        perspective_names = [x.name for x in plan.perspectives]
        yield ev(type="perspectives", perspectives=perspective_names)
        
        # Stage 3: Evidence Collection
        yield ev(type="stage", stage="collect")
        results = await asyncio.gather(*[
            p.collect_evidence(
                query=query,
                perspective_name=pers.name,
                search_queries=pers.search_queries
            )
            for pers in plan.perspectives
        ])
        for r in results:
            raw_evidence.extend(r)

    # FIX 1: Clean sequential ID assignment (doc_1, doc_2...)
    evidence = p.assign_sequential_ids(raw_evidence)

    # Stage 4: Evidence Critique
    yield ev(type="stage", stage="critique")
    crit = await p.critique_evidence(query, evidence)
    valid = [e for e in evidence if e.id in crit.validated_evidence_ids]

    # Stage 5: Synthesis
    yield ev(type="stage", stage="synthesize")
    synthesis = await p.synthesize(query, valid, crit)

    # Stage 6: Claim Verification (FIX 3)
    yield ev(type="stage", stage="verify")
    verification = await p.verify_claims(synthesis, valid)

    # Stage 7: Epistemic State Memory
    yield ev(type="stage", stage="epistemic")
    state = await p.update_epistemic_memory(query, crit, synthesis)

    # Separate limitations field
    limitations = state.uncertainties + state.disputed_claims

    yield ev(
        type="result",
        query_type=c.query_type,
        perspectives=perspective_names,
        evidence=[e.model_dump() for e in evidence],
        validated_evidence_ids=crit.validated_evidence_ids,
        conflicts=crit.conflicts_detected,
        weak_evidence=crit.weak_evidence_flagged,
        answer=synthesis,
        limitations=limitations,
        claim_verification=verification.model_dump(),
        epistemic=state.model_dump(),
    )


@app.get("/api/health")
def health():
    return {"ok": True, "mock": MOCK, "model": MODEL}


@app.post("/api/query")
async def query(body: QueryIn):
    async def gen():
        try:
            if not MOCK:
                if not os.getenv("GROQ_API_KEY"):
                    yield ev(type="error", message="GROQ_API_KEY is not set. Add it to .env or run with MOCK=1.")
                    return
                if not os.getenv("TAVILY_API_KEY"):
                    yield ev(type="error", message="TAVILY_API_KEY is not set. Add it to .env or run with MOCK=1.")
                    return
            runner = run_mock if MOCK else run_real
            async for chunk in runner(body.query.strip()):
                yield chunk
        except Exception as e:
            yield ev(type="error", message=f"{type(e).__name__}: {e}")

    return StreamingResponse(gen(), media_type="application/x-ndjson")