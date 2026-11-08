import asyncio
import json
import os
import re
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
            "id": f"doc_{p.lower()}_{i}",
            "title": f"Sample Source {i} for {p}",
            "source": f"{p} Research Repository",
            "url": f"https://example.com/{p.lower()}-{i}",
            "perspective": p,
            "content": f"Sample evidence {i} about '{query}' from the {p} view.",
            "relevance_score": 0.95 - (i * 0.05)
        }
        for p in (perspectives or ["General"]) for i in (1, 2)
    ]
    yield ev(type="stage", stage="critique")
    await asyncio.sleep(0.4)
    yield ev(type="stage", stage="synthesize")
    await asyncio.sleep(0.4)
    yield ev(type="stage", stage="epistemic")
    await asyncio.sleep(0.4)
    ids = [e["id"] for e in evidence]
    yield ev(
        type="result",
        query_type="COMPLEX" if complex_q else "SIMPLE",
        perspectives=perspectives,
        evidence=evidence,
        validated_evidence_ids=ids[:-1],
        conflicts=["Mock: sources disagree on migration cost."] if complex_q else [],
        weak_evidence=["Mock: one source is thin."] if complex_q else [],
        answer=(f"This is a mock answer to: {query}\n\n"
                f"First point is supported [{ids[0]}].\n\nSecond point [{ids[1]}]."),
        limitations=["Mock: evidence is placeholder data.", "Mock: one claim is disputed."],
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

    yield ev(type="stage", stage="classify")
    c = await p.classify_query(query)
    yield ev(type="classified", query_type=c.query_type, reasoning=c.reasoning)

    evidence = []
    perspective_names = []
    if c.query_type == "SIMPLE":
        yield ev(type="stage", stage="collect")
        evidence = await p.collect_evidence(
            query=query, 
            perspective_name="General", 
            search_queries=[query]
        )
    else:
        yield ev(type="stage", stage="plan")
        plan = await p.plan_perspectives(query)
        perspective_names = [x.name for x in plan.perspectives]
        yield ev(type="perspectives", perspectives=perspective_names)
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
            evidence.extend(r)

    yield ev(type="stage", stage="critique")
    crit = await p.critique_evidence(query, evidence)
    valid = [e for e in evidence if e.id in crit.validated_evidence_ids]

    yield ev(type="stage", stage="synthesize")
    synthesis = await p.synthesize(query, valid, crit)

    yield ev(type="stage", stage="epistemic")
    state = await p.update_epistemic_memory(query, crit, synthesis)

    yield ev(
        type="result",
        query_type=c.query_type,
        perspectives=perspective_names,
        evidence=[e.model_dump() for e in evidence],
        validated_evidence_ids=crit.validated_evidence_ids,
        conflicts=crit.conflicts_detected,
        weak_evidence=crit.weak_evidence_flagged,
        answer=synthesis,
        limitations=state.uncertainties + state.disputed_claims,
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