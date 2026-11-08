import asyncio
import os
import json
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ==========================================
# 1. Pydantic Schemas for Pipeline Nodes
# ==========================================

class QueryClassification(BaseModel):
    query_type: Literal["SIMPLE", "COMPLEX"]
    reasoning: str

class Perspective(BaseModel):
    name: str
    description: str
    search_queries: List[str]

class PerspectivePlan(BaseModel):
    perspectives: List[Perspective]

class EvidenceItem(BaseModel):
    id: str
    source: str
    perspective: Optional[str] = None
    content: str

class EvidenceCritique(BaseModel):
    grounding_assessment: str
    conflicts_detected: List[str] = Field(default_factory=list)
    weak_evidence_flagged: List[str] = Field(default_factory=list)
    validated_evidence_ids: List[str]

class EpistemicState(BaseModel):
    known_facts: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    disputed_claims: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)

class FinalResearchOutput(BaseModel):
    synthesized_response: str
    limitations: List[str]
    epistemic_summary: EpistemicState


# ==========================================
# 2. Groq Research Pipeline Engine
# ==========================================

class GroqDeepResearchPipeline:
    def __init__(
        self, 
        groq_api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b"
    ):
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable or .env file is required.")

        # Configure AsyncOpenAI client targeting Groq's endpoint
        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.model = model_name

    async def _parse_json_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
        """Helper to enforce structured JSON output matching Pydantic schemas."""
        json_schema_str = json.dumps(schema.model_json_schema(), indent=2)
        
        system_prompt = (
            "You are an AI research assistant. You MUST respond strictly in valid JSON format "
            f"conforming to the following JSON Schema:\n{json_schema_str}\n"
            "Do NOT wrap the response in markdown backticks or additional text."
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        return schema.model_validate_json(content)

    async def classify_query(self, query: str) -> QueryClassification:
        prompt = (
            f'Classify this query as SIMPLE or COMPLEX:\n"{query}"\n'
            'SIMPLE: Fact lookup, standard single-domain questions.\n'
            'COMPLEX: Open-ended, multi-perspective, strategic, or trade-off analysis.'
        )
        return await self._parse_json_schema(prompt, QueryClassification)

    async def plan_perspectives(self, query: str) -> PerspectivePlan:
        prompt = (
            f'Decompose the following complex query into 3 distinct operational perspectives '
            f'and generate 2 targeted search queries for each viewpoint:\n"{query}"'
        )
        return await self._parse_json_schema(prompt, PerspectivePlan)

    async def collect_evidence(self, query: str, perspective_name: str = "General") -> List[EvidenceItem]:
        # Integration point for search engines / APIs (Tavily, SearXNG, DuckDuckGo, vector DBs)
        print(f"  └─ Gathering evidence for perspective: [{perspective_name}]")
        return [
            EvidenceItem(
                id=f"doc_{perspective_name.lower().replace(' ', '_')}_1",
                source="Web Search Engine",
                perspective=perspective_name,
                content=f"Primary benchmark data and technical insights regarding '{query}' under '{perspective_name}' perspective."
            ),
            EvidenceItem(
                id=f"doc_{perspective_name.lower().replace(' ', '_')}_2",
                source="Industry Report",
                perspective=perspective_name,
                content=f"Secondary operational analysis and trade-offs related to '{query}'."
            )
        ]

    async def critique_evidence(self, query: str, evidence: List[EvidenceItem]) -> EvidenceCritique:
        evidence_str = "\n".join([f"[{e.id}] Perspective: {e.perspective}\nContent: {e.content}" for e in evidence])
        prompt = (
            f'Audit the following gathered evidence for query: "{query}"\n\n'
            f'Evidence List:\n{evidence_str}\n\n'
            'Check for grounding, identify conflicting facts, flag weak claims, '
            'and return the list of validated evidence IDs.'
        )
        return await self._parse_json_schema(prompt, EvidenceCritique)

    async def synthesize(self, query: str, valid_evidence: List[EvidenceItem], critique: EvidenceCritique) -> str:
        evidence_str = "\n".join([f"[{e.id}] ({e.perspective}): {e.content}" for e in valid_evidence])
        prompt = (
            f'Synthesize a comprehensive research report for the user query strictly grounded in validated evidence.\n\n'
            f'User Query: "{query}"\n\n'
            f'Validated Evidence:\n{evidence_str}\n\n'
            f'Identified Conflicts: {critique.conflicts_detected}\n'
            f'Flagged Weak Points: {critique.weak_evidence_flagged}\n\n'
            'Cite doc IDs inline in square brackets (e.g., [doc_id]).'
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

    async def update_epistemic_memory(self, query: str, critique: EvidenceCritique, synthesis: str) -> EpistemicState:
        prompt = (
            f'Analyze the synthesis report and critique context for query "{query}".\n'
            'Summarize current knowledge state into known_facts, uncertainties, disputed_claims, and unresolved_questions.\n\n'
            f'Synthesis Draft:\n{synthesis}'
        )
        return await self._parse_json_schema(prompt, EpistemicState)

    async def run(self, user_query: str) -> FinalResearchOutput:
        print(f"\n==========================================")
        print(f"Starting Pipeline using Groq [{self.model}]")
        print(f"==========================================\n")

        # 1. Query Classifier
        classification = await self.classify_query(user_query)
        print(f"[Query Classifier] Classified as: {classification.query_type}")
        print(f"Reasoning: {classification.reasoning}\n")

        all_evidence: List[EvidenceItem] = []

        # 2. Perspective Planning & Parallel Collection
        if classification.query_type == "SIMPLE":
            all_evidence = await self.collect_evidence(user_query)
        else:
            plan = await self.plan_perspectives(user_query)
            print(f"[Perspective Planner] Perspectives: {[p.name for p in plan.perspectives]}")
            
            tasks = [self.collect_evidence(user_query, p.name) for p in plan.perspectives]
            results = await asyncio.gather(*tasks)
            for res in results:
                all_evidence.extend(res)

        print(f"\n[Evidence Collector] Total items retrieved: {len(all_evidence)}")

        # 3. Evidence Critic
        print("[Evidence Critic] Auditing evidence quality and cross-source conflicts...")
        critique = await self.critique_evidence(user_query, all_evidence)
        valid_evidence = [e for e in all_evidence if e.id in critique.validated_evidence_ids]
        print(f"Validated Evidence Count: {len(valid_evidence)} / {len(all_evidence)}")

        # 4. Synthesis
        print("[Synthesis Agent] Generating evidence-backed final report...")
        synthesis = await self.synthesize(user_query, valid_evidence, critique)

        # 5. Epistemic Memory Update
        print("[Epistemic Memory] Updating knowledge state store...")
        epistemic_state = await self.update_epistemic_memory(user_query, critique, synthesis)

        return FinalResearchOutput(
            synthesized_response=synthesis,
            limitations=epistemic_state.uncertainties + epistemic_state.disputed_claims,
            epistemic_summary=epistemic_state
        )


# ==========================================
# 3. Pipeline Execution
# ==========================================

if __name__ == "__main__":
    pipeline = GroqDeepResearchPipeline(model_name="openai/gpt-oss-120b")

    query = "What are the trade-offs of transitioning a core financial system from a microservices architecture back to a modular monolith?"

    result = asyncio.run(pipeline.run(query))

    print("\n" + "="*50)
    print("FINAL RESEARCH REPORT")
    print("="*50)
    print(result.synthesized_response)

    print("\n" + "="*50)
    print("EPISTEMIC STATE & LIMITATIONS")
    print("="*50)
    print("Known Facts:", json.dumps(result.epistemic_summary.known_facts, indent=2))
    print("Limitations:", json.dumps(result.limitations, indent=2))