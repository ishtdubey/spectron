import asyncio
import os
import json
import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tavily import TavilyClient
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
    title: Optional[str] = None
    source: str
    url: Optional[str] = None
    perspective: Optional[str] = None
    content: str
    relevance_score: Optional[float] = None

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
        tavily_api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b"
    ):
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        tavily_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable or .env file is required.")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY environment variable or .env file is required.")

        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.tavily = TavilyClient(api_key=tavily_key)
        self.model = model_name

    async def _parse_json_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
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

    async def collect_evidence(
        self, 
        query: str, 
        perspective_name: str = "General",
        search_queries: Optional[List[str]] = None
    ) -> List[EvidenceItem]:
        print(f"  └─ Gathering evidence via Tavily for perspective: [{perspective_name}]")
        queries = search_queries or [query]

        tasks = [
            asyncio.to_thread(
                self.tavily.search,
                query=search_query,
                search_depth="advanced",
                max_results=3
            )
            for search_query in queries
        ]
        
        responses = await asyncio.gather(*tasks)
        evidence_items = []
        seen_urls = set()
        counter = 1

        for response in responses:
            for result in response.get("results", []):
                url = result.get("url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)

                doc_id = f"doc_{perspective_name.lower().replace(' ', '_')}_{counter}"
                counter += 1

                evidence_items.append(
                    EvidenceItem(
                        id=doc_id,
                        title=result.get("title"),
                        source=result.get("title") or url or "Unknown Source",
                        url=url,
                        perspective=perspective_name,
                        content=result.get("content", "No content retrieved."),
                        relevance_score=result.get("score")
                    )
                )

        return evidence_items

    async def critique_evidence(self, query: str, evidence: List[EvidenceItem]) -> EvidenceCritique:
        evidence_str = "\n".join([
            f"[{e.id}] Perspective: {e.perspective} | Source: {e.source}\nContent: {e.content}" 
            for e in evidence
        ])
        prompt = (
            f'Audit the following gathered evidence for query: "{query}"\n\n'
            f'Evidence List:\n{evidence_str}\n\n'
            'Check for grounding, identify conflicting facts, flag weak claims, '
            'and return the list of validated evidence IDs.'
        )
        return await self._parse_json_schema(prompt, EvidenceCritique)

    async def synthesize(self, query: str, valid_evidence: List[EvidenceItem], critique: EvidenceCritique) -> str:
        valid_ids = [e.id for e in valid_evidence]
        valid_ids_str = ", ".join(valid_ids) if valid_ids else "None"
        
        evidence_str = "\n".join([
            f"[{e.id}] (Perspective: {e.perspective or 'General'}, Source: {e.source}, URL: {e.url or 'N/A'}):\n{e.content}" 
            for e in valid_evidence
        ])
        
        system_prompt = (
            "You are an expert research analyst.\n\n"
            "STRICT CITATION RULES:\n"
            "1. Use square brackets ONLY for evidence citations (e.g. [doc_id]).\n"
            "2. Every citation inside square brackets MUST correspond to an exact EvidenceItem ID from the list below.\n"
            "3. Never invent, abbreviate, or modify citation IDs.\n"
            "4. NEVER write bracketed labels such as [User], [Source], [Evidence], [Context], [Query], or [Unknown].\n"
            "5. If referring to the user's question, prompt, or scenario, write 'the user' without square brackets.\n"
            "6. Use ONLY claims supported by the validated evidence. Every factual claim must be traceable to at least one valid EvidenceItem ID.\n"
            "7. If the evidence is insufficient, explicitly state that the claim cannot be established.\n\n"
            f"VALID EVIDENCE IDs YOU ARE ALLOWED TO CITE: {valid_ids_str}"
        )

        user_prompt = (
            f'Synthesize a comprehensive research report for the user query strictly grounded in validated evidence.\n\n'
            f'User Query: "{query}"\n\n'
            f'Validated Evidence:\n{evidence_str}\n\n'
            f'Identified Conflicts: {critique.conflicts_detected}\n'
            f'Flagged Weak Points: {critique.weak_evidence_flagged}\n'
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        synthesis_text = response.choices[0].message.content

        # Post-processing fallback: remove non-doc bracketed labels like [User] or [Source] if any slipped through
        synthesis_text = re.sub(r'\[(User|Source|Evidence|Context|Query|Unknown)\]', r'\1', synthesis_text, flags=re.IGNORECASE)

        return synthesis_text

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

        classification = await self.classify_query(user_query)
        print(f"[Query Classifier] Classified as: {classification.query_type}")

        all_evidence: List[EvidenceItem] = []

        if classification.query_type == "SIMPLE":
            all_evidence = await self.collect_evidence(
                query=user_query,
                perspective_name="General",
                search_queries=[user_query]
            )
        else:
            plan = await self.plan_perspectives(user_query)
            print(f"[Perspective Planner] Perspectives: {[p.name for p in plan.perspectives]}")
            
            tasks = [
                self.collect_evidence(
                    query=user_query,
                    perspective_name=p.name,
                    search_queries=p.search_queries
                )
                for p in plan.perspectives
            ]
            results = await asyncio.gather(*tasks)
            for res in results:
                all_evidence.extend(res)

        critique = await self.critique_evidence(user_query, all_evidence)
        valid_evidence = [e for e in all_evidence if e.id in critique.validated_evidence_ids]

        synthesis = await self.synthesize(user_query, valid_evidence, critique)
        epistemic_state = await self.update_epistemic_memory(user_query, critique, synthesis)

        return FinalResearchOutput(
            synthesized_response=synthesis,
            limitations=epistemic_state.uncertainties + epistemic_state.disputed_claims,
            epistemic_summary=epistemic_state
        )


if __name__ == "__main__":
    pipeline = GroqDeepResearchPipeline(model_name="openai/gpt-oss-120b")
    query = "What are the trade-offs of transitioning a core financial system from a microservices architecture back to a modular monolith?"
    result = asyncio.run(pipeline.run(query))
    print(result.synthesized_response)