"""
GSRS MCP Server - Answer Generator
Generates answers from evidence with citations.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.models.api import Citation
from app.services.evidence import EvidenceResult
from app.services.llm import LLMService


# System prompt templates for answer generation
_ANSWER_SYSTEM_PROMPT = """\
You are a GSRS (Global Substance Registration System) assistant. \
Answer the user's question using ONLY the provided evidence from the GSRS database.

Rules:
- Use ONLY the provided evidence to answer
- Do NOT invent identifiers, structures, relationships, or properties
- If evidence conflicts, say so explicitly
- If evidence is insufficient, indicate that you cannot verify the answer
- Cite the evidence for every material claim using [1], [2], etc. format
- Be precise and factual

For counting/list questions (e.g., "How many identifiers..."):
- Count items carefully from the evidence
- List all items found, grouped by type if possible
- If evidence is incomplete, say so

Answer style: {answer_style}
"""

_ANSWER_CONCISE = "concise - give a brief, direct answer"
_ANSWER_STANDARD = "standard - give a clear, complete answer"
_ANSWER_DETAILED = "detailed - give a thorough, comprehensive answer"

_STYLE_MAP = {
    "concise": _ANSWER_CONCISE,
    "standard": _ANSWER_STANDARD,
    "detailed": _ANSWER_DETAILED,
}


@dataclass
class GenerationTrace:
    """Internal trace describing how an answer was produced."""

    mode: str
    llm_attempted: bool
    used_llm: bool
    fallback_used: bool
    evidence_count: int
    citation_count: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "llm_attempted": self.llm_attempted,
            "used_llm": self.used_llm,
            "fallback_used": self.fallback_used,
            "evidence_count": self.evidence_count,
            "citation_count": self.citation_count,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class AnswerGenerator:
    """
    Generates answers from evidence with proper citations.
    - Answers strictly from evidence
    - Cites every material claim
    - Keeps unsupported content out
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        use_llm: bool = True,
    ):
        self.llm = llm_service
        self.use_llm = use_llm and llm_service is not None
        self.last_trace: Optional[GenerationTrace] = None

    def generate(
        self,
        query: str,
        evidence: List[EvidenceResult],
        answer_style: str = "standard",
    ) -> Tuple[str, List[Citation]]:
        """
        Generate an answer from evidence.

        Args:
            query: Original query
            evidence: List of evidence results
            answer_style: "concise", "standard", or "detailed"

        Returns:
            (answer_text, citations) tuple
        """
        if not evidence:
            self.last_trace = GenerationTrace(
                mode="no_evidence",
                llm_attempted=False,
                used_llm=False,
                fallback_used=True,
                evidence_count=0,
                citation_count=0,
            )
            return "I cannot answer this question based on the available GSRS evidence.", []

        if self.use_llm and self.llm:
            return self._generate_llm_answer(query, evidence, answer_style)

        answer, citations = self._generate_template_answer(query, evidence, answer_style)
        self.last_trace = GenerationTrace(
            mode="template",
            llm_attempted=False,
            used_llm=False,
            fallback_used=True,
            evidence_count=len(evidence),
            citation_count=len(citations),
        )
        return answer, citations

    def _generate_llm_answer(
        self,
        query: str,
        evidence: List[EvidenceResult],
        answer_style: str,
    ) -> Tuple[str, List[Citation]]:
        """Generate answer using LLM with evidence context."""
        style = _STYLE_MAP.get(answer_style, _ANSWER_STANDARD)
        system_prompt = _ANSWER_SYSTEM_PROMPT.format(answer_style=style)

        # Build evidence context
        evidence_context = self._build_evidence_context(evidence)

        user_prompt = f"""\
Question: {query}

Evidence:
{evidence_context}

Answer the question using only the evidence above. Cite each claim with [1], [2], etc."""

        try:
            answer = self.llm.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            )
            citations = [e.citation for e in evidence if e.score > 0.3]
            self.last_trace = GenerationTrace(
                mode="llm",
                llm_attempted=True,
                used_llm=True,
                fallback_used=False,
                evidence_count=len(evidence),
                citation_count=len(citations),
            )
            return answer, citations
        except Exception as exc:
            # Fallback to template
            answer, citations = self._generate_template_answer(query, evidence, answer_style)
            self.last_trace = GenerationTrace(
                mode="template_fallback",
                llm_attempted=True,
                used_llm=False,
                fallback_used=True,
                evidence_count=len(evidence),
                citation_count=len(citations),
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
            return answer, citations

    def _generate_template_answer(
        self,
        query: str,
        evidence: List[EvidenceResult],
        answer_style: str,
    ) -> Tuple[str, List[Citation]]:
        """Generate answer using template-based approach."""
        citations = [e.citation for e in evidence if e.score > 0.3]

        if len(evidence) == 1:
            e = evidence[0]
            answer = (
                "Direct answer:\n"
                f"{e.snippet or e.document.text[:500]}\n\n"
                f"Supporting section: {e.citation.section}"
            )
        else:
            parts = []
            for i, e in enumerate(evidence[:5], 1):
                snippet = (e.snippet or e.document.text[:300]).strip()
                parts.append(f"[{i}] ({e.citation.section}): {snippet}")

            answer = "Supporting evidence:\n\n" + "\n\n".join(parts)

        return answer, citations

    def _build_evidence_context(self, evidence: List[EvidenceResult]) -> str:
        """Build evidence context for LLM prompt."""
        parts = []
        for i, e in enumerate(evidence[:10], 1):
            snippet = (e.snippet or e.document.text[:500]).strip()
            section = e.citation.section
            parts.append(f"[{i}] Section: {section}\n{snippet}")
        return "\n\n---\n\n".join(parts)
