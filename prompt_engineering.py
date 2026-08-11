"""Prompt variants and lightweight evaluation helpers for document analysis."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, List


@dataclass(frozen=True)
class PromptVariant:
    name: str
    task: str
    template: str

    def render(self, **values: str) -> str:
        return self.template.format(**values)


PROMPT_VARIANTS = {
    "summary": (
        PromptVariant(
            "concise", "summary", "Summarize this document in 4 short bullet points.\n{document}"
        ),
        PromptVariant(
            "structured_grounded",
            "summary",
            """You are a precise document analyst. Summarize only the supplied document.
Return exactly 4 bullets. Each bullet must state one important claim and preserve
important numbers, dates, and qualifications. Do not add information not present.
DOCUMENT:
{document}""",
        ),
        PromptVariant(
            "audience_first",
            "summary",
            """Write an executive summary for a busy reader using only this document.
Include purpose, major findings, evidence, and implications in 4 labeled bullets.
Mark missing information as 'Not stated'.
DOCUMENT:
{document}""",
        ),
    ),
    "key_points": (
        PromptVariant(
            "evidence_tagged",
            "key_points",
            """Identify up to 5 key points from this document. For every point, include
the claim followed by its evidence in parentheses. If the document does not support
5 points, return fewer rather than inventing any.
DOCUMENT:
{document}""",
        ),
        PromptVariant(
            "bullet_points",
            "key_points",
            """Extract 5 key insights from this document and return them as a simple
bullet list. Keep each point brief and factual, using only language from the source.
DOCUMENT:
{document}""",
        ),
        PromptVariant(
            "executive_summary",
            "key_points",
            """Create a short executive summary of the document in 4-6 bullet points.
Focus on the most important outcomes, recommendations, and implications.
DOCUMENT:
{document}""",
        ),
    ),
    "qa": (
        PromptVariant(
            "concise", "qa", "Answer the question using only this context.\nContext: {context}\nQuestion: {question}"
        ),
        PromptVariant(
            "structured_grounded",
            "qa",
            """You answer questions from source context only.
If the answer is not explicitly supported, say: Not found in the document.
Give a concise answer. Then cite each supporting source by Document Name,
Page Number, Paragraph Number, and Section Name when available.
Never use outside knowledge.
CONTEXT:
{context}
QUESTION:
{question}""",
        ),
        PromptVariant(
            "reasoned_grounded",
            "qa",
            """Answer using only the provided context. First identify the supporting
fact(s), then provide the answer in 2-4 sentences. Distinguish stated facts from
uncertainty and respond 'Not found in the document' when evidence is absent.
CONTEXT:
{context}
QUESTION:
{question}""",
        ),
    ),
}

RECOMMENDED_VARIANT = "structured_grounded"


def build_prompt(task: str, variant_name: str | None = None, **values: str) -> str:
    """Render a prompt for a task using a named variant."""
    variants = PROMPT_VARIANTS[task]
    if variant_name is None or variant_name == "recommended":
        variant_name = RECOMMENDED_VARIANT
    variant = next((item for item in variants if item.name == variant_name), None)
    if variant is None:
        raise ValueError(f"Unknown prompt variant '{variant_name}' for task '{task}'")
    return variant.render(**values)


def get_prompt_variant_names(task: str) -> List[str]:
    return [variant.name for variant in PROMPT_VARIANTS[task]]


def get_prompt_variants(task: str) -> List[PromptVariant]:
    return list(PROMPT_VARIANTS[task])


def estimate_tokens(prompt: str) -> int:
    """Estimate prompt tokens without requiring a model tokenizer."""
    return max(1, (len(prompt) + 3) // 4)


def evaluate_prompt_variants(
    task: str,
    cases: Iterable[dict[str, str]],
    invoke: Callable[[str], str],
    score: Callable[[dict[str, str], str], float],
) -> list[dict[str, object]]:
    """Compare quality, latency, and estimated input tokens for each variant."""
    results = []
    for variant in PROMPT_VARIANTS[task]:
        scores = []
        durations = []
        tokens = []
        for case in cases:
            prompt = variant.render(**case)
            started = perf_counter()
            response = invoke(prompt)
            durations.append(perf_counter() - started)
            scores.append(float(score(case, response)))
            tokens.append(estimate_tokens(prompt))
        results.append(
            {
                "task": task,
                "variant": variant.name,
                "quality": sum(scores) / len(scores) if scores else 0.0,
                "response_time_seconds": sum(durations) / len(durations) if durations else 0.0,
                "estimated_input_tokens": sum(tokens),
                "case_count": len(scores),
            }
        )
    return results
