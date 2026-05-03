"""LLM-as-judge wrapper. Returns CORRECT/WRONG with reasoning."""
import json
import time
from dataclasses import dataclass

from openai import OpenAI

from .prompts import JUDGE_SYSTEM


@dataclass
class JudgeResult:
    label: str        # "CORRECT" or "WRONG"
    reasoning: str
    judge_model: str


def judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    *,
    client: OpenAI,
    model: str,
) -> JudgeResult:
    """Call the LLM judge and return a binary verdict."""
    prompt = JUDGE_SYSTEM.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)

    label = data.get("label", "WRONG").upper()
    if label not in {"CORRECT", "WRONG"}:
        label = "WRONG"

    return JudgeResult(
        label=label,
        reasoning=data.get("reasoning", ""),
        judge_model=model,
    )
