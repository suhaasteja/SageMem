"""Query phase: for each QA item, retrieve context and generate an answer."""
import time
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from ..adapters.base import MemoryAdapter
from ..data import Conversation, QAItem
from ..prompts import ANSWER_SYSTEM

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Prediction:
    conversation_id: str
    qa_id: str
    category: int
    question: str
    gold_answer: str
    predicted_answer: str
    retrieved_context: str
    retrieval_tokens: int
    prompt_tokens: int
    completion_tokens: int
    search_latency_ms: float
    generation_latency_ms: float


def query_conversation(
    adapter: MemoryAdapter,
    conv: Conversation,
    qa_items: list[QAItem],
    *,
    client: OpenAI,
    answer_model: str,
    top_k: int = 30,
) -> list[Prediction]:
    """Run all QA items for one conversation. Adapter must already be ingested."""
    predictions = []
    today = conv.latest_date or "unknown"

    for i, qa in enumerate(qa_items):
        search_result = adapter.search(conv.sample_id, qa.question, top_k=top_k)

        prompt = ANSWER_SYSTEM.format(
            today=today,
            retrieved_context=search_result.context,
            question=qa.question,
        )

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=answer_model,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        gen_latency = (time.perf_counter() - t0) * 1000

        predicted = (response.choices[0].message.content or "").strip()
        usage = response.usage

        predictions.append(Prediction(
            conversation_id=conv.sample_id,
            qa_id=f"{conv.sample_id}-q{i}",
            category=qa.category,
            question=qa.question,
            gold_answer=qa.answer,
            predicted_answer=predicted,
            retrieved_context=search_result.context,
            retrieval_tokens=search_result.retrieved_tokens,
            prompt_tokens=usage.prompt_tokens if usage else _count_tokens(prompt),
            completion_tokens=usage.completion_tokens if usage else _count_tokens(predicted),
            search_latency_ms=search_result.latency_ms,
            generation_latency_ms=gen_latency,
        ))

    return predictions
