"""Evaluate phase: run judge + compute F1/BLEU-1 over predictions."""
from dataclasses import asdict, dataclass

from openai import OpenAI

from ..judge import JudgeResult, judge
from ..metrics import bleu1, token_f1
from ..pipeline.query import Prediction


@dataclass
class EvaluatedPrediction:
    # All fields from Prediction
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
    # Evaluation fields
    judge_label: str
    judge_reasoning: str
    judge_model: str
    f1: float
    bleu1_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(
    predictions: list[Prediction],
    *,
    client: OpenAI,
    judge_model: str,
) -> list[EvaluatedPrediction]:
    results = []
    for pred in predictions:
        verdict: JudgeResult = judge(
            question=pred.question,
            gold_answer=pred.gold_answer,
            generated_answer=pred.predicted_answer,
            client=client,
            model=judge_model,
        )
        results.append(EvaluatedPrediction(
            **{k: v for k, v in pred.__dict__.items()},
            judge_label=verdict.label,
            judge_reasoning=verdict.reasoning,
            judge_model=verdict.judge_model,
            f1=token_f1(pred.predicted_answer, pred.gold_answer),
            bleu1_score=bleu1(pred.predicted_answer, pred.gold_answer),
        ))
    return results
