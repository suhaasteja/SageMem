"""F1 (token overlap) and BLEU-1 metrics, SQuAD-style normalisation."""
import re
import string
from collections import Counter

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    articles = {"a", "an", "the"}
    tokens = [t for t in tokens if t not in articles]
    return " ".join(tokens)


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (SQuAD style)."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    """Unigram BLEU (BLEU-1) between prediction and gold."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    matches = sum(
        min(count, gold_counts[token])
        for token, count in Counter(pred_tokens).items()
    )
    return matches / len(pred_tokens)
