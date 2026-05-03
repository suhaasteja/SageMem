"""Tests for F1 and BLEU-1 with known-answer fixtures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.metrics import bleu1, token_f1


def test_f1_exact_match():
    assert token_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_f1_partial_overlap():
    score = token_f1("the cat", "cat sat on mat")
    assert 0 < score < 1.0


def test_f1_no_overlap():
    assert token_f1("hello world", "foo bar baz") == 0.0


def test_f1_article_stripping():
    # "a" and "the" are stripped; should still match core tokens
    assert token_f1("a cat", "the cat") == 1.0


def test_f1_empty_prediction():
    assert token_f1("", "some answer") == 0.0


def test_f1_empty_gold():
    assert token_f1("some answer", "") == 0.0


def test_bleu1_exact():
    assert bleu1("the cat sat", "the cat sat") == 1.0


def test_bleu1_partial():
    score = bleu1("the cat", "the cat sat on the mat")
    assert 0 < score <= 1.0


def test_bleu1_no_overlap():
    assert bleu1("hello world", "foo bar baz") == 0.0


def test_bleu1_empty():
    assert bleu1("", "something") == 0.0
