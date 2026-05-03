"""Tests for the judge using a mock OpenAI client."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.judge import JudgeResult, judge


def _mock_client(label: str, reasoning: str = "test") -> MagicMock:
    response_content = json.dumps({"label": label, "reasoning": reasoning})
    choice = MagicMock()
    choice.message.content = response_content
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_judge_correct_for_paraphrase():
    client = _mock_client("CORRECT", "Paraphrase of the same fact.")
    result = judge("Who is Alice?", "A person", "Alice is a person", client=client, model="gpt-4o-mini")
    assert result.label == "CORRECT"
    assert isinstance(result.reasoning, str)


def test_judge_wrong_for_incorrect_fact():
    client = _mock_client("WRONG", "Different fact provided.")
    result = judge("Who is Alice?", "A doctor", "Alice is a teacher", client=client, model="gpt-4o-mini")
    assert result.label == "WRONG"


def test_judge_wrong_for_no_info_when_answer_exists():
    client = _mock_client("WRONG", "Model refused but answer exists.")
    result = judge("Who is Alice?", "A doctor", "No information available", client=client, model="gpt-4o-mini")
    assert result.label == "WRONG"


def test_judge_invalid_label_defaults_to_wrong():
    client = _mock_client("MAYBE")
    result = judge("Q", "A", "B", client=client, model="gpt-4o-mini")
    assert result.label == "WRONG"


def test_judge_model_recorded():
    client = _mock_client("CORRECT")
    result = judge("Q", "A", "A", client=client, model="gpt-4o-mini")
    assert result.judge_model == "gpt-4o-mini"
