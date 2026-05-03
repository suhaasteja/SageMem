"""Tests for the dataset loader using a synthetic mini-dataset."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import EVAL_CATEGORIES
from src.data import dataset_stats, filter_qa, load_dataset

MINI_DATASET = [
    {
        "sample_id": "conv-001",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "2023-01-01 10:00",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "Hello Bob."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Hi Alice."},
            ],
            "session_2_date_time": "2023-01-02 10:00",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "How are you?"},
            ],
        },
        "qa": [
            {"question": "Who greeted Bob?", "answer": "Alice", "category": 1, "evidence": ["D1:1"]},
            {"question": "What did Alice say?", "answer": "Hello Bob.", "category": 2, "evidence": []},
            {"question": "When did Alice ask how Bob was?", "answer": "January 2nd", "category": 3, "evidence": []},
            {"question": "Who are the speakers?", "answer": "Alice and Bob", "category": 4, "evidence": []},
            {"question": "What is Alice's favorite color?", "answer": "unknown", "category": 5, "evidence": []},
        ],
    }
]


def test_load_dataset_structure():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_DATASET, f)
        path = Path(f.name)

    convs = load_dataset(path)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.sample_id == "conv-001"
    assert 1 in conv.sessions and 2 in conv.sessions
    assert len(conv.sessions[1]) == 2
    assert len(conv.qa) == 5


def test_session_dates_parsed():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_DATASET, f)
        path = Path(f.name)

    convs = load_dataset(path)
    assert convs[0].session_dates[1] == "2023-01-01 10:00"
    assert convs[0].session_dates[2] == "2023-01-02 10:00"


def test_latest_date():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_DATASET, f)
        path = Path(f.name)

    convs = load_dataset(path)
    assert convs[0].latest_date == "2023-01-02 10:00"


def test_filter_qa_excludes_cat5():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_DATASET, f)
        path = Path(f.name)

    convs = load_dataset(path)
    filtered = filter_qa(convs[0].qa, EVAL_CATEGORIES)
    assert len(filtered) == 4
    assert all(q.category != 5 for q in filtered)


def test_dataset_stats():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MINI_DATASET, f)
        path = Path(f.name)

    convs = load_dataset(path)
    stats = dataset_stats(convs, EVAL_CATEGORIES)
    assert stats["conversations"] == 1
    assert stats["total_qa"] == 5
    assert stats["filtered_qa"] == 4
