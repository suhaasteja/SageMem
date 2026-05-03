"""Dataset loader and schema validation for locomo10.json."""
import json
import re
from pathlib import Path
from pydantic import BaseModel, field_validator


class Turn(BaseModel):
    speaker: str
    dia_id: str
    text: str
    blip_caption: str | None = None
    query: str | None = None


class QAItem(BaseModel):
    question: str
    answer: str
    category: int
    evidence: list[str] = []

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_answer(cls, v):
        return str(v) if not isinstance(v, str) else v


class Conversation(BaseModel):
    sample_id: str
    sessions: dict[int, list[Turn]]   # session_number -> turns
    session_dates: dict[int, str]      # session_number -> datetime string
    qa: list[QAItem]

    @property
    def latest_date(self) -> str | None:
        if not self.session_dates:
            return None
        latest_num = max(self.session_dates.keys())
        return self.session_dates[latest_num]


def _parse_sessions(conv_data: dict) -> tuple[dict[int, list[Turn]], dict[int, str]]:
    """Extract session turns and dates from the flat key structure of locomo10.json."""
    sessions: dict[int, list[Turn]] = {}
    dates: dict[int, str] = {}

    session_keys = sorted(
        (k for k in conv_data if re.match(r"^session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    )

    for key in session_keys:
        num = int(key.split("_")[1])
        raw_turns = conv_data[key]
        turns = []
        for t in raw_turns:
            if not isinstance(t, dict):
                continue
            turns.append(Turn(
                speaker=t.get("speaker", ""),
                dia_id=t.get("dia_id", ""),
                text=t.get("text", ""),
                blip_caption=t.get("blip_caption"),
                query=t.get("query"),
            ))
        sessions[num] = turns

        date_key = f"session_{num}_date_time"
        if date_key in conv_data:
            dates[num] = conv_data[date_key]

    return sessions, dates


def load_dataset(path: Path) -> list[Conversation]:
    """Load and validate locomo10.json. Returns list of Conversation objects."""
    raw = json.loads(path.read_text())
    conversations = []

    for item in raw:
        sample_id = item["sample_id"]
        conv_data = item["conversation"]
        sessions, dates = _parse_sessions(conv_data)

        qa_items = []
        for q in item.get("qa", []):
            qa_items.append(QAItem(
                question=q["question"],
                answer=q.get("answer") or q.get("adversarial_answer", ""),
                category=q["category"],
                evidence=q.get("evidence", []),
            ))

        conversations.append(Conversation(
            sample_id=sample_id,
            sessions=sessions,
            session_dates=dates,
            qa=qa_items,
        ))

    return conversations


def filter_qa(items: list[QAItem], categories: set[int]) -> list[QAItem]:
    """Return only QA items whose category is in the given set."""
    return [q for q in items if q.category in categories]


def dataset_stats(conversations: list[Conversation], eval_categories: set[int]) -> dict:
    total_qa = sum(len(c.qa) for c in conversations)
    filtered_qa = sum(len(filter_qa(c.qa, eval_categories)) for c in conversations)
    return {
        "conversations": len(conversations),
        "total_qa": total_qa,
        "filtered_qa": filtered_qa,
    }
