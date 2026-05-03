"""Session-by-session ingestion into a memory adapter."""
from ..adapters.base import MemoryAdapter
from ..data import Conversation


def ingest_conversation(adapter: MemoryAdapter, conv: Conversation) -> None:
    """Reset adapter state and ingest all sessions in chronological order."""
    adapter.reset(conv.sample_id)

    for session_num in sorted(conv.sessions.keys()):
        turns = conv.sessions[session_num]
        timestamp = conv.session_dates.get(session_num, "")
        turn_dicts = [
            {"speaker": t.speaker, "text": t.text, "dia_id": t.dia_id}
            for t in turns
        ]
        adapter.add_session(
            conversation_id=conv.sample_id,
            session_id=str(session_num),
            timestamp=timestamp,
            turns=turn_dicts,
        )
