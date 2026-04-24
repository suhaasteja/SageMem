"""Conflict resolution policies for L3 optimistic concurrency conflicts.

Stage 0: stubs. Wired in Stage 4.
"""

from typing import Any


def last_write_wins(local: Any, remote: Any) -> Any:
    """Return remote value unconditionally — simplest resolution policy."""
    return remote


def merge_additive(local: Any, remote: Any) -> Any:
    """Merge two dict values by union, preferring remote on key conflicts.

    Falls back to last_write_wins for non-dict values.
    """
    if isinstance(local, dict) and isinstance(remote, dict):
        return {**local, **remote}
    return last_write_wins(local, remote)
