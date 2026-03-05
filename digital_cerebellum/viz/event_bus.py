"""
EventBus — lightweight pub/sub for streaming cerebellum internals.

When no subscribers are registered, emit() is a no-op (~0 overhead).
The viz server subscribes and forwards events over WebSocket.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class CerebellumEvent:
    """A single event emitted by a cerebellum module."""
    event_type: str
    module: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class EventBus:
    """Global event bus — singleton used by all cerebellum components."""

    def __init__(self):
        self._subscribers: list[Callable[[CerebellumEvent], Any]] = []

    def subscribe(self, callback: Callable[[CerebellumEvent], Any]):
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[CerebellumEvent], Any]):
        self._subscribers = [s for s in self._subscribers if s is not callback]

    def emit(self, event_type: str, module: str, **data: Any):
        if not self._subscribers:
            return
        evt = CerebellumEvent(event_type=event_type, module=module, data=data)
        for sub in self._subscribers:
            try:
                sub(evt)
            except Exception:
                pass

    def clear(self):
        self._subscribers.clear()


event_bus = EventBus()
