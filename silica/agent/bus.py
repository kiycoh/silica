"""Thread-safe synchronous pub-sub event bus.

BUS is a process-global singleton importable anywhere. Tests should
monkeypatch silica.agent.bus.BUS with a fresh EventBus() per test to
avoid cross-test contamination.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Synchronous fan-out pub-sub with wildcard topic matching.

    publish() calls all matching subscribers inline on the calling thread.
    Subscriber exceptions are logged and never propagate to the caller.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Callable[[Any], None]]] = {}

    def subscribe(self, topic: str, fn: Callable[[Any], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(fn)

    def unsubscribe(self, topic: str, fn: Callable[[Any], None]) -> None:
        with self._lock:
            subs = self._subscribers.get(topic, [])
            try:
                subs.remove(fn)
            except ValueError:
                pass

    def publish(self, topic: str, event: Any) -> None:
        with self._lock:
            snapshot = [(p, list(fns)) for p, fns in self._subscribers.items()]
        for pattern, fns in snapshot:
            if not self._matches(pattern, topic):
                continue
            for fn in fns:
                try:
                    fn(event)
                except Exception:
                    logger.exception("Event subscriber failed on topic %r", topic)

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True
        if pattern.endswith("/*"):
            return topic.startswith(pattern[:-1])
        return False


BUS = EventBus()
