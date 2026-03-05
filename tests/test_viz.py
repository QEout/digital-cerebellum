"""Tests for the visualization module: EventBus + server basics."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from digital_cerebellum.viz.event_bus import EventBus, CerebellumEvent, event_bus


class TestCerebellumEvent:
    def test_to_dict(self):
        evt = CerebellumEvent(
            event_type="encode", module="FeatureEncoder",
            data={"step": 1}, timestamp=1000.0,
        )
        d = evt.to_dict()
        assert d["event_type"] == "encode"
        assert d["module"] == "FeatureEncoder"
        assert d["data"]["step"] == 1
        assert d["timestamp"] == 1000.0

    def test_default_timestamp(self):
        before = time.time()
        evt = CerebellumEvent(event_type="x", module="M")
        after = time.time()
        assert before <= evt.timestamp <= after


class TestEventBus:
    def test_emit_no_subscribers(self):
        bus = EventBus()
        bus.emit("encode", "FeatureEncoder", step=1)

    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda evt: received.append(evt))
        bus.emit("predict", "PredictionEngine", confidence=0.95)
        assert len(received) == 1
        assert received[0].event_type == "predict"
        assert received[0].data["confidence"] == 0.95

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda evt: received.append(evt)
        bus.subscribe(cb)
        bus.emit("a", "M")
        bus.unsubscribe(cb)
        bus.emit("b", "M")
        assert len(received) == 1

    def test_clear(self):
        bus = EventBus()
        bus.subscribe(lambda e: None)
        bus.subscribe(lambda e: None)
        assert len(bus._subscribers) == 2
        bus.clear()
        assert len(bus._subscribers) == 0

    def test_error_in_subscriber_doesnt_crash(self):
        bus = EventBus()
        ok_received = []
        bus.subscribe(lambda e: (_ for _ in ()).throw(ValueError("boom")))
        bus.subscribe(lambda e: ok_received.append(e))
        bus.emit("test", "M")
        assert len(ok_received) == 1

    def test_global_singleton(self):
        assert isinstance(event_bus, EventBus)


class TestServerImport:
    def test_fastapi_app_exists(self):
        from digital_cerebellum.viz.server import app
        assert app is not None
        assert app.title == "Digital Cerebellum Viz"

    def test_static_dir_exists(self):
        from digital_cerebellum.viz.server import STATIC_DIR
        assert (STATIC_DIR / "index.html").exists()

    def test_event_queue_integration(self):
        from digital_cerebellum.viz.server import _event_queue, _on_event
        evt = CerebellumEvent(event_type="test", module="Test", data={"v": 42})
        _on_event(evt)
        d = _event_queue.get_nowait()
        assert d["event_type"] == "test"
        assert d["data"]["v"] == 42
