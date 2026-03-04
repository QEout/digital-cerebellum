"""
Digital Cerebellum — a cerebellar-inspired cognitive architecture.

Three ways to use:

1. As a cerebellum SDK (plug into your own agent)::

    from digital_cerebellum import DigitalCerebellum
    from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

    cb = DigitalCerebellum()
    cb.register_microzone(ToolCallMicrozone())
    result = cb.evaluate("tool_call", {"tool_name": "send_email", ...})

2. As a step monitor (wrap any agent's execution loop)::

    from digital_cerebellum import StepMonitor

    monitor = StepMonitor()
    pred = monitor.before_step(action="click save", state="editor open")
    # ... execute action ...
    verdict = monitor.after_step(outcome="save dialog appeared")

3. As a reference brain (LLM + cerebellum, for demos)::

    from digital_cerebellum import DigitalBrain

    brain = DigitalBrain.from_yaml()
    brain.register_tool("search", search_fn, "Search the web")
    result = brain.think("What's the weather in Tokyo?")
"""

from digital_cerebellum.main import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.brain import DigitalBrain
from digital_cerebellum.monitor import StepMonitor
from digital_cerebellum.emergence import SomaticMarker, CuriosityDrive, SelfModel
from digital_cerebellum.micro_ops import MicroOpEngine, MicroOpConfig

__all__ = [
    "DigitalCerebellum", "CerebellumConfig", "DigitalBrain",
    "StepMonitor",
    "SomaticMarker", "CuriosityDrive", "SelfModel",
    "MicroOpEngine", "MicroOpConfig",
]
__version__ = "0.6.0"
