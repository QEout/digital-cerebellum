"""
Digital Cerebellum — a cerebellar-inspired cognitive architecture.

Two ways to use:

1. As a cerebellum SDK (plug into your own agent)::

    from digital_cerebellum import DigitalCerebellum
    from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

    cb = DigitalCerebellum()
    cb.register_microzone(ToolCallMicrozone())
    result = cb.evaluate("tool_call", {"tool_name": "send_email", ...})

2. As a complete brain (LLM + cerebellum, no framework needed)::

    from digital_cerebellum import DigitalBrain

    brain = DigitalBrain.from_yaml()
    brain.register_tool("search", search_fn, "Search the web")
    result = brain.think("What's the weather in Tokyo?")
"""

from digital_cerebellum.main import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.brain import DigitalBrain

__all__ = ["DigitalCerebellum", "CerebellumConfig", "DigitalBrain"]
__version__ = "0.1.0"
