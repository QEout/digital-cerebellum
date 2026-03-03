"""
Digital Cerebellum — a cerebellar-inspired prediction-correction engine for LLM Agents.

Quick start::

    from digital_cerebellum import DigitalCerebellum
    from digital_cerebellum.microzones.tool_call import ToolCallMicrozone

    cb = DigitalCerebellum()
    cb.register_microzone(ToolCallMicrozone())

    result = cb.evaluate("tool_call", {
        "tool_name": "send_email",
        "tool_params": {"to": "alice@example.com"},
    })
"""

from digital_cerebellum.main import DigitalCerebellum, CerebellumConfig

__all__ = ["DigitalCerebellum", "CerebellumConfig"]
__version__ = "0.1.0"
