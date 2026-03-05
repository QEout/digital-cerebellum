"""
Visualization module — 3D real-time cerebellum dashboard.

    pip install digital-cerebellum[viz]
    python examples/viz_demo.py
"""

from digital_cerebellum.viz.event_bus import event_bus, CerebellumEvent

__all__ = ["event_bus", "CerebellumEvent"]
