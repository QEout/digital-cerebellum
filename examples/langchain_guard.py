"""
Example: Using Digital Cerebellum as a LangChain tool-call guard.

Wrap any LangChain agent with a cerebellar pre-evaluation layer.
The cerebellum intercepts tool invocations before execution and
blocks dangerous ones in < 50ms.

Usage:
    pip install langchain langchain-openai
    python examples/langchain_guard.py

Note: This example shows the integration pattern. Adapt the
`CerebellumGuard` class to your specific LangChain setup.
"""

from __future__ import annotations

from typing import Any

from digital_cerebellum import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone


class CerebellumGuard:
    """
    Drop-in guard for LangChain tool execution.

    Usage with LangChain::

        from langchain.agents import AgentExecutor

        guard = CerebellumGuard()

        # Option A: Wrap individual tool calls
        result = guard.check(tool_name="run_command", tool_params={"command": "ls"})
        if result["safe"]:
            agent.invoke(...)

        # Option B: Use as a pre-execution hook
        original_run = tool.run
        def guarded_run(input):
            check = guard.check(tool_name=tool.name, tool_params={"input": input})
            if not check["safe"]:
                return f"BLOCKED: {check.get('risk_type', 'unknown risk')}"
            return original_run(input)
        tool.run = guarded_run
    """

    def __init__(self, config: CerebellumConfig | None = None):
        cfg = config or CerebellumConfig.from_yaml()
        self.cb = DigitalCerebellum(cfg)
        self.cb.register_microzone(ToolCallMicrozone())

    def check(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        context: str = "",
    ) -> dict[str, Any]:
        """
        Pre-evaluate a tool call.

        Returns dict with at least:
            - safe: bool
            - safety_score: float (0=dangerous, 1=safe)
            - confidence: float (0=uncertain, 1=certain)
        """
        return self.cb.evaluate("tool_call", {
            "tool_name": tool_name,
            "tool_params": tool_params,
        }, context=context)

    def guard_tool(self, tool: Any) -> Any:
        """
        Monkey-patch a LangChain tool with cerebellar pre-evaluation.

        The tool's `run` method is wrapped so that dangerous invocations
        are blocked before execution.
        """
        original_run = tool.run
        guard = self

        def guarded_run(input_str: str, **kwargs):
            result = guard.check(
                tool_name=getattr(tool, "name", "unknown"),
                tool_params={"input": input_str},
            )
            if not result["safe"]:
                return (
                    f"[Cerebellum BLOCKED] Tool '{tool.name}' was blocked: "
                    f"risk_type={result.get('risk_type', 'unknown')}, "
                    f"safety_score={result.get('safety_score', 0):.3f}"
                )
            return original_run(input_str, **kwargs)

        tool.run = guarded_run
        return tool


# -- Demo without actual LangChain dependency ----------------------------

if __name__ == "__main__":
    guard = CerebellumGuard()

    scenarios = [
        ("send_email", {"to": "colleague@work.com", "body": "Meeting at 3pm"}),
        ("run_command", {"command": "rm -rf /"}),
        ("search_web", {"query": "weather today"}),
        ("execute_sql", {"query": "DROP TABLE users"}),
    ]

    print("LangChain Cerebellum Guard Demo")
    print("=" * 50)

    for tool, params in scenarios:
        result = guard.check(tool, params)
        status = "ALLOW" if result["safe"] else "BLOCK"
        print(f"  [{status}] {tool}({params})")
        print(f"         safety={result.get('safety_score', 0):.3f} "
              f"confidence={result.get('confidence', 0):.3f}")
