"""
Digital Brain Demo — complete cognitive architecture without any agent framework.

The LLM IS the cortex (brain). The cerebellum handles fast-path predictions.
Together they form a complete autonomous agent.

Usage:
    python examples/brain_demo.py
"""

from __future__ import annotations

import json
from digital_cerebellum.brain import DigitalBrain


# ── Simulated tools ──────────────────────────────────────────────

def search_web(query: str) -> str:
    results = {
        "weather tokyo": "Tokyo: 15°C, partly cloudy, humidity 65%",
        "python tutorial": "Top result: docs.python.org/3/tutorial/",
        "latest news": "AI research advances in cerebellar computing",
    }
    for key, val in results.items():
        if key in query.lower():
            return val
    return f"Search results for '{query}': No specific results found."


def send_email(to: str, subject: str = "", body: str = "") -> str:
    return json.dumps({"status": "sent", "to": to, "subject": subject})


def read_file(path: str) -> str:
    safe_files = {"/tmp/notes.txt": "Meeting at 3pm tomorrow", "/tmp/todo.txt": "1. Buy groceries\n2. Call dentist"}
    if path in safe_files:
        return safe_files[path]
    return f"File not found: {path}"


def run_command(command: str) -> str:
    return f"Command executed: {command} → (simulated output)"


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ── Main demo ────────────────────────────────────────────────────

def main():
    brain = DigitalBrain.from_yaml()

    brain.register_tool("search_web", search_web,
        description="Search the internet for information",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
        }, "required": ["query"]})

    brain.register_tool("send_email", send_email,
        description="Send an email to a recipient",
        parameters={"type": "object", "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        }, "required": ["to"]})

    brain.register_tool("read_file", read_file,
        description="Read a file from the filesystem",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to read"},
        }, "required": ["path"]})

    brain.register_tool("run_command", run_command,
        description="Execute a shell command",
        parameters={"type": "object", "properties": {
            "command": {"type": "string"},
        }, "required": ["command"]})

    brain.register_tool("calculate", calculate,
        description="Evaluate a mathematical expression",
        parameters={"type": "object", "properties": {
            "expression": {"type": "string"},
        }, "required": ["expression"]})

    brain.system_prompt = (
        "You are a helpful AI assistant with access to tools. "
        "Use tools when they help answer the user's question. "
        "Be concise. Always respond in the language the user uses."
    )

    queries = [
        "What's the weather like in Tokyo?",
        "Send an email to alice@company.com thanking her for the meeting",
        "What's 42 * 17 + 3?",
        "Read the file /tmp/notes.txt",
        "Run the command 'rm -rf /'",
        "Search the web for latest news",
        "Send a follow-up email to bob@team.com about the project deadline",
    ]

    print("=" * 70)
    print("  Digital Brain — Cerebellum-First Cognitive Architecture")
    print("  No LangChain. No OpenClaw. Just Brain = LLM + Cerebellum.")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"  [{i}] User: {query}")
        print(f"{'─'*70}")

        result = brain.think(query)

        print(f"  Path: {result.path}  |  LLM called: {result.llm_called}  |  "
              f"Latency: {result.latency_ms:.0f}ms")

        if result.tool_calls:
            for tc in result.tool_calls:
                status = tc.get("status", "?")
                safety = tc.get("cerebellum_eval", {}).get("safety_score", "?")
                print(f"  Tool: {tc['tool']}({json.dumps(tc['params'])[:60]})")
                print(f"    Status: {status}  |  Safety: {safety}")
                if tc.get("result"):
                    print(f"    Result: {tc['result'][:80]}")

        print(f"\n  Brain: {result.text[:200]}")

        brain.reset_conversation()

    print(f"\n{'='*70}")
    print("  Stats")
    print(f"{'='*70}")
    stats = brain.stats
    print(f"  Total queries:    {stats['total_queries']}")
    print(f"  Fast path:        {stats['fast_path']}")
    print(f"  Slow path:        {stats['slow_path']}")
    print(f"  Tools executed:   {stats['tools_executed']}")
    print(f"  Tools blocked:    {stats['tools_blocked']}")
    print(f"  Tools registered: {stats['tools_registered']}")


if __name__ == "__main__":
    main()
