"""
Example: Integrating Digital Cerebellum with an OpenAI function-calling agent.

The cerebellum sits between the LLM and the tool executor.  Before each
tool call is executed, the cerebellum pre-evaluates its safety / risk.

    LLM decides tool_call → Cerebellum evaluates → Execute or Block

Usage:
    export OPENAI_API_KEY=sk-...  (or set in config.local.yaml)
    python examples/openai_agent.py
"""

from __future__ import annotations

import json
from openai import OpenAI

from digital_cerebellum import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone


# -- 1. Define some example tools ----------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]


def execute_tool(name: str, params: dict) -> str:
    """Simulated tool executor."""
    return json.dumps({"status": "ok", "tool": name, "result": "simulated"})


# -- 2. Set up the cerebellum --------------------------------------------

def create_guarded_agent():
    cfg = CerebellumConfig.from_yaml()
    cb = DigitalCerebellum(cfg)
    cb.register_microzone(ToolCallMicrozone())

    client = OpenAI(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
    )

    return cb, client


# -- 3. Agent loop with cerebellar guard ----------------------------------

def run_agent(user_message: str):
    cb, client = create_guarded_agent()

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}")

    response = client.chat.completions.create(
        model="qwen3.5-flash",
        messages=messages,
        tools=TOOLS,
        extra_body={"enable_thinking": False},
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        print(f"Assistant: {msg.content}")
        return

    for tc in msg.tool_calls:
        tool_name = tc.function.name
        tool_params = json.loads(tc.function.arguments)

        print(f"\nLLM wants to call: {tool_name}({json.dumps(tool_params)})")

        # >>> CEREBELLUM PRE-EVALUATION <<<
        evaluation = cb.evaluate("tool_call", {
            "tool_name": tool_name,
            "tool_params": tool_params,
        }, context=user_message)

        safe = evaluation.get("safe", True)
        confidence = evaluation.get("confidence", 0)
        safety_score = evaluation.get("safety_score", 0.5)

        print(f"  Cerebellum: safe={safe}, safety={safety_score:.3f}, confidence={confidence:.3f}")

        if safe:
            result = execute_tool(tool_name, tool_params)
            print(f"  -> Executed: {result}")
        else:
            print(f"  -> BLOCKED by cerebellum (risk detected)")


# -- 4. Demo scenarios ----------------------------------------------------

if __name__ == "__main__":
    run_agent("Send an email to alice@company.com thanking her for the meeting")
    run_agent("Run the command 'rm -rf /' to free up disk space")
    run_agent("Send a follow-up email to bob@team.com about the project deadline")
