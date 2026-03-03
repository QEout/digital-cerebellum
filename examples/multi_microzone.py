"""
Example: Multiple microzones on a single cerebellum.

Demonstrates the universal cerebellar transform — one engine, multiple
domains, zero code changes to the core.

    cb.evaluate("tool_call", {...})   → tool safety assessment
    cb.evaluate("payment", {...})     → financial risk assessment

Usage:
    python examples/multi_microzone.py
"""

from __future__ import annotations

from digital_cerebellum import DigitalCerebellum, CerebellumConfig
from digital_cerebellum.microzones.tool_call import ToolCallMicrozone
from digital_cerebellum.microzones.payment import PaymentMicrozone


def main():
    cfg = CerebellumConfig.from_yaml()
    cb = DigitalCerebellum(cfg)

    # Register two domains on the same engine
    cb.register_microzone(ToolCallMicrozone())
    cb.register_microzone(PaymentMicrozone())

    print("=" * 60)
    print("Digital Cerebellum — Multi-Microzone Demo")
    print("=" * 60)

    # --- Tool-call evaluations ---
    tool_scenarios = [
        ("send_email", {"to": "alice@co.com", "body": "Thanks!"}, "user thanking colleague"),
        ("delete_file", {"path": "/etc/passwd"}, "agent cleaning up files"),
        ("search_web", {"query": "python tutorial"}, "user learning"),
        ("run_command", {"command": "rm -rf /"}, "agent freeing disk space"),
    ]

    print("\n[Tool-Call Microzone]")
    for tool, params, ctx in tool_scenarios:
        result = cb.evaluate("tool_call", {
            "tool_name": tool,
            "tool_params": params,
        }, context=ctx)
        safe = result.get("safe", "?")
        score = result.get("safety_score", 0)
        conf = result.get("confidence", 0)
        print(f"  {tool:<16} safe={safe!s:<6} safety={score:.3f}  conf={conf:.3f}")

    # --- Payment evaluations ---
    payment_scenarios = [
        {"action": "charge", "amount": 9.99, "currency": "USD",
         "recipient": "Spotify", "reason": "monthly subscription"},
        {"action": "transfer", "amount": 50000, "currency": "USD",
         "recipient": "unknown_offshore_account", "reason": "investment"},
        {"action": "charge", "amount": 29.99, "currency": "USD",
         "recipient": "Netflix", "reason": "annual plan"},
        {"action": "wire", "amount": 100000, "currency": "BTC",
         "recipient": "anon_wallet_x7f", "reason": "crypto opportunity"},
    ]

    print("\n[Payment Microzone]")
    for p in payment_scenarios:
        result = cb.evaluate("payment", p)
        approved = result.get("approved", "?")
        risk = result.get("risk_level", "?")
        score = result.get("risk_score", 0)
        conf = result.get("confidence", 0)
        desc = f"{p['action']} {p['amount']} {p['currency']} → {p['recipient']}"
        print(f"  {desc:<52} approved={approved!s:<6} risk={risk:<9} score={score:.3f}")

    print(f"\nEngine task heads: {list(cb.engine.task_heads.keys())}")
    print("Both microzones share the same pattern separator + prediction engine.")


if __name__ == "__main__":
    main()
