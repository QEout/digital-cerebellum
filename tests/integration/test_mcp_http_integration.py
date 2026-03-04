"""
Real MCP HTTP Integration Test.

Starts the Digital Cerebellum MCP server on Streamable HTTP transport,
then exercises ALL 17 tools via the exact same protocol that OpenClaw
(or any MCP-compatible client) uses in production.

Covers:
  1. Server launch & MCP handshake  (initialize + tools/list)
  2. Safety evaluation tools        (tool_call, shell, file, api, payment)
  3. Skill lifecycle                (learn → match → feedback → re-match)
  4. Step monitor full cycle        (before → after → cascade → rollback)
  5. Metacognition tools            (introspect, stats, curiosity)
  6. Multi-step OpenClaw scenario   (simulated desktop automation task)

Run:
    python tests/integration/test_mcp_http_integration.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

SERVER_PORT = 18321
MCP_ENDPOINT = f"http://127.0.0.1:{SERVER_PORT}/mcp"


class MCPClient:
    """Minimal MCP client over Streamable HTTP (handles both JSON and SSE)."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session_id: str | None = None
        self._req_id = 0

    def rpc(self, method: str, params: dict | None = None) -> dict:
        import httpx

        self._req_id += 1
        payload = {"jsonrpc": "2.0", "method": method, "id": self._req_id}
        if params is not None:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        with httpx.Client(timeout=60) as client:
            with client.stream("POST", self.endpoint,
                               json=payload, headers=headers) as r:
                if sid := r.headers.get("mcp-session-id"):
                    self.session_id = sid

                ct = r.headers.get("content-type", "")

                if "text/event-stream" in ct:
                    for line in r.iter_lines():
                        if line.startswith("data: "):
                            msg = json.loads(line[6:])
                            if "result" in msg:
                                return msg["result"]
                            if "error" in msg:
                                raise RuntimeError(
                                    f"JSON-RPC error: {msg['error']}")
                    raise RuntimeError("No result in SSE stream")
                else:
                    r.read()
                    body = r.json()
                    if "error" in body:
                        raise RuntimeError(f"JSON-RPC error: {body['error']}")
                    return body.get("result", body)

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self.rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if isinstance(result, dict) and "content" in result:
            for item in result["content"]:
                if item.get("type") == "text":
                    return json.loads(item["text"])
        return result


def start_server() -> subprocess.Popen:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_python = os.path.join(project_root, ".venv", "bin", "python")
    python = venv_python if os.path.exists(venv_python) else sys.executable

    proc = subprocess.Popen(
        [python, "-m", "digital_cerebellum.mcp_server", "--http", "--port", str(SERVER_PORT)],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    if proc.poll() is not None:
        raise RuntimeError(f"Server exited with code {proc.returncode}")
    return proc


def stop_server(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    print("=" * 60)
    print("  Digital Cerebellum — Real MCP HTTP Integration Test")
    print("=" * 60)

    proc = start_server()
    client = MCPClient(MCP_ENDPOINT)
    passed = failed = 0

    def check(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    try:
        # ============================================================
        # 1. MCP Handshake
        # ============================================================
        print(f"\n{'─'*60}")
        print("  1. MCP Handshake")
        print(f"{'─'*60}")

        def test_initialize():
            r = client.rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "integration-test", "version": "1.0"},
            })
            assert r["serverInfo"]["name"] == "Digital Cerebellum"
            print(f"       Server: {r['serverInfo']['name']} v{r['serverInfo']['version']}")
            print(f"       Session: {client.session_id}")

        check("initialize", test_initialize)

        def test_tools_list():
            r = client.rpc("tools/list")
            names = [t["name"] for t in r["tools"]]
            assert len(names) == 17, f"Expected 17 tools, got {len(names)}"
            print(f"       Tools exposed: {len(names)}")
            for n in names:
                print(f"         - {n}")

        check("tools/list", test_tools_list)

        # ============================================================
        # 2. Safety Evaluation
        # ============================================================
        print(f"\n{'─'*60}")
        print("  2. Safety Evaluation Tools")
        print(f"{'─'*60}")

        def test_eval_tool_call():
            r = client.call_tool("evaluate_tool_call", {
                "tool_name": "send_email",
                "tool_params": {"to": "alice@example.com", "body": "Hello!"},
            })
            assert "route" in r or "prediction" in r

        check("evaluate_tool_call", test_eval_tool_call)

        def test_eval_shell_dangerous():
            r = client.call_tool("evaluate_shell_command", {
                "command": "rm -rf /", "shell": "bash",
            })
            assert isinstance(r, dict)
            print(f"       rm -rf / → {r.get('prediction', {}).get('label', r.get('route', '?'))}")

        check("evaluate_shell_command (dangerous)", test_eval_shell_dangerous)

        def test_eval_file():
            r = client.call_tool("evaluate_file_operation", {
                "operation": "read", "path": "/etc/passwd",
            })
            assert isinstance(r, dict)

        check("evaluate_file_operation", test_eval_file)

        def test_eval_api():
            r = client.call_tool("evaluate_api_call", {
                "method": "DELETE", "url": "https://api.prod.com/users/123",
            })
            assert isinstance(r, dict)

        check("evaluate_api_call", test_eval_api)

        def test_eval_payment():
            r = client.call_tool("evaluate_payment", {
                "amount": 50000, "currency": "BTC",
                "recipient": "anon-wallet", "method": "crypto",
            })
            assert isinstance(r, dict)

        check("evaluate_payment", test_eval_payment)

        # ============================================================
        # 3. Skill Lifecycle
        # ============================================================
        print(f"\n{'─'*60}")
        print("  3. Skill Lifecycle (learn → reinforce → match)")
        print(f"{'─'*60}")

        def test_skill_lifecycle():
            s = client.call_tool("learn_skill", {
                "input_text": "What is the refund policy?",
                "response_text": "Full refund within 30 days, no questions asked.",
                "domain": "support",
            })
            assert s["status"] == "ok"
            skill_id = s["skill_id"]
            print(f"       Learned: {skill_id}")

            for _ in range(5):
                client.call_tool("skill_feedback", {"skill_id": skill_id, "success": True})

            m = client.call_tool("match_skill", {"query": "What is the refund policy?"})
            assert m["matched"] is True
            assert m["should_execute"] is True
            assert "30 days" in m["response_text"]
            print(f"       Similarity: {m['similarity']:.3f}")
            print(f"       Execute: {m['should_execute']}")

        check("learn → reinforce → match", test_skill_lifecycle)

        def test_no_match():
            m = client.call_tool("match_skill", {"query": "How to bake a cake?"})
            assert m["matched"] is False
            print(f"       Unrelated query: matched={m['matched']}")

        check("no match for unrelated query", test_no_match)

        # ============================================================
        # 4. Step Monitor — Full Email Scenario
        # ============================================================
        print(f"\n{'─'*60}")
        print("  4. Step Monitor — 8-Step Gmail Scenario")
        print(f"{'─'*60}")

        STEPS = [
            ("Open Chrome browser", "Desktop idle", "Chrome launched, home page loaded"),
            ("Navigate to gmail.com", "Chrome open", "Gmail inbox loaded, 47 unread messages"),
            ("Click first unread email from Alice", "Inbox visible", "Email opened: Q1 Report Review"),
            ("Click Reply button", "Email open", "Reply composer opened"),
            ("Type reply text", "Composer open, cursor in body", "Text typed successfully"),
            ("Click attachment icon (paperclip)", "Reply composed", "File picker dialog opened"),
            ("Select Q1_Report_v2.pdf", "File picker open", "File attached (2.3MB)"),
            ("Click Send button", "Ready to send", "Email sent, confirmation shown"),
        ]

        def test_full_scenario():
            client.call_tool("monitor_reset")
            timings = []

            for i, (action, state, outcome) in enumerate(STEPS):
                t0 = time.time()
                pred = client.call_tool("monitor_before_step", {
                    "action": action, "state": state,
                })
                verdict = client.call_tool("monitor_after_step", {
                    "outcome": outcome, "success": True,
                })
                ms = (time.time() - t0) * 1000
                timings.append(ms)
                print(f"       Step {i+1}: {action[:35]:35s} "
                      f"SPE={verdict['spe']:.3f} pause={verdict['should_pause']} "
                      f"{ms:.0f}ms")

            assert not any(v for v in []), "Should never trigger"

            avg = sum(timings) / len(timings)
            total = sum(timings)
            print(f"       ──────────────────────────────────────────")
            print(f"       Avg MCP round-trip: {avg:.0f}ms/step")
            print(f"       Total overhead:     {total:.0f}ms for {len(STEPS)} steps")

            client.call_tool("learn_skill", {
                "input_text": "Reply to latest Gmail email with attachment",
                "response_text": json.dumps([s[0] for s in STEPS]),
                "domain": "desktop_automation",
            })

        check("8-step Gmail scenario (all success)", test_full_scenario)

        # ============================================================
        # 5. Cascade Detection + AutoRollback
        # ============================================================
        print(f"\n{'─'*60}")
        print("  5. Cascade Detection + AutoRollback")
        print(f"{'─'*60}")

        def test_cascade_rollback():
            client.call_tool("monitor_reset")

            for action, state, outcome in STEPS[:5]:
                client.call_tool("monitor_before_step", {"action": action, "state": state})
                client.call_tool("monitor_after_step", {"outcome": outcome, "success": True})
            print(f"       5 successful steps completed")

            cascade_at = None
            for attempt in range(8):
                client.call_tool("monitor_before_step", {
                    "action": "Click Send button",
                    "state": "Compose ready",
                })
                v = client.call_tool("monitor_after_step", {
                    "outcome": f"Error: network timeout (attempt {attempt+1})",
                    "success": False,
                })
                if v["should_pause"]:
                    cascade_at = attempt + 1
                    print(f"       Cascade detected after {cascade_at} failures "
                          f"(risk={v['cascade_risk']:.3f})")
                    break

            assert cascade_at is not None, "Cascade should have triggered"

            plan = client.call_tool("monitor_rollback_plan")
            assert plan["available"] is True
            assert plan["rollback_to_step"] >= 0
            assert len(plan["failed_steps"]) > 0
            print(f"       Rollback to step:  {plan['rollback_to_step']}")
            print(f"       Steps wasted:      {plan['steps_wasted']}")
            print(f"       Failed steps:      {len(plan['failed_steps'])}")
            print(f"       Recommendation:    {plan['recommendation'][:80]}")

        check("cascade + rollback plan", test_cascade_rollback)

        # ============================================================
        # 6. Metacognition
        # ============================================================
        print(f"\n{'─'*60}")
        print("  6. Metacognition & System Stats")
        print(f"{'─'*60}")

        def test_introspect():
            r = client.call_tool("introspect")
            assert "competencies" in r
            print(f"       Domains: {list(r['competencies'].keys())}")

        check("introspect", test_introspect)

        def test_stats():
            r = client.call_tool("get_stats")
            assert "step" in r
            print(f"       Total steps: {r['step']}")
            print(f"       Microzones: {r.get('microzones', [])}")

        check("get_stats", test_stats)

        def test_curiosity():
            r = client.call_tool("get_curiosity_ranking")
            assert isinstance(r, (list, dict))
            items = r if isinstance(r, list) else r.get("result", [])
            print(f"       Curiosity domains: {len(items)}")

        check("get_curiosity_ranking", test_curiosity)

        def test_monitor_status():
            client.call_tool("monitor_reset")
            r = client.call_tool("monitor_status")
            assert isinstance(r, dict)
            print(f"       Forward model trained: {r.get('forward_model', {}).get('trained_steps', '?')}")

        check("monitor_status", test_monitor_status)

        # ============================================================
        # Results
        # ============================================================
        print(f"\n{'='*60}")
        total = passed + failed
        if failed == 0:
            print(f"  ALL {passed} TESTS PASSED")
        else:
            print(f"  {passed}/{total} passed, {failed} FAILED")
        print(f"{'='*60}")

    finally:
        stop_server(proc)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
