"""
Cortex Interface — cortico-cerebellar loop analogue.

Handles:
1. Slow-path LLM calls  (with cerebellar hint injection)
2. Response parsing      (extract action + outcome embeddings)
3. Consolidation buffer  (record interaction pairs for later distillation)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class CortexInterface:
    """
    Talks to an LLM via OpenAI-compatible API.

    Phase 0: synchronous calls, no streaming.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # ------------------------------------------------------------------
    def evaluate_tool_call(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        conversation_context: str = "",
        cerebellar_hint: dict | None = None,
    ) -> dict[str, Any]:
        """
        Ask the LLM to evaluate a tool call's safety and expected outcome.

        Returns a structured dict with keys:
            safe (bool), risk_type (str), reasoning (str),
            expected_outcome (str)
        """
        system_prompt = (
            "You are a tool-call safety evaluator. Given a tool invocation, "
            "assess whether it is safe, identify any risks, and predict the outcome.\n"
            "Respond ONLY with a JSON object: "
            '{"safe": bool, "risk_type": "none"|"wrong_param"|"wrong_tool"|"hallucinated_data"|"dangerous_action", '
            '"reasoning": "...", "expected_outcome": "..."}'
        )

        user_msg_parts = [
            f"Tool: {tool_name}",
            f"Params: {json.dumps(tool_params, ensure_ascii=False)}",
        ]
        if conversation_context:
            user_msg_parts.append(f"Context: {conversation_context}")
        if cerebellar_hint:
            user_msg_parts.append(
                f"Cerebellar hint (for reference only, may be inaccurate): "
                f"{json.dumps(cerebellar_hint, ensure_ascii=False)}"
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n".join(user_msg_parts)},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            text = resp.choices[0].message.content or "{}"
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception as e:
            log.warning("LLM evaluation failed: %s", e)
            return {
                "safe": True,
                "risk_type": "none",
                "reasoning": f"LLM call failed ({e}), defaulting to safe",
                "expected_outcome": "unknown",
            }

    # ------------------------------------------------------------------
    def get_action_embedding(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        encoder,
    ) -> np.ndarray:
        """Encode a tool call as an action embedding using the sentence encoder."""
        text = f"{tool_name}({json.dumps(tool_params, ensure_ascii=False)})"
        return encoder.encode(text)

    # ------------------------------------------------------------------
    def get_outcome_embedding(
        self,
        outcome_text: str,
        encoder,
    ) -> np.ndarray:
        """Encode an outcome description as an embedding."""
        return encoder.encode(outcome_text)
