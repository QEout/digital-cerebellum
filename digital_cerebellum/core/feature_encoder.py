"""
Feature Encoder — mossy fibre analogue.

Converts raw events (especially ToolCallEvents) into fixed-length
numerical feature vectors suitable for the pattern separator.

Phase 0: uses sentence-transformers for text encoding + cyclic time features.
"""

from __future__ import annotations

import json
import math
import time
from functools import lru_cache

import numpy as np


class FeatureEncoder:
    """
    Encodes CerebellumEvent / ToolCallEvent into a numerical vector.

    Output dim ≈ embedding_dim (384) + time_features (4) + type_onehot.
    """

    KNOWN_TOOLS: list[str] = []   # populated dynamically

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._embedding_model_name = embedding_model
        self._encoder = None       # lazy load
        self._embedding_dim: int | None = None

    # ------------------------------------------------------------------
    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._embedding_model_name)
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()
        return self._encoder

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        if self._embedding_dim is None:
            self._get_encoder()
        return self._embedding_dim + 4  # text_emb + 4 time features

    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        encoder = self._get_encoder()
        return encoder.encode(text, normalize_embeddings=True)

    # ------------------------------------------------------------------
    def encode_tool_call_raw(
        self,
        text: str,
        timestamp: float | None = None,
    ) -> np.ndarray:
        """
        Generic encoder: text → [text_embedding | time_features].

        Used by the general-purpose pipeline.  Microzones provide the text
        via their ``format_input`` method.
        """
        text_emb = self.encode_text(text)
        ts = timestamp or time.time()
        time_feats = self._time_features(ts)
        return np.concatenate([text_emb, time_feats]).astype(np.float32)

    # ------------------------------------------------------------------
    def encode_tool_call(
        self,
        tool_name: str,
        tool_params: dict,
        context: str = "",
        timestamp: float | None = None,
    ) -> np.ndarray:
        """
        Backward-compatible: encode a tool call into a feature vector.
        """
        text = f"{tool_name}({json.dumps(tool_params, ensure_ascii=False)})"
        if context:
            text = f"{context} -> {text}"
        return self.encode_tool_call_raw(text, timestamp)

    # ------------------------------------------------------------------
    @staticmethod
    def _time_features(ts: float) -> np.ndarray:
        """Cyclic encoding of hour-of-day and day-of-week."""
        import datetime
        dt = datetime.datetime.fromtimestamp(ts)
        hour = dt.hour + dt.minute / 60.0
        weekday = dt.weekday()
        return np.array([
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * weekday / 7),
            math.cos(2 * math.pi * weekday / 7),
        ], dtype=np.float32)
