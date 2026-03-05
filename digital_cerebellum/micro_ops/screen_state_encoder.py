"""
ScreenStateEncoder — converts screen imagery to numeric state vectors.

Biological basis:
  The visual cortex (V1→V4) processes retinal images into feature maps.
  The cerebellum receives these as mossy-fibre inputs — not raw pixels,
  but condensed, task-relevant state descriptions.

  This encoder provides three strategies, ordered by fidelity/speed tradeoff:

  - **roi** (default): extract positions of known UI elements → compact vector.
    Fastest (<0.5ms), best for structured GUIs where element positions matter.

  - **downsample**: resize image to small grid → flatten to vector.
    Fast (~1ms), captures spatial layout without semantic understanding.

  - **hybrid**: ROI for known elements + downsampled context.
    Medium (~2ms), combines structured and visual information.

  All strategies output a fixed-length float32 vector compatible with
  MicroOpEngine's StateEncoder → PatternSeparator pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ROISpec:
    """Specification for a region-of-interest in screen space."""
    name: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class ScreenStateConfig:
    """Configuration for the screen state encoder."""
    strategy: str = "roi"
    downsample_size: tuple[int, int] = (16, 16)
    grayscale: bool = True
    roi_specs: list[ROISpec] = field(default_factory=list)
    screen_w: int = 1920
    screen_h: int = 1080
    normalize: bool = True


class ScreenStateEncoder:
    """
    Encodes screen captures into fixed-length numeric state vectors.

    Designed for the MicroOpEngine's Environment protocol: the observe()
    call produces a screen image, this encoder converts it to the numeric
    state vector that the cerebellar pipeline processes at 285Hz.
    """

    def __init__(self, cfg: ScreenStateConfig | None = None):
        self.cfg = cfg or ScreenStateConfig()
        self._count = 0
        self._running_mean: np.ndarray | None = None
        self._running_var: np.ndarray | None = None

    @property
    def state_dim(self) -> int:
        """Output vector dimension, depends on strategy."""
        if self.cfg.strategy == "roi":
            return len(self.cfg.roi_specs) * 4 + 2
        elif self.cfg.strategy == "downsample":
            h, w = self.cfg.downsample_size
            channels = 1 if self.cfg.grayscale else 3
            return h * w * channels
        elif self.cfg.strategy == "hybrid":
            roi_dim = len(self.cfg.roi_specs) * 4 + 2
            h, w = self.cfg.downsample_size
            return roi_dim + h * w
        raise ValueError(f"Unknown strategy: {self.cfg.strategy}")

    def encode_rois(self, rois: list[dict[str, Any]]) -> np.ndarray:
        """
        Encode ROI positions into a state vector.

        Each ROI contributes [norm_cx, norm_cy, norm_w, norm_h].
        Appends [cursor_x, cursor_y] if provided.

        Parameters
        ----------
        rois : list of dicts with keys: name, x, y, w, h, and optional cursor_x/y

        Returns
        -------
        (state_dim,) float32 vector
        """
        parts = []
        sw, sh = self.cfg.screen_w, self.cfg.screen_h

        for roi in rois:
            cx = (roi.get("x", 0) + roi.get("w", 0) / 2) / sw
            cy = (roi.get("y", 0) + roi.get("h", 0) / 2) / sh
            nw = roi.get("w", 0) / sw
            nh = roi.get("h", 0) / sh
            parts.extend([cx, cy, nw, nh])

        while len(parts) < len(self.cfg.roi_specs) * 4:
            parts.extend([0.0, 0.0, 0.0, 0.0])

        cursor_x = rois[0].get("cursor_x", 0.5) if rois else 0.5
        cursor_y = rois[0].get("cursor_y", 0.5) if rois else 0.5
        parts.extend([cursor_x, cursor_y])

        vec = np.array(parts[:self.state_dim], dtype=np.float32)
        return self._maybe_normalize(vec)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode a screen image by downsampling to a flat vector.

        Parameters
        ----------
        image : (H, W) or (H, W, C) uint8/float array

        Returns
        -------
        (state_dim,) float32 vector
        """
        if image.ndim == 3 and self.cfg.grayscale:
            image = np.mean(image, axis=2)

        h, w = self.cfg.downsample_size
        img_h, img_w = image.shape[:2]

        step_h = max(1, img_h // h)
        step_w = max(1, img_w // w)
        downsampled = image[::step_h, ::step_w][:h, :w]

        if downsampled.shape[0] < h or downsampled.shape[1] < w:
            padded = np.zeros((h, w), dtype=np.float32)
            padded[:downsampled.shape[0], :downsampled.shape[1]] = downsampled
            downsampled = padded

        vec = downsampled.flatten().astype(np.float32)
        if vec.max() > 1.0:
            vec = vec / 255.0

        return self._maybe_normalize(vec)

    def encode_hybrid(
        self,
        rois: list[dict[str, Any]],
        image: np.ndarray,
    ) -> np.ndarray:
        """Combine ROI positions with downsampled context image."""
        roi_vec = self.encode_rois(rois)
        img_vec = self.encode_image(image)
        combined = np.concatenate([roi_vec, img_vec])
        return combined[:self.state_dim]

    def encode(
        self,
        rois: list[dict[str, Any]] | None = None,
        image: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Auto-dispatch to the appropriate encoding strategy.

        For roi strategy: pass rois.
        For downsample strategy: pass image.
        For hybrid strategy: pass both.
        """
        if self.cfg.strategy == "roi":
            if rois is None:
                raise ValueError("ROI strategy requires rois argument")
            return self.encode_rois(rois)
        elif self.cfg.strategy == "downsample":
            if image is None:
                raise ValueError("Downsample strategy requires image argument")
            return self.encode_image(image)
        elif self.cfg.strategy == "hybrid":
            if rois is None or image is None:
                raise ValueError("Hybrid strategy requires both rois and image")
            return self.encode_hybrid(rois, image)
        raise ValueError(f"Unknown strategy: {self.cfg.strategy}")

    def _maybe_normalize(self, vec: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return vec
        self._count += 1
        if self._running_mean is None:
            self._running_mean = np.zeros_like(vec)
            self._running_var = np.ones_like(vec)
        if self._count < 5:
            return vec
        delta = vec - self._running_mean
        self._running_mean += delta / self._count
        delta2 = vec - self._running_mean
        self._running_var += (delta * delta2 - self._running_var) / self._count
        std = np.sqrt(self._running_var + 1e-8)
        return (vec - self._running_mean) / std
