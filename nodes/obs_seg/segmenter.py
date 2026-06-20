#!/usr/bin/env python3
"""
CLIPSeg traversability segmenter (importable, ROS-free)
=======================================================

Wraps the CLIPSeg model so it is loaded **once** and reused. The framework is
general: it takes a list of *traversable* prompts and a list of *untraversable*
prompts and classifies every pixel by the single best-matching prompt
(argmax over all prompts). A pixel whose best score falls below a confidence
threshold is labeled UNKNOWN.

The single-"floor"-prompt case is just the special case where the traversable
list has one entry and the untraversable list is empty (then every non-floor
pixel falls below threshold and is UNKNOWN, which callers treat as blocked).

No ROS dependency, so it can be unit-tested and imported by both the offline
CLI (cli.py) and the ROS costmap node.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from . import FREE, OCCUPIED, UNKNOWN

MODEL_ID = "CIDAS/clipseg-rd64-refined"

# internal per-prompt class tags
_TRAVERSABLE, _UNTRAVERSABLE = 0, 1


class TraversabilitySegmenter:
    """Loads CLIPSeg once; classifies pixels as FREE / OCCUPIED / UNKNOWN."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()

    @staticmethod
    def _as_pil(image) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")

    def prompt_heatmaps(self, image, prompts: Sequence[str]) -> np.ndarray:
        """Return (N, H, W) sigmoid heatmaps, one per prompt, at image resolution."""
        image = self._as_pil(image)
        W, H = image.size
        prompts = list(prompts)
        inputs = self.processor(text=prompts, images=[image] * len(prompts),
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits      # (N, 352, 352) or (352, 352) if N==1
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        return np.stack([cv2.resize(p, (W, H)) for p in probs], axis=0)

    def classify(self, image,
                 traversable_prompts: Sequence[str] = ("the floor",),
                 untraversable_prompts: Sequence[str] = (),
                 threshold: float = 0.5):
        """Per-pixel occupancy labels via argmax over all prompts + confidence threshold.

        Returns (labels, info):
            labels : int8 (H, W) with values FREE / OCCUPIED / UNKNOWN
            info   : dict with 'idx' (winning prompt index per pixel), 'conf'
                     (winning score), 'maps' (N,H,W heatmaps), 'prompts', 'classes',
                     'known' (conf >= threshold) — used by the CLI for visualization.
        """
        trav = list(traversable_prompts)
        obst = list(untraversable_prompts)
        prompts = trav + obst
        if not prompts:
            raise ValueError("classify() needs at least one prompt")
        classes = [_TRAVERSABLE] * len(trav) + [_UNTRAVERSABLE] * len(obst)

        maps = self.prompt_heatmaps(image, prompts)     # (N, H, W)
        idx = maps.argmax(axis=0)                        # winning prompt per pixel
        conf = maps.max(axis=0)                          # winning score per pixel
        known = conf >= threshold
        pixel_class = np.array(classes)[idx]

        labels = np.full(idx.shape, UNKNOWN, dtype=np.int8)
        labels[known & (pixel_class == _TRAVERSABLE)] = FREE
        labels[known & (pixel_class == _UNTRAVERSABLE)] = OCCUPIED

        info = dict(idx=idx, conf=conf, maps=maps,
                    prompts=prompts, classes=classes, known=known)
        return labels, info

    def traversable_mask(self, image,
                         traversable_prompts: Sequence[str] = ("the floor",),
                         untraversable_prompts: Sequence[str] = (),
                         threshold: float = 0.5) -> np.ndarray:
        """Convenience: boolean (H, W), True where the pixel is confidently traversable."""
        labels, _ = self.classify(image, traversable_prompts, untraversable_prompts, threshold)
        return labels == FREE
