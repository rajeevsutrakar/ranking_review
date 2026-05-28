"""
Persistent running-average cache for CLIP similarity scores.

Structure on disk (JSON):
{
    "<product_id>": {
        "<image_url>": {"avg": 0.85, "count": 3},
        ...
    },
    ...
}

Running average update formula (no need to store all past scores):
    new_avg   = (old_avg * old_count + new_score) / (old_count + 1)
    new_count = old_count + 1
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class CLIPScoreCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        # {product_id_str: {image_url: {"avg": float, "count": int}}}
        self._data: Dict[str, Dict[str, Dict[str, float | int]]] = {}
        if path.exists():
            self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2),
            encoding="utf-8",
        )

    # ── read ───────────────────────────────────────────────────────────────

    def get_avg(self, product_id: int, image_url: str) -> float:
        """Current running average for this (product, image) pair, or 0.0."""
        entry = self._data.get(str(product_id), {}).get(image_url)
        return float(entry["avg"]) if entry else 0.0

    def get_count(self, product_id: int, image_url: str) -> int:
        entry = self._data.get(str(product_id), {}).get(image_url)
        return int(entry["count"]) if entry else 0

    def get_product_scores(self, product_id: int) -> Dict[str, float]:
        """Return {image_url: avg_score} for all images scored for this product."""
        return {
            url: float(entry["avg"])
            for url, entry in self._data.get(str(product_id), {}).items()
        }

    # ── write ──────────────────────────────────────────────────────────────

    def update(self, product_id: int, image_url: str, new_score: float) -> float:
        """
        Update running average for (product_id, image_url) with new_score.
        Returns the updated average.
        """
        pid_key = str(product_id)
        if pid_key not in self._data:
            self._data[pid_key] = {}

        entry = self._data[pid_key].get(image_url)
        if entry is None:
            self._data[pid_key][image_url] = {"avg": new_score, "count": 1}
            return new_score

        old_avg = float(entry["avg"])
        old_count = int(entry["count"])
        new_avg = (old_avg * old_count + new_score) / (old_count + 1)
        self._data[pid_key][image_url] = {"avg": new_avg, "count": old_count + 1}
        return new_avg

    def update_batch(
        self,
        product_id: int,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Update running averages for all (image_url → score) pairs in one call.
        Returns {image_url: updated_avg}.
        """
        return {url: self.update(product_id, url, score) for url, score in scores.items()}
