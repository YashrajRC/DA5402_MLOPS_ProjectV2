import json
import logging
import math
from collections import Counter, deque
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger("api.drift")


class DriftDetector:
    def __init__(self, baseline_path: str, window_size: int = 200):
        self.baseline_path = Path(baseline_path)
        self.window_size = window_size
        self.recent_texts = deque(maxlen=window_size)
        self.lock = Lock()
        self.baseline: Optional[dict] = None
        self._load_baseline()

    def _load_baseline(self):
        if self.baseline_path.exists():
            with open(self.baseline_path) as f:
                self.baseline = json.load(f)
            log.info(f"Drift baseline loaded from {self.baseline_path}")
        else:
            log.warning(f"Drift baseline not found: {self.baseline_path}")

    def add_text(self, text: str):
        with self.lock:
            self.recent_texts.append(text)

    def _js_divergence(self, p: dict, q: dict) -> float:
        """Jensen-Shannon divergence between two word-freq distributions."""
        vocab = set(p) | set(q)
        eps = 1e-12
        js = 0.0
        for w in vocab:
            pw = p.get(w, 0.0) + eps
            qw = q.get(w, 0.0) + eps
            m = 0.5 * (pw + qw)
            js += 0.5 * pw * math.log(pw / m) + 0.5 * qw * math.log(qw / m)
        return min(max(js, 0.0), 1.0)

    def compute(self) -> float:
        with self.lock:
            n = len(self.recent_texts)
            if not self.baseline or n < 20:
                return 0.0
            all_words = []
            for t in self.recent_texts:
                all_words.extend(t.split())

        if not all_words:
            return 0.0

        counts = Counter(all_words)
        top_words = dict(counts.most_common(1000))
        top_total = sum(top_words.values()) or 1
        current = {w: c / top_total for w, c in top_words.items()}
        baseline_freq = self.baseline.get("word_freq_top1000", {})
        return float(self._js_divergence(current, baseline_freq))
