"""Unit tests for drift detection."""
import json
import tempfile
from pathlib import Path

from src.api.drift import DriftDetector


def _make_baseline(path, words):
    total = sum(words.values())
    data = {
        "word_freq_top1000": {w: c / total for w, c in words.items()},
        "avg_text_length": 10.0,
        "class_distribution": {},
    }
    Path(path).write_text(json.dumps(data))


def test_drift_zero_when_no_data():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "baseline.json"
        _make_baseline(p, {"a": 10, "b": 5})
        det = DriftDetector(str(p))
        assert det.compute() == 0.0


def test_drift_low_for_similar_distribution():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "baseline.json"
        _make_baseline(p, {"i": 100, "feel": 80, "sad": 70, "anxious": 60})
        det = DriftDetector(str(p))
        for _ in range(30):
            det.add_text("i feel sad and anxious")
        assert det.compute() < 0.3


def test_drift_high_for_ood_inputs():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "baseline.json"
        _make_baseline(p, {"i": 100, "feel": 80, "sad": 70, "anxious": 60})
        det = DriftDetector(str(p))
        for _ in range(30):
            det.add_text("xyzzy plugh quux foobar frobnicate")
        assert det.compute() > 0.2
