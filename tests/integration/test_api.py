"""
Integration tests — run against a live API.
Usage (after `docker compose up`):
    pytest tests/integration/
"""
import os

import pytest
import requests

API = os.getenv("API_URL", "http://localhost:8000")


def _api_up():
    try:
        return requests.get(f"{API}/health", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="API not running")


def test_health():
    r = requests.get(f"{API}/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "container_id" in body


def test_ready():
    r = requests.get(f"{API}/ready", timeout=5)
    assert r.status_code in (200, 503)


def test_metrics_exposed():
    r = requests.get(f"{API}/metrics", timeout=5)
    assert r.status_code == 200
    assert b"http_requests_total" in r.content


def test_predict_returns_valid_probs():
    r = requests.post(f"{API}/predict", json={"text": "I feel anxious"}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_class" in body
    assert 0.0 <= body["confidence"] <= 1.0
    total = sum(body["probabilities"].values())
    assert 0.99 <= total <= 1.01


def test_predict_empty_text_rejected():
    r = requests.post(f"{API}/predict", json={"text": ""}, timeout=5)
    assert r.status_code == 422


def test_predict_missing_field_rejected():
    r = requests.post(f"{API}/predict", json={}, timeout=5)
    assert r.status_code == 422


def test_feedback_accepted():
    r = requests.post(f"{API}/feedback", json={
        "text": "sample",
        "predicted_label": "Depression",
        "correct_label": "Anxiety",
        "was_correct": False,
    }, timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "logged"


def test_load_distribution_across_containers():
    """Make 30 requests; at least 2 distinct container IDs should appear."""
    ids = set()
    for _ in range(30):
        r = requests.post(f"{API}/predict", json={"text": "I am sad"}, timeout=10)
        if r.status_code == 200:
            ids.add(r.json()["container_id"])
    assert len(ids) >= 2, f"Expected load-balancing across replicas, got: {ids}"
