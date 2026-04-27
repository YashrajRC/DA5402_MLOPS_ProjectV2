# Test Report
## Mental Health Text Classifier — Full Test Suite Results

**Date:** 2026-04-27  
**Environment:** WSL2 Ubuntu, Docker Compose, Python 3.11.15  
**Test runner:** pytest 8.3.3  
**Champion model:** XGBoost v8, test macro-F1 = 0.780

---

## 1. Summary

| Category | Total | Passed | Failed | Skipped |
|---|---|---|---|---|
| Unit | 12 | **12** | 0 | 0 |
| Integration | 8 | **8** | 0 | 0 |
| **Total** | **20** | **20** | **0** | **0** |

**Result: ALL PASS ✅**

---

## 2. Acceptance Criteria Check

| Criterion | Target | Actual | Result |
|---|---|---|---|
| Unit tests pass | 100% (12/12) | 12/12 | ✅ PASS |
| Integration tests pass | ≥ 95% (≥ 8/8) | 8/8 | ✅ PASS |
| p95 inference latency | < 200ms | **~9.5ms** | ✅ PASS |
| Test set macro-F1 | ≥ 0.75 | **0.780** | ✅ PASS |
| Load distribution | ≥ 2 container IDs | **3 container IDs** | ✅ PASS |

---

## 3. Unit Test Results

### `tests/unit/test_clean.py`

| Test ID | Description | Result | Time |
|---|---|---|---|
| TC01 | `clean_text` lowercases input | ✅ PASS | 0.001s |
| TC02 | `clean_text` strips HTTP/HTTPS URLs | ✅ PASS | 0.001s |
| TC03 | `clean_text` strips @mentions | ✅ PASS | 0.001s |
| TC04 | `clean_text` collapses whitespace | ✅ PASS | 0.001s |
| TC05 | `clean_text` returns "" for None and 123 | ✅ PASS | 0.001s |
| TC06 | `clean_text` preserves `!` and `?` | ✅ PASS | 0.001s |

### `tests/unit/test_drift.py`

| Test ID | Description | Result | Time |
|---|---|---|---|
| TC07 | `DriftDetector.compute()` returns 0.0 with empty window (< 20 texts) | ✅ PASS | 0.003s |
| TC08 | Low drift for similar word distribution | ✅ PASS | 0.008s |
| TC09 | Higher drift for OOD vocabulary | ✅ PASS | 0.006s |

### `tests/unit/test_features.py`

| Test ID | Description | Result | Time |
|---|---|---|---|
| TC10 | `HandcraftedFeatures.transform` output shape (2, 12) | ✅ PASS | 0.512s |
| TC11 | `HandcraftedFeatures` outputs all finite (no NaN/inf) | ✅ PASS | 0.498s |
| TC12 | `build_tfidf` output shape: rows match input count | ✅ PASS | 0.002s |

---

## 4. Integration Test Results

All tests run against live services: `mlflow`, `fastapi` (3 replicas), `nginx`.

| Test ID | Description | Result | Time |
|---|---|---|---|
| TC13 | `GET /health` returns 200 with `status=ok` and `container_id` | ✅ PASS | 0.012s |
| TC14 | `GET /ready` returns 200 (model loaded) | ✅ PASS | 0.008s |
| TC15 | `GET /metrics` returns 200 with `http_requests_total` in body | ✅ PASS | 0.015s |
| TC16 | `POST /predict` returns valid probabilities summing to ≈ 1.0 | ✅ PASS | 0.127s |
| TC17 | `POST /predict` with empty text returns 422 | ✅ PASS | 0.009s |
| TC18 | `POST /predict` with missing field returns 422 | ✅ PASS | 0.008s |
| TC19 | `POST /feedback` accepted and returns `status=logged` | ✅ PASS | 0.011s |
| TC20 | 30 requests show ≥ 2 distinct `container_id` values | ✅ PASS | 0.374s |

**Load distribution observed:** 3 distinct container IDs (all 3 replicas served requests).

---

## 5. Performance Metrics (from Prometheus during test run)

| Metric | Value |
|---|---|
| Request rate | 0.402 req/s |
| p95 inference latency | 9.5ms |
| Error rate | 0.0 (no 5xx errors) |
| Drift score during test | 0.302 |

---

## 6. How to Regenerate

```bash
# Full suite (requires docker compose up)
.venv/bin/pytest -v

# Unit tests only (no Docker needed)
.venv/bin/pytest tests/unit/ -v

# Integration tests only
.venv/bin/pytest tests/integration/ -v

# With HTML report
.venv/bin/pytest -v --tb=short 2>&1 | tee docs/test_output.txt
```

---

## 7. Known Limitations

| Item | Description | Status |
|---|---|---|
| Predict tests skip in CI | `test_predict_returns_valid_probs` and `test_load_distribution_across_containers` skip in GitHub Actions because no model is trained in the CI runner. They pass fully in local runs with a loaded champion. | By design — `model_ready` fixture controls skipping |
| Drift calibration | `DriftDetector` compares a 200-text rolling window against a 500-text baseline; even in-distribution text shows JSD ≈ 0.3 due to sample-size difference. The test threshold (`< 0.3`) in `test_drift_low_for_similar_distribution` is correctly set to reflect this. | Documented limitation |
| Load test not automated | `scripts/load_tester.py` runs a load test but is not yet included in the pytest suite. p95 latency is verified manually via Prometheus/Grafana. | Future: add pytest-benchmark integration |
