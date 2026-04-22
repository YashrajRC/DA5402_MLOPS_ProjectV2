# Test Plan

## 1. Scope

Tests cover the core classification pipeline (data cleaning, feature engineering, drift detection) and the REST API (contract, validation, load distribution). Out of scope: Airflow task-level testing, Prometheus/Grafana rendering (manual verification).

## 2. Test Types

| Type | Tool | Location |
|---|---|---|
| Unit | pytest | `tests/unit/` |
| Integration | pytest + requests | `tests/integration/` |
| Load | `scripts/load_tester.py` | `scripts/` |

## 3. How to Run

```bash
# Unit tests (no services needed)
pytest tests/unit/

# Integration tests (services must be running)
docker compose up -d
pytest tests/integration/

# Load test
python scripts/load_tester.py
```

## 4. Acceptance Criteria

The build passes if **all** of the following hold:

1. **All unit tests pass** (≥ 12 cases)
2. **≥ 95% of integration tests pass** (≥ 8 cases)
3. **p95 latency under load < 200ms** (via `load_tester.py` + Grafana)
4. **Macro-F1 on test set ≥ 0.75** (reported by `dvc metrics show`)
5. **Load tester shows ≥ 2 distinct container IDs** (proves Nginx distribution)

## 5. Deviation Handling

If any criterion fails, the result is logged in `test_report.md` under "Known Issues" with a remediation plan.
