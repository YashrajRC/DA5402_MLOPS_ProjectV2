# Test Cases

| ID | Description | Input | Expected | Type |
|---|---|---|---|---|
| TC01 | clean_text lowercases | "HELLO World" | "hello world" | Unit |
| TC02 | clean_text removes URLs | "check http://x.com" | URL absent | Unit |
| TC03 | clean_text removes @mentions | "hi @user how are you" | "@user" absent | Unit |
| TC04 | clean_text collapses whitespace | "hello    world" | "hello world" | Unit |
| TC05 | clean_text handles non-string | None, 123 | "" | Unit |
| TC06 | clean_text preserves punctuation | "Hi!" | "!" retained | Unit |
| TC07 | HandcraftedFeatures output shape | 2 samples | shape (2, 12) | Unit |
| TC08 | HandcraftedFeatures all finite | ["", "short", long_str] | no NaN/inf | Unit |
| TC09 | TF-IDF fit_transform shape | 3 docs | rows = 3 | Unit |
| TC10 | DriftDetector returns 0 when window empty | baseline, no texts | 0.0 | Unit |
| TC11 | DriftDetector low on similar data | baseline=[i,feel,sad], adds same words | < 0.3 | Unit |
| TC12 | DriftDetector high on OOD | baseline=[i,feel,sad], adds gibberish | > 0.2 | Unit |
| TC13 | GET /health returns 200 | — | 200 + status=ok | Integration |
| TC14 | GET /ready returns 200 when model loaded | — | 200 or 503 | Integration |
| TC15 | GET /metrics exposes Prometheus format | — | 200, contains http_requests_total | Integration |
| TC16 | POST /predict valid text | {"text":"I feel anxious"} | 200, probs sum ≈ 1.0 | Integration |
| TC17 | POST /predict empty text rejected | {"text":""} | 422 | Integration |
| TC18 | POST /predict missing field rejected | {} | 422 | Integration |
| TC19 | POST /feedback accepted | full payload | 200, status=logged | Integration |
| TC20 | Nginx distributes across containers | 30 requests | ≥ 2 distinct container_ids | Integration |
| TC21 | p95 latency under load | 100 concurrent | p95 < 200ms | Load |
