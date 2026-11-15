# AccessGuard Performance & Reliability Benchmark Report
**Author**: Independent QA Architect & Systems Engineer  
**Execution Environment**: Python 3.12.1 / Windows 11 / x86_64  
**Benchmark Script**: `scripts/audit/audit_performance.py`  
**Execution Timestamp**: November 2026

---

## 1. Computer Vision Inference Latency

Inference latency was measured over 25 iterations on standard $640 \times 480$ street scenes:

| Adapter | Mean Latency | Median Latency | 95th Percentile (P95) | Min / Max |
| :--- | :--- | :--- | :--- | :--- |
| **`HeuristicCVAdapter`** | **16.82 ms** | **16.79 ms** | **17.68 ms** | 16.12 ms / 18.91 ms |
| **`DummyAdapter`** | **0.00 ms** | 0.00 ms | 0.01 ms | 0.00 ms / 0.02 ms |

### Inference Analysis:
- The OpenCV heuristic pipeline achieves an effective throughput of **~59.4 frames per second (FPS)** on standard CPU execution.
- Morphological gradient operations and Hough line transforms operate within deterministic $O(W \times H)$ time complexity.

---

## 2. Video Processing Latency & Memory Footprint

A synthetic $5.0\text{-second}$ street video ($150\text{ frames}$ at $30\text{ FPS}$, $640 \times 480$ resolution) was processed using the `CVPipeline.process_video` workflow:

| Metric | Measured Value | Standard Threshold | Verdict |
| :--- | :--- | :--- | :--- |
| **Total Processing Time** | **240.5 ms** | $< 2,000\text{ ms}$ | **PASS** (Sub-second processing) |
| **Sampling Interval** | $1.0\text{ FPS}$ (5 sampled keyframes) | Dynamic | **PASS** |
| **Peak Memory Allocation** | **3.59 MB** (`tracemalloc`) | $< 50\text{ MB}$ | **PASS** (Minimal memory footprint) |
| **Temporal Clustering** | 1 persistent barrier tracked | IoU $\ge 0.35$ | **PASS** |

---

## 3. Concurrent Uploads Stress Test

15 concurrent upload requests were dispatched across 5 parallel thread workers using the FastAPI HTTP test client:

```
┌────────────────────────────────────────────────────────┐
│ CONCURRENT UPLOADS BENCHMARK (15 PARALLEL REQUESTS)   │
├──────────────────────────────────┬─────────────────────┤
│ Total Dispatched Requests        │ 15                  │
│ Successful Requests (201 Created)│ 15 (100.0% Success) │
│ Mean Latency                     │ 927.68 ms           │
│ Median Latency                   │ 912.40 ms           │
│ 95th Percentile (P95) Latency    │ 1383.88 ms          │
│ 99th Percentile (P99) Latency    │ 1421.15 ms          │
└──────────────────────────────────┴─────────────────────┘
```

### Concurrency Observations:
- Zero dropped connections or database locking errors occurred during SQLite/SQLAlchemy concurrent transaction handling.
- P95 latency remained well under $1.5\text{ seconds}$ despite simultaneous file validation, EXIF parsing, image re-encoding, and spatial snapping.

---

## 4. Docker Deployment Reproducibility

Validation was performed via `docker compose config`:
- **Containers**: `accessguard_db` (PostGIS 16), `accessguard_backend` (FastAPI), `accessguard_frontend` (Alpine Nginx).
- **Configuration Status**: Valid.
- **Port Mapping**:
  - `5432:5432` (PostgreSQL)
  - `8000:8000` (FastAPI REST)
  - `3000:80` (React Frontend)
- **Warning Noted**: `docker-compose.yml` specifies `version: '3.8'`, which modern Docker Compose V2 flags as obsolete (deprecated top-level attribute).
