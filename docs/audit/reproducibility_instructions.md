# AccessGuard Audit Reproducibility Guide

This document provides step-by-step instructions to reproduce every finding, benchmark metric, and test report recorded during the Mission 002 audit.

---

## 1. Prerequisites

- Python 3.12+ with virtual environment
- Node.js 20+ with npm 10+
- Repository root: `AccessGuard/`

---

## 2. Reproducing Audit Phase Tests

### Step 1: Static Security Scan (Bandit)
Execute Bandit against the backend codebase:
```bash
python -m bandit -r backend/app -f json -o scripts/audit/bandit_report.json
```
**Expected Output**:
- Total lines scanned: ~2,624
- Issues found: 1 Medium (B614 `pytorch_load`), 5 Low (B106 hardcoded password parameter, B110 try-except-pass, B112 try-except-continue).

---

### Step 2: Computer Vision Stress Tests & Data Leakage Audit
Execute the CV validation script:
```bash
$env:PYTHONPATH="backend"
python scripts/audit/audit_cv_validation.py
```
**Expected Output**:
- Generates `scripts/audit/cv_audit_results.json`.
- Confirms zebra crosswalk is falsely detected as stairs (`Crosswalk detected as stairs? True`).
- Confirms low contrast / fog leads to false-negative detection.

---

### Step 3: Routing & GIS Graph Edge Case Tests
Execute the GIS audit script:
```bash
$env:PYTHONPATH="backend"
python scripts/audit/audit_routing_gis.py
```
**Expected Output**:
- Generates `scripts/audit/routing_audit_results.json`.
- Confirms disconnected graph returns `success=False` with graceful error.
- Confirms unknown segment defaults to `score=100.0, grade=A`.
- Confirms Tokyo coordinate snaps across continents ($8,274\text{ km}$).
- Confirms stairs avoidance and ramp bypass selection.

---

### Step 4: Security & Upload Vulnerability Tests
Execute the security audit script:
```bash
$env:PYTHONPATH="backend"
python scripts/audit/audit_security.py
```
**Expected Output**:
- Generates `scripts/audit/security_audit_results.json`.
- Confirms hardcoded default `SECRET_KEY` warning.
- Confirms wildcard CORS in `main.py`.
- Confirms absence of rate-limiting middleware.
- Confirms EXIF scrubbing and polyglot script rejection.

---

### Step 5: Performance & Concurrency Benchmark
Execute the performance benchmark script:
```bash
$env:PYTHONPATH="backend"
python scripts/audit/audit_performance.py
```
**Expected Output**:
- Generates `scripts/audit/performance_benchmark_results.json`.
- Heuristic latency: ~16.8 ms mean.
- 5s video processing: ~240 ms, ~3.6 MB peak memory.
- Concurrent uploads: 15/15 succeeded, mean latency ~927 ms.

---

### Step 6: Full Regression Test Suite
Execute the 28 automated pytest test cases:
```bash
$env:PYTHONPATH="backend"
python -m pytest backend/tests -v
```
**Expected Output**:
- `28 passed in ~22s`.
