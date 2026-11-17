# Dataset Integrity Audit Report (Mission 006.1)

**Audit Date:** 2026-11-16T06:14:16Z  
**Status:** **DEFECTS_DETECTED**  
**Audit Duration:** 0.53s  

## Summary Metrics

| Metric | Measured Value | Acceptance Threshold | Result |
|---|---|---|---|
| Total Images Checked | 179 | > 0 | PASS |
| Image Decodability Rate | 64.8% | 100% | DEFECT |
| Total Annotations Checked | 25 | > 0 | PASS |
| Annotation Validity Rate | 100.0% | 100% | PASS |
| Duplicate Images | 85 | 0 | WARNING |
| Total Defects Logged | 63 | 0 | DEFECTS DETECTED |

## Inspected Datasets

### indian_streetscapes
- **Directory:** `data/benchmarks/indian_streetscapes`
- **Images Valid:** 25 / 25
- **Annotations Valid:** 25 / 25
- **Defects Count:** 0

### processed
- **Directory:** `data/processed`
- **Images Valid:** 81 / 144
- **Annotations Valid:** 0 / 0
- **Defects Count:** 63

### uploads
- **Directory:** `data/uploads`
- **Images Valid:** 10 / 10
- **Annotations Valid:** 0 / 0
- **Defects Count:** 0

## Duplicate Images Detected

- **SHA-256:** `3fe3422dc6762817...`
  - File A: `data/processed\05b07c34-d826-44d9-9472-e7fc67a40810.jpg`
  - File B: `data/processed\09881425-777e-4fa5-81fc-6ebfc86daf75.jpg`
- **SHA-256:** `3fe3422dc6762817...`
  - File A: `data/processed\05b07c34-d826-44d9-9472-e7fc67a40810.jpg`
  - File B: `data/processed\09ae929a-ec1a-489a-8fa3-412e73c44106.jpg`
- **SHA-256:** `739c19c60795f59b...`
  - File A: `data/processed\078c3253-6548-4410-a9af-7dd63c2619cd.jpg`
  - File B: `data/processed\0a59fce7-3b34-4c36-b2c2-40cc9413531c.jpg`
- **SHA-256:** `3fe3422dc6762817...`
  - File A: `data/processed\05b07c34-d826-44d9-9472-e7fc67a40810.jpg`
  - File B: `data/processed\0f0dc91f-0d0c-415b-a9d1-21fa86b692f3.jpg`
- **SHA-256:** `3fe3422dc6762817...`
  - File A: `data/processed\05b07c34-d826-44d9-9472-e7fc67a40810.jpg`
  - File B: `data/processed\16c16d65-5b16-4fbe-85ee-9d6183cc3c33.jpg`
## Logged Defects

- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
- **Type:** `IMAGE_INTEGRITY_FAILURE` | **Dataset:** `processed` | **Details:** Blank or uniform placeholder (std=0.00)
