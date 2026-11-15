# AccessGuard Prioritized Remediation Backlog

This backlog organizes all engineering findings from Mission 002 by priority (P0 to P3), providing specific acceptance criteria and remediation blueprints.

---

## 1. Backlog Overview

| Priority | ID | Issue Title | Domain | Effort | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0** | **REM-01** | Fix Silent Grade A Classification for Unsurveyed Sidewalks | GIS / Safety | 2h | **Remediated** |
| **P0** | **REM-02** | Enforce Maximum Endpoint Snap Distance Threshold ($500\text{m}$) | GIS / Routing | 1h | **Remediated** |
| **P0** | **REM-03** | Fix Unsafe PyTorch Deserialization (`weights_only=True`) | Security | 30m | **Remediated** |
| **P0** | **REM-04** | Enforce Production Environment Check for `SECRET_KEY` | Security | 30m | **Remediated** |
| **P1** | **REM-05** | Filter Zebra Crosswalk False Positives from Stair Detector | Computer Vision | 3h | **Remediated** |
| **P1** | **REM-06** | Enforce Restricted CORS Origins in `main.py` | Security | 30m | **Remediated** |
| **P1** | **REM-07** | Add Image Pixel Dimension Bounds (Decompression Bomb Protection) | Security / Reliability | 1h | **Remediated** |
| **P1** | **REM-08** | Implement API Rate Limiting on Auth and Upload Endpoints | Security | 3h | **Remediated** |
| **P2** | **REM-09** | Expand Barrier Taxonomy & Evaluation for Indian Infrastructure | CV / Domain | 1d | Backlog |
| **P2** | **REM-10** | Provide Default CPU Weights for `TorchvisionAdapter` | CV / DL | 1d | Backlog |
| **P2** | **REM-11** | Formalize Database Migrations with Alembic | Database | 4h | Backlog |
| **P3** | **REM-12** | Clean Up Deprecated `version` in `docker-compose.yml` | DevOps | 15m | **Remediated** |
| **P3** | **REM-13** | Replace Silent JSON Exception Swallowing with Structured Logs | Code Quality | 1h | Backlog |

---

## 2. Priority P0 (Immediate Safety & Security Remediations)

### REM-01: Fix Silent Grade A Classification for Unsurveyed Sidewalks
- **Problem**: When a pedestrian segment has 0 observations, `accessibility_score` defaults to $100.0$ and `score_grade` defaults to `"A"`. This falsely signals that unsurveyed segments are pristine wheelchair-accessible corridors.
- **Remediation**:
  1. Add `score_grade = "U"` (Unassessed) and `accessibility_score = None` (or $50.0$) when observations count is 0.
  2. In `calculate_segment_accessibility_score`, if `len(observations) == 0`:
     - Return `score = 50.0`, `grade = "U"`, with clear breakdown note: `"Segment has not been field surveyed. Accessibility is unconfirmed."`

### REM-02: Enforce Maximum Endpoint Snap Distance Threshold ($500\text{m}$)
- **Problem**: Passing coordinates in Tokyo against San Francisco network snaps across the Pacific Ocean ($8,274\text{ km}$).
- **Remediation**:
  In `backend/app/gis/routing_engine.py`:
  ```python
  MAX_SNAP_DISTANCE_METERS = 500.0
  if d_start > MAX_SNAP_DISTANCE_METERS or d_end > MAX_SNAP_DISTANCE_METERS:
    return RouteResponse(
        success=False,
        ...,
        routing_notes=[
            f"Endpoint is too far from mapped pedestrian network (Start:"
            f" {d_start:.0f}m, End: {d_end:.0f}m. Max: {MAX_SNAP_DISTANCE_METERS}m)."
        ],
    )
```

### REM-03: Fix Unsafe PyTorch Deserialization (`weights_only=True`)
- **Problem**: Bandit B614 flag on `torch.load(self.weights_path)`.
- **Remediation**:
  In `backend/app/cv/adapters/torchvision_adapter.py:42`:
  ```python
  self.model = torch.load(
      self.weights_path, map_location=self.device, weights_only=True
  )
```

### REM-04: Enforce Production Environment Check for `SECRET_KEY`
- **Problem**: Default fallback secret key is predictable.
- **Remediation**:
  In `backend/app/config.py`: Add validator checking if `ENVIRONMENT == "production"` and `SECRET_KEY` matches default development string, raising a `ValueError`.

---

## 3. Priority P1 (High Priority Remediations)

### REM-05: Filter Zebra Crosswalk False Positives from Stair Detector
- **Problem**: Zebra crosswalk white painted stripes trigger stair detection with $94\%$ confidence.
- **Remediation**:
  In `HeuristicCVAdapter._detect_stairs`:
  1. Check aspect ratio and width of stripes: painted zebra stripes are uniformly thick and flat on the ground plane without riser shadow gradients.
  2. Compute vertical gradient derivative: concrete stairs have sharp riser-to-tread drop-offs with dark under-step shadow crevices. Painted stripes have flat reflectance without drop-off shadows.

### REM-06: Enforce Restricted CORS Origins
- **Remediation**: In `backend/app/main.py:77`, change `allow_origins=["*"]` to `allow_origins=settings.BACKEND_CORS_ORIGINS`.

### REM-07: Decompression Bomb Safeguard
- **Remediation**: In `backend/app/core/sanitize.py`, enforce `MAX_PIXELS = 16_000_000` ($4000 \times 4000$). Check image size using `Image.open` metadata before allocating byte arrays.
