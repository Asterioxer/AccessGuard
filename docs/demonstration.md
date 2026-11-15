# AccessGuard Demonstration Walkthrough & Functional Verification

This guide walks through the end-to-end verification of AccessGuard's 10 core product requirements on a live deployment.

---

## 1. Prerequisites & Launch

1. Start the backend API:
   ```bash
   $env:PYTHONPATH="backend"; uvicorn app.main:app --port 8000
   ```
2. Start the frontend development server:
   ```bash
   cd frontend; npm run dev
   ```
3. Open browser to `http://localhost:5173`.

---

## 2. Step-by-Step Demonstration Workflow

### Step 1: Interactive Map Dashboard Exploration
- **Action**: Open the **Map Dashboard** tab.
- **Observation**:
  - The map loads centered on the Downtown Innovation Corridor.
  - Pedestrian segments are rendered as colored polylines matching their accessibility score:
    - `seg_central_plaza_stairs` shows as **Rose Red (Grade F / Score ~30)** due to unramped stairs.
    - `seg_plaza_bypass_ramp_north` shows as **Emerald Green (Grade A / Score 100)** with ADA ramp.
    - `seg_civic_walkway_west` shows as **Amber (Grade C / Score ~70)** with surface cracks.
  - Interactive barrier pins mark detected obstacles:
    - 🪜 Stairs marker on Central Plaza.
    - ♿ Accessible curb ramp on Plaza East.
    - ⚡ Surface damage on Civic Walkway.

### Step 2: Inspecting Evidence & Confidence Metrics
- **Action**: Click on the 🪜 **Stairs marker** at Central Plaza.
- **Observation**:
  - The **Observation Detail Modal** opens.
  - Normalized bounding box is rendered directly over the stair rungs.
  - **Confidence**: `94%`.
  - **Uncertainty**: `0.10` (Low uncertainty, high evidence).
  - **Status Badge**: `Confirmed by Evidence`.
  - **Evidence Table**: Explains the exact CV feature extracted: `5 concrete step rungs detected. Impassable for manual wheelchair without lift or ramp.`

### Step 3: Transparent Sidewalk Accessibility Criteria
- **Action**: Click on the red sidewalk segment line (`Central Plaza Steps`).
- **Observation**:
  - The **Segment Scoring Panel** slides in from the top right.
  - Displays:
    - Base score: `100.0 pts`
    - Stairs penalty: `-45.0 pts` (Impassable vertical barrier)
    - Net score: `30.0 / 100 (Grade F)`
  - Transparent itemized list detailing why wheelchair transit is impeded.

### Step 4: Wheelchair-Accessible Route Planning (Stair Avoidance)
- **Action**: Click the **Accessible Routing** tab on the top navigation bar.
- **Action**: Click the benchmark preset button: **Central Plaza Transit** (Origin: `37.7765, -122.4194`, Destination: `37.7778, -122.4194`).
- **Action**: Click **Calculate Accessible Route**.
- **Observation**:
  - The routing engine executes graph Dijkstra/A* traversal with barrier penalties.
  - Direct straight-line distance would pass through the stairs at Central Plaza.
  - **Result**: The router actively detours around the stairs, traversing `seg_plaza_bypass_ramp_east` and `seg_plaza_bypass_ramp_north`.
  - **Green Badge**: `Verified Stair-Free Accessible Route (Grade A)`.
  - Glowing cyan route polyline is displayed on the map showing the safe accessible path.

### Step 5: Community Human Verification & Consensus
- **Action**: Click the **Verification Queue** tab.
- **Observation**:
  - Displays the tentative report on `Civic Center West Promenade` (Cracked concrete surface damage).
  - Status is currently `Tentative` awaiting peer review.
- **Action**: Sign in using the Quick Demo Account: `Verifier` (`verifier@accessguard.org`).
- **Action**: Click **Confirm Barrier**.
- **Observation**:
  - Verification review is registered with reviewer ID.
  - Consensus threshold is updated, upgrading the observation status.
  - Linked sidewalk segment score recalculates automatically.

### Step 6: Street Image Upload & Privacy Sanitize
- **Action**: Click the **Upload Observation** button in the header.
- **Action**: Drag and drop a street scene photo (or select test file).
- **Action**: Select the model adapter: `OpenCV Heuristic Pipeline` or `PyTorch Mask R-CNN`.
- **Action**: Click **Run Inference & Submit**.
- **Observation**:
  - Upload sanitizer validates magic bytes.
  - Faces and vehicle plates are blurred using OpenCV Gaussian redaction.
  - EXIF GPS metadata is stripped before saving.
  - Detected obstacle appears immediately on the map snapped to the nearest sidewalk segment.

---

## 3. Current Limitations & Roadmap

1. **Depth Estimation**: The current CV pipeline estimates obstacle height from camera aspect ratios and edge frequencies; integrating stereo vision or LiDAR point-clouds would yield millimeter-accurate curb height readings.
2. **Dynamic Construction Feeds**: Municipal sidewalk closure permits could be ingested via GTFS-Pathways to complement crowdsourced visual reports.
3. **Automated Satellite Alignment**: Future versions will support automatic segmentation alignment against high-resolution orthophotography.
