# AccessGuard System Architecture

## 1. Executive Summary

AccessGuard is an open-source, public-benefit accessibility intelligence platform that analyzes street-level images and short videos to detect wheelchair accessibility barriers, maintain a verifiable geospatial record, score individual sidewalk segments, and recommend barrier-free pedestrian routes.

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client["Client Application (React 18 + TypeScript + TailwindCSS + Leaflet)"]
        Dashboard["Map Dashboard\n(Color-coded Segments & Barrier Markers)"]
        UploadUI["Media Upload\n(Drag & Drop, EXIF, Video Keyframes)"]
        InspectUI["Barrier Inspector\n(Bounding Box, Mask & Evidence Breakdown)"]
        RoutingUI["Wheelchair Route Planner\n(Avoids Stairs, Slope Penalization)"]
        QueueUI["Human Verification Queue\n(Community Peer Review)"]
    end

    subgraph API_GW["API Layer (FastAPI 0.110+)"]
        AuthMiddleware["RBAC & JWT Auth\n(Admin, Verifier, Contributor, Public)"]
        Sanitizer["Sanitization & Privacy Engine\n(Face/Plate Blurring & EXIF Stripping)"]
        ObsRouter["Observations Router"]
        SegmentRouter["Pedestrian Segments Router"]
        RoutingRouter["Accessible Routing Router"]
        VerifyRouter["Verification & Consensus Router"]
        AnalyticsRouter["Analytics & Health Router"]
    end

    subgraph CV_Pipeline["Computer Vision Pipeline"]
        AdapterManager["Interchangeable Model Adapter Manager"]
        HeuristicCV["OpenCV Heuristic Feature Engine\n(Sobel Horizontal Rungs, Laplacian Cracks)"]
        PyTorchTorchvision["PyTorch Mask R-CNN Adapter\n(Instance Segmentation)"]
        YoloONNX["YOLO / ONNX Interchangeable Adapter"]
        UncertaintyCalibrator["Uncertainty Calibrator & Evidence Generator\n(ECE Calibration, Signal-to-Noise Ratio)"]
    end

    subgraph GIS_Engine["Geospatial & Routing Engine"]
        SpatialIndex["Spatial Snap & Distance Engine\n(Shapely Point-to-Line Projection)"]
        ScoringEngine["Transparent Accessibility Scoring\n(Base 100 - Penalties + Bonuses)"]
        AStarRouter["Graph Routing Engine\n(NetworkX Dijkstra/A* Barrier-Aware Cost)"]
    end

    subgraph Persistence["Storage & Database Layer"]
        PostgresDB[("PostgreSQL 16 + PostGIS\n(Spatial LineStrings, Points, GIST Indexes)")]
        MediaStorage[("Sanitized Media Storage\n(Anonymized JPEG/MP4 Frames)")]
    end

    Client -->|HTTPS / REST| API_GW
    API_GW --> Sanitizer
    Sanitizer --> CV_Pipeline
    CV_Pipeline --> SpatialIndex
    SpatialIndex --> PostgresDB
    GIS_Engine --> PostgresDB
    API_GW --> MediaStorage
    API_GW --> GIS_Engine
```

---

## 3. Component Deep Dive

### 3.1. Computer Vision Pipeline

AccessGuard enforces strict interchangeability across vision backends through the `ModelAdapter` abstract interface:

```mermaid
classDiagram
    class ModelAdapter {
        <<abstract>>
        +name: str
        +is_initialized: bool
        +load() bool*
        +predict(image_bgr: ndarray) List~DetectedBarrier~*
        +predict_batch(images: List~ndarray~) List~List~DetectedBarrier~~
    }

    class HeuristicCVAdapter {
        +_detect_stairs(gray, w, h) DetectedBarrier
        +_detect_surface_damage(gray, w, h) DetectedBarrier
        +_detect_vertical_obstacles(gray, w, h) DetectedBarrier
        +_detect_curb_ramps(gray, w, h) DetectedBarrier
    }

    class TorchvisionAdapter {
        +device: torch.device
        +model: MaskRCNN
        +transforms: Compose
    }

    class YOLOAdapter {
        +model_path: str
        +net: cv2.dnn.Net
    }

    class DummyAdapter {
        +mode: str
    }

    ModelAdapter <|-- HeuristicCVAdapter
    ModelAdapter <|-- TorchvisionAdapter
    ModelAdapter <|-- YOLOAdapter
    ModelAdapter <|-- DummyAdapter
```

#### Uncertainty Calibration & Evidence Extraction
- **Edge Rung Analysis**: Stairs are identified by clustering horizontal gradient rungs using Sobel filters and Hough line transforms. The step count, regular vertical spacing, and contrast standard deviation are recorded as deterministic evidence.
- **Surface Fracture Roughness**: Cracked or degraded asphalt/concrete is measured via high-frequency edge density and Laplacian variance in the pedestrian corridor region.
- **Explicit Uncertainty**: Borderline detections with confidence $< 0.70$ or signal ambiguity $> 0.35$ are tagged with `is_uncertain = true`. No barrier is marked as `is_confirmed = true` without exceeding $0.85$ confidence AND human consensus.

### 3.2. GIS Spatial Indexing & Snapping

- Pedestrian sidewalks and crosswalks are represented as `LineString` geometries in WGS84 coordinates.
- Incoming GPS coordinates (from EXIF or manual user geotagging) are projected onto candidate street segments using geodesic projection.
- Observations within the $25.0\text{ m}$ snap tolerance are linked to the segment foreign key (`segment_id`).

### 3.3. Transparent Accessibility Scoring Algorithm

Segment accessibility is calculated deterministically with full mathematical transparency:

$$\text{Net Score} = \max\left(0.0, \min\left(100.0, 100.0 - \sum \text{Penalties} + \sum \text{Bonuses}\right)\right)$$

| Barrier Type | Raw Impact | Justification |
| :--- | :--- | :--- |
| **Stairs (Unramped)** | -45 pts | Complete vertical obstruction for manual wheelchairs. |
| **Missing Curb Ramp** | -35 pts | Impassable street crossing drop-off. |
| **Steep Incline** | -30 pts | Exceeds ADA maximum grade ($> 8.3\%$). |
| **Construction / Narrow Path** | -25 pts | Corridor width constriction $< 90\text{ cm}$. |
| **Surface Damage / Potholes** | -20 pts | Caster wheel trap / tip-over hazard. |
| **Accessible Curb Ramp** | +15 pts | Compliant flared ADA transition. |

**Weighting by Verification Status:**
- `verified` (multi-peer confirmed): $1.0\times$ weight
- `tentative` (unreviewed machine flag): $0.5\times$ weight
- `disputed` / `rejected`: $0.0\times$ weight

---

## 4. Multi-Agent Responsibilities

To ensure separation of concerns and robust verification, AccessGuard is organized into dedicated architectural sub-disciplines:

1. **System Architecture Agent**: Defines interfaces, directory contracts, and configuration schemas.
2. **CV Model Pipeline Agent**: Implements model adapters, video frame samplers, and evidence extraction.
3. **Backend & Database Agent**: Enforces PostGIS schemas, CRUD operations, and async media handlers.
4. **GIS Processing Agent**: Manages spatial indexing, snapping, transparent scoring, and graph A* routing.
5. **Frontend & Maps Agent**: Builds interactive Leaflet visualization, route planning controls, and peer review queues.
6. **Security & Threat Modeling Agent**: Audits upload validation, enforces RBAC, strips EXIF, and prevents facial recognition.
7. **QA & Integration Agent**: Develops test fixtures, validates mAP and ECE benchmarks, and configures CI workflows.
