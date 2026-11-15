# AccessGuard REST API Documentation

The AccessGuard API follows standard RESTful patterns with JSON request and response payloads, OpenAPI 3.0 specification compliance, and JWT bearer authentication.

- **Base URL**: `/api/v1`
- **Interactive Swagger UI**: `/docs`
- **ReDoc Documentation**: `/redoc`

---

## 1. Authentication Endpoints

### `POST /auth/register`
Creates a new user account.
- **Request Body**:
  ```json
  {
    "email": "contributor@example.org",
    "password": "SecurePassword123!",
    "role": "contributor"
  }
  ```
- **Response `201 Created`**:
  ```json
  {
    "id": "c1f7a3f8-8e81-4223-96b6-3eb3717dfb11",
    "email": "contributor@example.org",
    "role": "contributor",
    "reputation_score": 10,
    "is_active": true,
    "created_at": "2026-11-15T12:00:00Z"
  }
  ```

### `POST /auth/login`
Authenticates credentials and issues signed JWT bearer token.
- **Request Body**:
  ```json
  {
    "email": "contributor@example.org",
    "password": "SecurePassword123!"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsIn...",
    "token_type": "bearer",
    "user_id": "c1f7a3f8-8e81-4223-96b6-3eb3717dfb11",
    "email": "contributor@example.org",
    "role": "contributor"
  }
  ```

---

## 2. Observations Endpoints

### `POST /observations/upload`
Uploads a street-level image or short video for automated obstacle inference.
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: Media binary (`image/jpeg`, `image/png`, `video/mp4`)
  - `latitude` (optional): Float, e.g. `37.7765`
  - `longitude` (optional): Float, e.g. `-122.4194`
  - `adapter` (optional): `"heuristic"`, `"torchvision"`, `"yolo"`, or `"dummy"`
- **Response `201 Created`**:
  ```json
  {
    "id": "obs-91823-abc",
    "contributor_id": "c1f7a3f8-8e81-4223-96b6-3eb3717dfb11",
    "segment_id": "seg_central_plaza_stairs",
    "media_url": "/api/v1/observations/obs-91823-abc/media",
    "media_type": "image",
    "latitude": 37.7771,
    "longitude": -122.4194,
    "status": "tentative",
    "verification_score": 0,
    "detections": [
      {
        "id": "det-101",
        "barrier_type": "stairs",
        "confidence": 0.94,
        "uncertainty_score": 0.10,
        "is_confirmed": true,
        "is_uncertain": false,
        "bbox": {
          "ymin": 0.45,
          "xmin": 0.12,
          "ymax": 0.88,
          "xmax": 0.88
        },
        "evidence": {
          "feature_type": "periodic_horizontal_edges",
          "measured_metric": 5.0,
          "threshold_applied": 3.0,
          "explanation": "5 concrete step rungs detected. Impassable for standard wheelchair."
        }
      }
    ]
  }
  ```

### `GET /observations`
Lists observations with optional query filtering.
- **Query Parameters**:
  - `barrier_type`: Filter by obstacle class (e.g. `stairs`)
  - `status`: Filter by review state (`tentative`, `verified`, `disputed`)
  - `segment_id`: Filter by associated pedestrian sidewalk
  - `limit`: Integer, default `50`
  - `offset`: Integer, default `0`

---

## 3. Pedestrian Segments & Scoring Endpoints

### `GET /segments`
Retrieves all pedestrian segments with current accessibility scores and transparent criteria.

### `GET /segments/geojson`
Returns the complete sidewalk network as a GeoJSON `FeatureCollection` for interactive map layers.

### `POST /segments/{id}/recalculate`
Forces real-time recalculation of the transparent score using the segment's linked observations.

---

## 4. Accessible Routing Endpoints

### `POST /routing/accessible-route`
Computes an optimal wheelchair-accessible pedestrian route avoiding stairs and impassable obstacles.
- **Request Body**:
  ```json
  {
    "start": { "latitude": 37.7765, "longitude": -122.4194 },
    "destination": { "latitude": 37.7778, "longitude": -122.4194 },
    "wheelchair_profile": "manual_wheelchair",
    "avoid_unverified_barriers": true
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "success": true,
    "total_distance_meters": 184.2,
    "estimated_duration_minutes": 3.1,
    "overall_accessibility_score": 96.5,
    "route_grade": "A",
    "avoids_stairs": true,
    "barriers_encountered": [],
    "path_coordinates": [
      [37.7765, -122.4194],
      [37.7768, -122.4186],
      [37.7771, -122.4178],
      [37.7778, -122.4194]
    ],
    "segments_traversed": [
      "seg_plaza_bypass_ramp_east",
      "seg_plaza_bypass_ramp_north"
    ],
    "routing_notes": [
      "Optimized for manual wheelchair.",
      "Selected route avoids severe stairs via Plaza East Accessible Ramp Bypass.",
      "Segment accessibility average: 96.5/100 (Grade A)."
    ],
    "geojson": {
      "type": "FeatureCollection",
      "features": [...]
    }
  }
  ```

---

## 5. Human Verification Endpoints

### `POST /verifications/submit`
Submits community verification or dispute. Requires authenticated account.
- **Request Body**:
  ```json
  {
    "observation_id": "obs-91823-abc",
    "vote": "confirm",
    "notes": "Verified in person: 5 concrete steps with no ramp."
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "review_id": "rev-501",
    "observation_id": "obs-91823-abc",
    "new_status": "verified",
    "verification_score": 3,
    "consensus_reached": true,
    "status_changed": true,
    "created_at": "2026-11-15T12:30:00Z"
  }
  ```

### `GET /verifications/queue`
Retrieves unverified/tentative observations requiring community peer review.
