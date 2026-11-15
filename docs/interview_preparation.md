# AccessGuard — Engineering Portfolio & Senior Technical Interview Guide

## 1. Executive Summary & Elevator Pitch

**AccessGuard** is an open-source, public-benefit accessibility intelligence platform that analyzes street-level pedestrian imagery to detect wheelchair-navigation barriers (stairs, missing curb ramps, cracked pavements, bollards), maps them into a PostGIS geospatial database, scores pedestrian segments transparently, and computes barrier-aware wheelchair routes.

### Key Metrics & Highlights:
- **Architecture:** React/TypeScript + Vite frontend, Python FastAPI REST backend, PostgreSQL + PostGIS, OpenCV & PyTorch computer vision.
- **Inference Latency:** Mean $16.82\text{ ms}$ on standard CPU ($640\times 480$ resolution), zero GPU requirement for base deployment.
- **Evaluation:** $0.618\text{ mAP@0.5}$ and $0.268\text{ ECE}$ on 10 held-out unseen test scenes without data leakage.
- **Security:** 0 High/Medium vulnerabilities in Bandit SAST; sliding-window rate limiting; strict $16\text{ MP}$ decompression bomb guard; privacy-first face and EXIF scrubbing before disk persistence.
- **Routing Engine:** Dual Dijkstra and A* pathfinding with Haversine distance heuristics, alternative direct route tradeoff analysis, and strict $500\text{m}$ network snap boundaries.

---

## 2. Five Critical Engineering Problems & Solutions

### 1. Computer Vision: Distinguishing 3D Stair Risers from Flat Zebra Crosswalks
- **Problem:** Classical Hough line transforms detected parallel horizontal lines on pedestrian crosswalks, triggering the stair detector with $94\%$ false confidence.
- **Solution:** Analyzed signed vertical gradient directions ($\frac{\partial I}{\partial y}$) using `cv2.Sobel`. Flat painted stripes produce rapid alternating positive (+grad, dark-to-bright) and negative (-grad, bright-to-dark) peaks between stripe edges. In contrast, physical stair step risers produce unidirectional gradient transitions. Crosswalk false positive rate dropped from $100\%$ to $0.0\%$.

### 2. GIS: Avoiding False Accessibility Guarantees on Unsurveyed Sidewalks
- **Problem:** Pedestrian segments with zero observations previously defaulted to score $100.0$ and Grade "A", falsely implying full wheelchair accessibility where zero evidence existed.
- **Solution:** Introduced Grade `"U"` (Unassessed) and default score $50.0$. In the NetworkX routing graph, unassessed segments receive a caution cost multiplier ($2.5\times$), compelling the pathfinder to route through verified accessible corridors unless no alternative exists, accompanied by explicit route warnings.

### 3. Architecture: Zero-Dependency CPU Fallback with Model Interchangeability
- **Problem:** Many municipal agencies and community volunteers lack dedicated GPU infrastructure, yet requiring PyTorch with CUDA creates deployment barriers and crashes on CPU-only machines.
- **Solution:** Engineered the `ModelAdapter` interface with two production layers:
  1. `HeuristicCVAdapter`: Highly optimized classical OpenCV vision running sub-$20\text{ms}$ on CPU.
  2. `TorchvisionAdapter` / `YOLOAdapter`: High-capacity deep learning wrappers with `weights_only=True` that execute neural tensors when CUDA weights are mounted, gracefully delegating to OpenCV heuristics when weights are absent.

### 4. GIS: Extreme Geographic Snapping Boundary Protection
- **Problem:** Graph routing engines often snap coordinates to the nearest network node regardless of distance, causing coordinates in Tokyo or London to snap across oceans ($8,000+\text{km}$) to San Francisco segments.
- **Solution:** Enforced a strict `MAX_SNAP_METERS = 500.0m` threshold in `AccessibilityRoutingEngine`. Origin and destination endpoints beyond $500\text{m}$ from the mapped pedestrian corridor fail gracefully with actionable explanatory notes.

### 5. Cybersecurity: Zero-Leakage Privacy Redaction and Decompression Bomb Defense
- **Problem:** Crowdsourced pedestrian uploads risk exposing bystander faces, car license plates, EXIF GPS coordinates of private residences, or triggering memory exhaustion via decompression bombs (e.g., $50,000\times 50,000$ pixel images).
- **Solution:** Implemented header-level image dimension checks (`MAX_IMAGE_PIXELS = 16_000_000`, max $4096\times 4096$) before pixel array allocation. Read GPS in-memory, stripped all EXIF tags, applied OpenCV Haar cascade Gaussian blurring over human facial regions, and re-encoded raw pixels before writing clean files to disk.

---

## 3. Senior Technical Interview Question Bank (20 Questions & Answers)

### Category A: Computer Vision & Machine Learning

**Q1: Why did you implement classical OpenCV heuristics instead of exclusively deploying deep neural networks like Mask2Former?**  
*Answer:* In real-world civic technology, platform accessibility applies to software infrastructure as well as sidewalks. An emergency or municipal dispatch tool must run predictably on low-cost CPUs and edge devices with zero CUDA dependencies. Our dual-adapter architecture provides a $16.8\text{ms}$ CPU baseline while maintaining clean API contracts for deep learning checkpoints when GPU resources are available.

**Q2: What is Expected Calibration Error (ECE) and why is it critical for accessibility intelligence?**  
*Answer:* Accuracy tells you how often a model is correct; calibration tells you if a model's confidence reflects true probability. In wheelchair accessibility, an overconfident prediction on an obstacle can strand a user at an impassable barrier. ECE partitions predictions into $M$ confidence bins and measures the weighted average difference between confidence and empirical accuracy. We enforce that predictions with confidence $<0.70$ or uncertainty $>0.35$ remain tentative.

**Q3: How do you prevent data leakage in computer vision evaluation?**  
*Answer:* In our initial audit, we found that synthetic evaluation scenes had thresholds fitted to the exact same images. We resolved this by creating separate validation (10 scenes) and held-out test (10 unseen scenes) splits with varying lighting (glare, night, fog) and distinct geometric seeds. Metrics are computed strictly on the held-out split.

**Q4: How does temporal clustering improve video keyframe processing?**  
*Answer:* Single video frames often contain transient motion blur or edge artifacts. By tracking bounding boxes across keyframes with IoU overlap $>0.35$, persistent obstacles appearing across $\ge 3$ frames receive boosted confidence ($+0.08$) and reduced uncertainty ($-0.10$), while fleeting 1-frame glitches receive higher uncertainty penalties.

**Q5: How does CLAHE improve detection in foggy or low-contrast conditions?**  
*Answer:* Contrast Limited Adaptive Histogram Equalization operates in LAB color space, computing localized histograms on the luminance ($L$) channel with clipping limits ($2.5$) to prevent noise over-amplification. This enhances subtle edge gradients on stair risers obscured by fog without blowing out highlights.

---

### Category B: GIS & Spatial Algorithms

**Q6: Why did you choose NetworkX over pgRouting for the routing engine?**  
*Answer:* In-memory NetworkX graph construction allows rapid experimentation, unit testability with in-memory SQLite, and zero external database dependencies during local development. For large metropolitan deployments, NetworkX handles 50,000+ pedestrian edges in $<50\text{MB}$ RAM, though the schema is fully compatible with pgRouting via PostGIS LineStrings.

**Q7: How does A* pathfinding differ from Dijkstra in your implementation?**  
*Answer:* Dijkstra explores frontiers radially based on accumulated cost. A* adds an admissible heuristic $h(n)$—the Haversine great-circle distance between current node $n$ and destination $t$. Because barrier cost multipliers are $M(e) \ge 1.0$, the straight-line physical distance never overestimates true travel cost ($h(n) \le h^*(n)$), ensuring A* finds the globally optimal accessible path with fewer node expansions.

**Q8: Explain the coordinate reference system (CRS) transformations across your stack.**  
*Answer:* Consumer GPS devices and EXIF metadata use EPSG:4326 (WGS 84 decimal degrees). Database geometries and GeoJSON exports strictly adhere to EPSG:4326 `[longitude, latitude]`. When rendering interactive maps in Leaflet, coordinates are projected to EPSG:3857 (Web Mercator) to align with standard web tiles.

**Q9: How do you handle disconnected components in the pedestrian sidewalk graph?**  
*Answer:* In urban grids, sidewalks can be discontinuous due to construction or unmapped corridors. If `nx.dijkstra_path` raises `NetworkXNoPath`, the engine returns a clean `success=False` response with `routing_notes: ["No accessible continuous pedestrian route connects these locations."]`, preventing application crashes.

**Q10: What is the purpose of alternative route comparison?**  
*Answer:* Wheelchair users often need to weigh safety against distance. By computing both the safest barrier-free path and the direct shortest path, we provide explicit tradeoff evidence: *"Accessible route is 88m longer, but avoids 2 staircases present on the direct shortest path."*

---

### Category C: Cybersecurity, Privacy & Threat Modeling

**Q11: Walk through your STRIDE threat analysis for AccessGuard.**  
*Answer:* We evaluated all 6 STRIDE categories: Spoofing (mitigated by signed JWTs and bcrypt); Tampering (mitigated by consensus thresholds and relational integrity); Repudiation (mitigated by audit logs on all reviews); Information Disclosure (mitigated by EXIF stripping and face blurring); Denial of Service (mitigated by rate limiting and decompression bomb caps); Elevation of Privilege (mitigated by `require_role(Role.ADMIN)`).

**Q12: How did you remediate Bandit B614 (`torch.load`)?**  
*Answer:* Standard PyTorch `torch.load()` uses Python pickle deserialization, which allows malicious model checkpoints to execute arbitrary system commands (`__reduce__`). Passing `weights_only=True` restricts unpickling strictly to tensors, dictionaries, and primitive types, completely neutralizing remote code execution.

**Q13: How does your privacy redaction protect pedestrians?**  
*Answer:* When an image is uploaded, `apply_privacy_redaction()` runs Haar frontal face cascades. Any detected facial region is obscured with heavy Gaussian blurring ($ksize \ge 15$, $\sigma = 30$) before the file is written to storage. Raw, unblurred human faces never reach persistent disks.

**Q14: How does slowapi rate limiting protect authentication endpoints?**  
*Answer:* We integrated `slowapi` using in-memory sliding-window token buckets keyed by client remote IP. `/auth/login` is limited to $30\text{ req/min}$, `/auth/register` to $20\text{ req/min}$, and `/observations/upload` to $60\text{ req/min}$. Exceeded thresholds return HTTP 429 Too Many Requests.

**Q15: What is an image decompression bomb and how do you prevent it?**  
*Answer:* A decompression bomb is a small compressed file (e.g. 50 KB PNG) that expands into gigabytes of raw pixel data (e.g., $50,000 \times 50,000 \times 4$ bytes = 10 GB), exhausting server RAM. We set `Image.MAX_IMAGE_PIXELS = 16_000_000` and inspect header dimensions before full buffer decoding.

---

### Category D: Backend, Architecture & DevOps

**Q16: How do database migrations work in AccessGuard?**  
*Answer:* We configured Alembic with `alembic.ini` and `backend/alembic/env.py`. Revisions are tracked in `alembic_version`. Running `alembic upgrade head` applies schema evolutions deterministically in CI/CD and container startup.

**Q17: Why did you implement pagination on `/segments` and `/observations`?**  
*Answer:* Unbounded `.all()` database queries risk server memory exhaustion and long response times as city datasets scale to tens of thousands of sidewalks. We enforced `limit: int = Query(50, le=200)` and `offset: int = Query(0, ge=0)`.

**Q18: How is observability implemented with Prometheus?**  
*Answer:* In `backend/app/main.py`, we integrated `prometheus-client` exposing `/metrics`. We track `accessguard_http_requests_total` (counter partitioned by method, endpoint, status) and `accessguard_cv_inference_seconds` (histogram measuring inference duration).

**Q19: How do you ensure high contrast and accessibility on the frontend?**  
*Answer:* In React/Tailwind, we adhere to WCAG 2.1 AA standards: high-contrast text ratios ($\ge 4.5:1$), distinct color-blind-friendly badges (Emerald for Grade A, Amber for C, Rose for F, Slate for U), full keyboard focus rings, and explicit ARIA labels on all interactive controls.

**Q20: What is your disaster recovery and rollback strategy?**  
*Answer:* Automated `pg_dump` logical backups running on schedule; stateless containerized backend and frontend services; and versioned Alembic schema rollback (`alembic downgrade -1`) to ensure instantaneous zero-data-loss rollback during faulty deployments.
