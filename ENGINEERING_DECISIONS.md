# AccessGuard — Engineering Decision Records (EDRs)

This document records architectural decisions made throughout the engineering lifecycle of AccessGuard, including the technical rationale, tradeoffs considered, and implementation consequences.

---

## EDR-001: Separation of Heuristic CV from Deep Learning Inference
- **Context:** AccessGuard requires real-time pedestrian barrier detection from edge-uploaded imagery, but users run deployments on diverse hardware (from laptops with no GPU to cloud instances with CUDA).
- **Decision:** Implement a dual-mode adapter architecture:
  1. `HeuristicCVAdapter`: Zero-dependency, classical OpenCV computer vision (Sobel, Hough, Otsu, Laplacian). Runs on any CPU with sub-25ms latency.
  2. `TorchvisionAdapter` / `YOLOAdapter`: Deep learning adapters that execute PyTorch tensors when custom weights and CUDA are available, falling back gracefully to OpenCV heuristics on CPU.
- **Consequences:** Transparent, highly reliable baseline that works out-of-the-box anywhere without failing or hanging on missing CUDA drivers.

---

## EDR-002: Default Scoring Semantics for Unsurveyed Sidewalks (Grade "U")
- **Context:** Initially, pedestrian segments with zero observations defaulted to score 100.0 and Grade "A", falsely conveying that unobserved sidewalks are wheelchair-accessible corridors.
- **Decision:** Introduce Grade `"U"` (Unassessed) and default score $50.0$ with `is_assessed: False`. In Dijkstra/A* routing, apply a caution cost weight multiplier ($2.5\times$) to unassessed segments, forcing the route optimizer to prioritize verified accessible sidewalks.
- **Consequences:** Adheres strictly to the accessibility safety principle: *absence of evidence is not evidence of accessibility*.

---

## EDR-003: Signed Gradient Direction Check for Zebra Crosswalk Disambiguation
- **Context:** Parallel horizontal lines on pedestrian zebra crosswalks triggered the stair detector with $94\%$ false confidence in classical Hough line transforms.
- **Decision:** Use signed vertical Sobel gradients (`cv2.Sobel`) across candidate line clusters. Flat painted stripes produce rapid alternating positive (+grad, dark-to-bright) and negative (-grad, bright-to-dark) peaks across stripe edges. Stair risers produce unidirectional gradient transitions.
- **Consequences:** False-positive rate on standard zebra crosswalks dropped from $100\%$ to $0\%$, verified with executable tests.

---

## EDR-004: In-Memory Sliding Window Rate Limiting via SlowAPI
- **Context:** Authentication and media upload endpoints were vulnerable to brute-force credential stuffing and denial of service via file flood attacks.
- **Decision:** Integrate `slowapi` with in-memory key-based token bucket tracking:
  - `/api/v1/auth/login`: 30 requests/minute
  - `/api/v1/auth/register`: 20 requests/minute
  - `/api/v1/observations/upload`: 60 requests/minute
- **Consequences:** Protects application against automated attacks without requiring external Redis dependencies in local/standalone deployments.

---

## EDR-005: A* Search Integration with Haversine Heuristic
- **Context:** NetworkX Dijkstra computes shortest paths exhaustively expanding radial frontiers. On larger urban pedestrian grids, search overhead increases $O(V \log V + E)$.
- **Decision:** Introduce A* search using the great-circle Haversine distance from current node to destination as the admissible heuristic $h(n)$.
- **Consequences:** Speeds up path computation on large networks while preserving optimal barrier avoidance.
