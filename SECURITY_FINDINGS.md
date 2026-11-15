# AccessGuard — Security Findings & Threat Model Register

**Assessment Framework:** STRIDE Threat Modeling + Static Application Security Testing (Bandit) + Dynamic API Verification  
**Evaluation Date:** November 2026  
**Status:** ALL CRITICAL & HIGH VULNERABILITIES REMEDIATED (0 High, 0 Medium)

---

## 1. Vulnerability Findings & Remediation Register

| Vulnerability ID | Title | Threat Category (STRIDE) | Severity | Status | Remediated In |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-001** | Unsafe PyTorch Deserialization (`torch.load`) | Elevation of Privilege | **High** (Bandit B614) | **REMEDIATED** | `backend/app/cv/adapters/torchvision_adapter.py` |
| **SEC-002** | Hardcoded Development `SECRET_KEY` in Config | Information Disclosure / Spoofing | **High** | **REMEDIATED** | `backend/app/config.py` |
| **SEC-003** | Missing API Rate Limiting on Auth & Upload Endpoints | Denial of Service | **High** | **REMEDIATED** | `backend/app/core/limiter.py` |
| **SEC-004** | Wildcard Permissive CORS Policy in `main.py` | Information Disclosure | **Medium** | **REMEDIATED** | `backend/app/main.py` |
| **SEC-005** | Image Decompression Bomb Vulnerability | Denial of Service | **Medium** | **REMEDIATED** | `backend/app/core/sanitize.py` |
| **SEC-006** | EXIF Privacy Metadata & Facial Identity Leakage | Information Disclosure | **Medium** | **VERIFIED CLEAN** | `backend/app/core/sanitize.py` |
| **SEC-007** | Sybil Verification Manipulation | Tampering | **Low** | **MITIGATED** | `backend/app/api/v1/verifications.py` |

---

## 2. STRIDE Threat Model Assessment

### S — Spoofing Identity
- **Risk:** Attackers submitting false observations or counterfeit verifications under spoofed contributor identities.
- **Defense Implemented:** Signed HS256 JWT tokens with 24-hour expiration, hashed passwords using salted bcrypt (cost 12), and role claims embedded in token payloads.

### T — Tampering with Data
- **Risk:** Contributor altering observation statuses or corrupting pedestrian graph edges.
- **Defense Implemented:** Segment edge geometry and accessibility scores are persisted in relational tables with foreign keys; consensus algorithm requires $\ge 3$ reviews with $\ge 67\%$ consensus to alter status.

### R — Repudiation
- **Risk:** Contributors disputing uploaded observations or verification actions.
- **Defense Implemented:** Every observation and verification review logs contributor ID, reviewer ID, timestamp, and audit trail in `verification_reviews`.

### I — Information Disclosure
- **Risk:** Pedestrian face recognition, license plate surveillance, or home GPS leak via uploaded photos.
- **Defense Implemented:** In-memory EXIF extraction followed by immediate byte stripping and re-encoding; OpenCV Haar cascades apply aggressive Gaussian blur over human faces before persistence. Zero raw EXIF is stored on disk.

### D — Denial of Service
- **Risk:** Pixel flood decompression bombs, massive video uploads, or auth endpoint brute forcing.
- **Defense Implemented:** Strict `MAX_IMAGE_PIXELS = 16_000_000` ($4096\times 4096$) header bounds; sliding-window rate limiting via `slowapi` ($30\text{ req/min}$ on login, $60\text{ req/min}$ on upload); video duration capped at 60 seconds.

### E — Elevation of Privilege
- **Risk:** Unprivileged contributors approving administrative overrides or executing code via model checkpoints.
- **Defense Implemented:** Hierarchical RBAC enforcement via `require_role(Role.ADMIN)` on administrative endpoints; `weights_only=True` enforced on PyTorch model loaders.
