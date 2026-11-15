# AccessGuard Security Audit & Vulnerability Assessment
**Author**: Independent Application Security Engineer  
**Audit Standard**: OWASP Top 10 API Security (2023), STRIDE Threat Modeling  
**Target Codebase**: `backend/app/`  
**Execution Timestamp**: November 2026

---

## 1. Vulnerability Findings Summary

| ID | Title | Severity | CVSS v3.1 | CWE | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | Missing Rate Limiting on Authentication and Uploads | **HIGH** | 7.5 | CWE-770 | Confirmed |
| **SEC-02** | Unsafe PyTorch Deserialization in Torchvision Adapter | **MEDIUM** | 6.8 | CWE-502 | Confirmed (Bandit B614) |
| **SEC-03** | Permissive CORS Configuration with Wildcard Origins | **MEDIUM** | 5.3 | CWE-942 | Confirmed |
| **SEC-04** | Hardcoded Development SECRET_KEY Default in Config | **MEDIUM** | 5.9 | CWE-1188 | Confirmed |
| **SEC-05** | Decompression Bomb (Pixel Flood) Vulnerability | **MEDIUM** | 6.5 | CWE-400 | Confirmed |
| **SEC-06** | Sybil Consensus Manipulation in Verification Queue | **MEDIUM** | 6.3 | CWE-842 | Confirmed |
| **SEC-07** | Silent Exception Masking in JSON Parsing | **LOW** | 3.1 | CWE-703 | Confirmed (Bandit B110) |

---

## 2. Detailed Vulnerability Analyses

### SEC-01: Missing Rate Limiting on Authentication & Upload Endpoints
- **Severity**: **HIGH (CVSS: 7.5 / AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H)**
- **Location**: `backend/app/main.py`, `backend/app/api/v1/auth.py`, `backend/app/api/v1/observations.py`
- **Description**:
  The FastAPI application does not incorporate any rate-limiting middleware (such as `slowapi` or Redis-backed token buckets).
- **Impact**:
  1. **Credential Brute-Forcing**: Attackers can execute rapid automated dictionary attacks against `/api/v1/auth/login`.
  2. **Disk & CPU Exhaustion**: Attackers can flood `/api/v1/observations/upload` with repeated video or image uploads, exhausting server storage and starving inference CPU cycles.
- **Remediation**:
  Integrate `slowapi` with memory or Redis backend. Enforce limits:
  - `/api/v1/auth/login`: 5 requests / minute per IP.
  - `/api/v1/observations/upload`: 20 requests / hour per user/IP.

---

### SEC-02: Unsafe PyTorch Deserialization (`torch.load`)
- **Severity**: **MEDIUM (CVSS: 6.8 / AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H)**
- **Tool**: Bandit Issue `B614:pytorch_load`
- **Location**: `backend/app/cv/adapters/torchvision_adapter.py:42`
- **Description**:
  ```python
  self.model = torch.load(self.weights_path, map_location=self.device)
  ```
  Standard `torch.load` utilizes Python's `pickle` under the hood. Loading untrusted or tampered `.pt` weight files can execute arbitrary system commands via pickle deserialization gadget chains.
- **Remediation**:
  Enforce safe tensor loading via `weights_only=True`:
  ```python
  self.model = torch.load(
      self.weights_path, map_location=self.device, weights_only=True
  )
```

---

### SEC-03: Permissive CORS Configuration with Wildcard Origins
- **Severity**: **MEDIUM (CVSS: 5.3 / AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)**
- **Location**: `backend/app/main.py:77`
- **Description**:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
  Configuring `allow_origins=["*"]` allows any malicious website loaded in a user's browser to execute authenticated cross-origin requests against AccessGuard.
- **Remediation**:
  Replace `allow_origins=["*"]` with `settings.BACKEND_CORS_ORIGINS`.

---

### SEC-04: Hardcoded Development SECRET_KEY Default
- **Severity**: **MEDIUM (CVSS: 5.9 / AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N)**
- **Location**: `backend/app/config.py:12`
- **Description**:
  ```python
  SECRET_KEY: str = (
      "accessguard-development-secret-key-replace-in-production-32bytesmin"
  )
  ```
  If AccessGuard is launched in a production environment without providing an explicit `SECRET_KEY` environment variable, an attacker can forge administrative JWT tokens offline using the known default secret.
- **Remediation**:
  Add validation in `config.py` to raise an error if `ENVIRONMENT == "production"` and `SECRET_KEY` is the default string.

---

### SEC-05: Decompression Bomb (Pixel Flood) Vulnerability
- **Severity**: **MEDIUM (CVSS: 6.5 / AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H)**
- **Location**: `backend/app/core/sanitize.py:62, 106`, `backend/app/cv/pipeline.py:44`
- **Description**:
  While file byte size is checked ($\le 20\text{ MB}$), image pixel dimensions are never validated prior to decompression. A $1.5\text{ MB}$ compressed JPEG with dimensions $40,000 \times 40,000$ pixels will consume over $4.8\text{ GB}$ of uncompressed RAM during `cv2.imdecode` or `Image.open`.
- **Remediation**:
  Enforce `Image.MAX_IMAGE_PIXELS = 4096 * 4096` ($16.7\text{ MP}$) and inspect image headers before decoding full pixel arrays into memory.

---

### SEC-06: Sybil Consensus Manipulation in Verification Queue
- **Severity**: **MEDIUM (CVSS: 6.3 / AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N)**
- **Location**: `backend/app/api/v1/verifications.py:22-68`
- **Description**:
  Anyone can call `POST /auth/register` to create unlimited accounts without email verification. An automated bot can register 3 accounts and submit 3 "confirm" votes on a false obstacle, immediately flipping the segment status to `verified` and distorting pedestrian routing.
- **Remediation**:
  1. Enforce minimum account age or minimum reputation score before votes count toward consensus.
  2. Implement IP-based voting velocity limits.
