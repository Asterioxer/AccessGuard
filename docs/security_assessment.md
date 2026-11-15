# AccessGuard Security Assessment & Threat Model

## 1. Executive Summary

As a public-benefit crowdsourced platform dealing with physical urban geography and pedestrian user media, AccessGuard treats security and privacy as fundamental non-negotiable architectural tenets.

The system incorporates defensive architectures against:
- Malicious file execution and polyglot payloads.
- Geolocation stalking and pedestrian surveillance risks.
- Facial recognition and PII leakage.
- Unauthorized state modification and voting manipulation.

---

## 2. STRIDE Threat Model

| Threat Category | Potential Attack Vector | AccessGuard Mitigation Control |
| :--- | :--- | :--- |
| **Spoofing** | Attacker impersonates an authorized verifier to distort obstacle data. | JWT Bearer tokens signed with HS256/RS256; RBAC hierarchy verified at router dependencies. |
| **Tampering** | User submits synthetic or poisoned imagery to block accessible routes maliciously. | Upload validation with magic byte verification; multi-party consensus threshold required before routing impact. |
| **Repudiation** | User denies having submitted false obstacle flags or invalid votes. | Immutable audit log in `verification_reviews` with `reviewer_id`, timestamps, and action types. |
| **Information Disclosure** | Contributor home address or personal device serials exposed via EXIF metadata. | Strict privacy pipeline: GPS extracted in memory, followed by 100% EXIF stripping and re-encoding. |
| **Denial of Service** | Attacker floods upload endpoint with large video files or zip bombs. | File size limit (20MB image, 100MB video); duration capping (60s max); frame sampling throttles. |
| **Elevation of Privilege** | Contributor role attempts to invoke admin recalculation or delete accounts. | Granular `require_role(min_role)` dependency enforcing hierarchical authorization levels. |

---

## 3. Privacy Preservation & Zero Facial Recognition Policy

### 3.1. Zero Facial Recognition Guarantee
- AccessGuard explicitly **prohibits** facial recognition algorithms, feature embeddings, or demographic profiling.
- Media preprocessing runs aggressive OpenCV face and license plate detection solely for the purpose of **redaction and heavy Gaussian blurring** before media is saved to disk.

### 3.2. EXIF Data Scrubbing
- Street photos captured on mobile devices contain sensitive Exchangeable Image File (EXIF) metadata, including exact GPS coordinates, altitude, device serial numbers, and camera timestamps.
- The pipeline parses latitude/longitude into application memory, immediately sanitizes the raw pixel buffer, and strips all EXIF tags (`strip_exif_and_reencode`).

### 3.3. Ephemeral Retention Policy
- Original uploaded raw media buffers are never retained permanently in unredacted state.
- Retention settings (`MEDIA_RETENTION_HOURS`) allow automated pruning of raw files after obstacle confirmation.

---

## 4. OWASP API Security Top 10 Mitigation Matrix

1. **API1: Broken Object Level Authorization (BOLA)**: Observation and review updates verify user identity and check ownership/role prior to mutation.
2. **API2: Broken Authentication**: Password hashing using 12-round salted Bcrypt; expiration on JWT access tokens; constant-time string comparisons.
3. **API3: Broken Object Property Level Authorization**: Pydantic input schemas strictly filter writable fields.
4. **API4: Unrestricted Resource Consumption**: Strict payload size limitations, video duration checks, and pagination limits (`limit=50`, max `200`).
5. **API5: Broken Function Level Authorization**: Admin recalculations and verifier actions protected by `require_role("verifier")` and `require_role("admin")`.
6. **API6: Unrestricted Access to Sensitive Business Flows**: Consensus voting rules prevent duplicate votes on the same observation by the same user ID.
7. **API7: Server Side Request Forgery (SSRF)**: The application does not fetch external user-supplied URLs.
8. **API8: Security Misconfiguration**: Strict CORS configuration, production Docker containers running under unprivileged user (`appuser`, UID 1000).
9. **API9: Improper Inventory Management**: Versioned REST API (`/api/v1`) with automated OpenAPI specification generation.
10. **API10: Unsafe Consumption of APIs**: All database interactions use parameterized SQLAlchemy ORM statements, preventing SQL injection.
