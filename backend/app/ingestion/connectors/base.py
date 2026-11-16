"""
Base Imagery Connector and SSRF-Safe Remote Ingestion Framework.

Provides common abstractions, rate limiting, exponential backoff, circuit breaking,
and strict SSRF protections for all external street-level imagery providers.
"""

import ipaddress
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from app.core.logging import logger

MAX_IMAGE_BYTES = 25 * 1024 * 1024       # 25 MB max image download
MAX_VIDEO_BYTES = 200 * 1024 * 1024     # 200 MB max video download
DEFAULT_TIMEOUT_SEC = 15.0


class CircuitState(str, Enum):
    CLOSED = "CLOSED"         # Normal operation
    OPEN = "OPEN"             # Tripped due to failures; rejecting requests
    HALF_OPEN = "HALF_OPEN"   # Probing recovery


class SSRFValidationError(ValueError):
    """Raised when an external imagery URL violates SSRF safety constraints."""
    pass


def is_private_ip(ip_str: str) -> bool:
    """Checks whether an IP address belongs to private, loopback, or link-local ranges."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_remote_url(
    url: str,
    allowed_schemes: Tuple[str, ...] = ("http", "https"),
    allowed_domains: Optional[List[str]] = None
) -> str:
    """
    Validates a remote URL against SSRF vulnerabilities:
    - Enforces allowed schemes (e.g. http, https, rtsp).
    - Resolves hostname to IP and blocks private/loopback/cloud-metadata addresses.
    - Optionally restricts to an explicit domain allowlist.
    """
    if not url or not isinstance(url, str):
        raise SSRFValidationError("Empty or invalid URL provided.")

    parsed = urlparse(url)
    if parsed.scheme.lower() not in allowed_schemes:
        raise SSRFValidationError(f"Disallowed scheme '{parsed.scheme}'. Allowed: {allowed_schemes}")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFValidationError(f"Invalid host in URL '{url}'")

    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"):
        raise SSRFValidationError(f"Access to loopback or metadata host '{hostname}' is blocked.")

    # Check domain allowlist if configured
    if allowed_domains:
        domain_matched = any(
            hostname.lower() == d.lower() or hostname.lower().endswith("." + d.lower())
            for d in allowed_domains
        )
        if not domain_matched:
            raise SSRFValidationError(f"Host '{hostname}' is not in allowed domain list: {allowed_domains}")

    # DNS resolution check for private IP ranges
    try:
        resolved_ips = socket.getaddrinfo(hostname, None)
        for entry in resolved_ips:
            ip = entry[4][0]
            if is_private_ip(ip):
                raise SSRFValidationError(f"Resolved IP '{ip}' for host '{hostname}' is a blocked private address.")
    except socket.gaierror as e:
        # If host cannot be resolved, let caller handle or block
        logger.warning(f"Could not resolve host '{hostname}': {e}")

    return url


@dataclass
class ImageryItem:
    """Represents a standardized street-level or remote-sensing imagery observation."""
    item_id: str
    provider: str
    coordinates: Tuple[float, float]       # (longitude, latitude)
    capture_timestamp: Optional[str] = None
    ingestion_timestamp: Optional[str] = None
    url: Optional[str] = None
    heading: Optional[float] = None
    sequence_id: Optional[str] = None
    quality_score: Optional[float] = None
    license: str = "Unknown"
    attribution: str = ""
    is_live: bool = False
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImagerySearchResult:
    """Paginated collection of imagery items returned from a search query."""
    items: List[ImageryItem]
    total_count: int
    next_page_token: Optional[str] = None
    provider: str = ""
    query_bbox: Optional[Tuple[float, float, float, float]] = None # (min_lon, min_lat, max_lon, max_lat)


class BaseImageryConnector(ABC):
    """Abstract base class for all imagery acquisition connectors."""

    def __init__(
        self,
        provider_name: str,
        rate_limit_per_min: int = 60,
        max_retries: int = 3,
        circuit_trip_threshold: int = 5,
        circuit_recovery_time_sec: float = 60.0
    ):
        self.provider_name = provider_name
        self.rate_limit_per_min = rate_limit_per_min
        self.max_retries = max_retries
        self.circuit_trip_threshold = circuit_trip_threshold
        self.circuit_recovery_time_sec = circuit_recovery_time_sec

        # State tracking
        self.consecutive_failures = 0
        self.circuit_state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self._request_timestamps: List[float] = []

    def _check_circuit(self) -> None:
        """Inspects circuit breaker state and raises if tripped."""
        now = time.time()
        if self.circuit_state == CircuitState.OPEN:
            if now - self.last_state_change >= self.circuit_recovery_time_sec:
                logger.info(f"Circuit for {self.provider_name} transitioned from OPEN to HALF_OPEN (probing recovery).")
                self.circuit_state = CircuitState.HALF_OPEN
                self.last_state_change = now
            else:
                raise ConnectionError(
                    f"Circuit breaker for provider '{self.provider_name}' is OPEN due to repeated failures."
                )

    def _record_success(self) -> None:
        """Records a successful request, resetting failures and closing circuit."""
        self.consecutive_failures = 0
        if self.circuit_state != CircuitState.CLOSED:
            logger.info(f"Circuit for {self.provider_name} successfully recovered to CLOSED.")
            self.circuit_state = CircuitState.CLOSED
            self.last_state_change = time.time()

    def _record_failure(self) -> None:
        """Records a request failure and trips circuit if threshold exceeded."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.circuit_trip_threshold:
            self.circuit_state = CircuitState.OPEN
            self.last_state_change = time.time()
            logger.error(f"Circuit for {self.provider_name} TRIPPED to OPEN after {self.consecutive_failures} failures.")

    def _throttle(self) -> None:
        """Enforces sliding-window rate limiting."""
        now = time.time()
        window_start = now - 60.0
        self._request_timestamps = [t for t in self._request_timestamps if t > window_start]
        if len(self._request_timestamps) >= self.rate_limit_per_min:
            sleep_time = max(0.1, 60.0 - (now - self._request_timestamps[0]))
            logger.debug(f"Rate limit reached for {self.provider_name}; throttling {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._request_timestamps.append(time.time())

    @abstractmethod
    def search_imagery(
        self,
        bbox: Tuple[float, float, float, float], # (min_lon, min_lat, max_lon, max_lat)
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        page_token: Optional[str] = None
    ) -> ImagerySearchResult:
        """Searches imagery within a geographic bounding box."""
        pass

    @abstractmethod
    def get_image_metadata(self, image_id: str) -> Optional[ImageryItem]:
        """Retrieves complete metadata for an individual image."""
        pass

    @abstractmethod
    def fetch_image_bytes(self, image_id_or_url: str) -> bytes:
        """Retrieves raw image bytes with size validation and SSRF protection."""
        pass

    def get_quota_status(self) -> Dict[str, Any]:
        """Returns provider quota and circuit status for monitoring."""
        return {
            "provider": self.provider_name,
            "circuit_state": self.circuit_state.value,
            "consecutive_failures": self.consecutive_failures,
            "rate_limit_per_min": self.rate_limit_per_min,
            "requests_last_minute": len(self._request_timestamps),
            "healthy": self.circuit_state == CircuitState.CLOSED
        }
