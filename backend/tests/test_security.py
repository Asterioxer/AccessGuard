import pytest
from app.core.sanitize import validate_upload, strip_exif_and_reencode, extract_exif_gps
from PIL import Image
import io

def test_unauthenticated_verification_submission_rejected(client):
    response = client.post(
        "/api/v1/verifications/submit",
        json={"observation_id": "dummy-id", "vote": "confirm"}
    )
    assert response.status_code == 401

def test_upload_sanitizer_rejects_polyglots_and_scripts():
    # PHP/shell script disguised as JPEG
    malicious_bytes = b"<?php echo 'compromise'; ?>\r\n\x00\x00"
    is_valid, mime, _ = validate_upload(malicious_bytes, "exploit.jpg", max_size_bytes=1000000)
    assert is_valid is False
    assert "mismatched" in mime or "unsupported" in mime

def test_exif_metadata_stripped_for_privacy():
    # Create an image with synthetic EXIF
    img = Image.new("RGB", (100, 100), color=(120, 120, 120))
    exif_bytes = io.BytesIO()
    # Save with some basic metadata
    img.save(exif_bytes, format="JPEG")
    raw_bytes = exif_bytes.getvalue()

    clean_bytes = strip_exif_and_reencode(raw_bytes, output_format="JPEG")
    clean_lat, clean_lon = extract_exif_gps(clean_bytes)
    
    # Must be stripped and free of any GPS tags
    assert clean_lat is None
    assert clean_lon is None
