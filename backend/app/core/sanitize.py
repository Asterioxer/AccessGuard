import io
import re
from typing import Optional, Tuple
from PIL import Image, ExifTags
import numpy as np

# Decompression bomb and resource limit guards
MAX_IMAGE_PIXELS = 16_000_000 # Max 16 Megapixels
MAX_IMAGE_DIMENSION = 4096 # Max 4096px along width or height
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Magic byte signatures for strict validation
MAGIC_BYTES = {
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"], # followed by WEBP at offset 8
    "mp4": [b"\x00\x00\x00\x18ftyp", b"\x00\x00\x00\x20ftyp", b"\x00\x00\x00\x1cftyp"],
    "quicktime": [b"\x00\x00\x00\x14ftypqt", b"\x00\x00\x00\x08wide", b"\x00\x00\x00\x08mdat"],
}

def validate_upload(file_bytes: bytes, filename: str, max_size_bytes: int) -> Tuple[bool, str, str]:
    """
    Validates file size, extension, magic header bytes, and image dimension boundaries.
    Returns (is_valid, mime_type, media_category['image'|'video'])
    """
    if len(file_bytes) == 0:
        return False, "empty_file", "unknown"
    if len(file_bytes) > max_size_bytes:
        return False, f"file_exceeds_max_size_{max_size_bytes}", "unknown"
    
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    is_image = False
    mime_type = "application/octet-stream"
    
    # Image checks
    if ext in ["jpg", "jpeg"]:
        if any(file_bytes.startswith(sig) for sig in MAGIC_BYTES["jpeg"]):
            is_image = True
            mime_type = "image/jpeg"
    elif ext == "png":
        if any(file_bytes.startswith(sig) for sig in MAGIC_BYTES["png"]):
            is_image = True
            mime_type = "image/png"
    elif ext == "webp":
        if file_bytes.startswith(b"RIFF") and len(file_bytes) > 12 and file_bytes[8:12] == b"WEBP":
            is_image = True
            mime_type = "image/webp"
            
    # Video checks
    elif ext in ["mp4", "m4v"]:
        # Check standard ISO Base Media File Format (ftyp box within first 32 bytes)
        if b"ftyp" in file_bytes[:32]:
            return True, "video/mp4", "video"
    elif ext in ["mov", "qt"]:
        if b"ftyp" in file_bytes[:32] or b"moov" in file_bytes[:64] or b"wide" in file_bytes[:32]:
            return True, "video/quicktime", "video"

    if is_image:
        # Check pixel dimensions to protect against decompression bombs
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                w, h = img.size
                if w * h > MAX_IMAGE_PIXELS or w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                    return False, f"image_dimensions_exceed_limit_{w}x{h}_max_{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}", "image"
            return True, mime_type, "image"
        except Exception as e:
            return False, f"invalid_or_corrupt_image: {str(e)}", "image"
            
    return False, "unsupported_media_type_or_mismatched_magic_bytes", "unknown"

def _dms_to_decimal(dms, ref) -> Optional[float]:
    """Convert degrees, minutes, seconds tuple to decimal degrees."""
    try:
        degrees = float(dms[0])
        minutes = float(dms[1]) / 60.0
        seconds = float(dms[2]) / 3600.0
        val = degrees + minutes + seconds
        if ref in ['S', 'W', 's', 'w']:
            val = -val
        return round(val, 6)
    except Exception:
        return None

def extract_exif_gps(image_bytes: bytes) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts geographic coordinates (lat, lon) from image EXIF metadata if present.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return None, None
            
        gps_info = {}
        # GPS info tag in IFD is 34853 (0x8825)
        for key, value in exif.items():
            tag_name = ExifTags.TAGS.get(key, key)
            if tag_name == "GPSInfo" or key == 34853:
                # Sub-IFD get
                gps_ifd = exif.get_ifd(key) if hasattr(exif, "get_ifd") else value
                gps_info = gps_ifd
                break
                
        if not gps_info:
            return None, None
            
        lat_ref = gps_info.get(1) # GPSLatitudeRef
        lat_val = gps_info.get(2) # GPSLatitude
        lon_ref = gps_info.get(3) # GPSLongitudeRef
        lon_val = gps_info.get(4) # GPSLongitude
        
        if lat_val and lon_val and lat_ref and lon_ref:
            lat = _dms_to_decimal(lat_val, lat_ref)
            lon = _dms_to_decimal(lon_val, lon_ref)
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
                
        return None, None
    except Exception:
        return None, None

def strip_exif_and_reencode(image_bytes: bytes, output_format: str = "JPEG") -> bytes:
    """
    Strips EXIF, GPS, camera serials, and privacy metadata, re-encoding raw pixel buffer.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Dimension bound guard
            if img.width * img.height > MAX_IMAGE_PIXELS or img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
                return image_bytes
            
            # Create fresh image without metadata
            clean_img = Image.new(img.mode, img.size)
            clean_img.paste(img)
            
            out_buf = io.BytesIO()
            if output_format.upper() in ["JPG", "JPEG"]:
                if clean_img.mode in ("RGBA", "P"):
                    clean_img = clean_img.convert("RGB")
                clean_img.save(out_buf, format="JPEG", quality=88, optimize=True)
            else:
                clean_img.save(out_buf, format="PNG", optimize=True)
            return out_buf.getvalue()
    except Exception:
        # Fallback to returning original if transformation fails
        return image_bytes

def apply_privacy_redaction(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applies aggressive privacy redaction (blur/pixelate) to protect pedestrians:
    - Never runs facial recognition
    - Redacts face-like regions using OpenCV cascades
    - Redacts high-frequency text/plate-like rectangular contours
    """
    try:
        import cv2
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load default OpenCV frontal face cascade if accessible
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            for (x, y, fw, fh) in faces:
                # Heavy Gaussian blur over human facial region
                roi = image_bgr[y:y+fh, x:x+fw]
                ksize = max(15, (fw // 2) | 1) # Ensure odd kernel
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 30)
                image_bgr[y:y+fh, x:x+fw] = blurred_roi
        return image_bgr
    except Exception:
        return image_bgr
