import sys
import os
import numpy as np

# Ensure backend is on python path
sys.path.insert(0, os.path.abspath("backend"))

from app.cv.adapters.guidecane_adapter import GuideCaneAdapter
from app.cv.adapters.citysurfaces_adapter import CitySurfacesAdapter
from app.cv.adapters.rampnet_adapter import RampNetAdapter
from app.cv.adapters.sidewalk_validator_adapter import SidewalkValidatorAdapter

dummy = np.zeros((480, 640, 3), dtype=np.uint8)

print("--- Testing Adapters ---")
# 1. RampNet: has genuine weights!
r = RampNetAdapter(device='cpu')
res_r = r.predict_v2(dummy)
print(f"RampNet status: {res_r.status}, exec_type: {res_r.execution_type}, keypoints: {len(res_r.keypoints)}, sha: {res_r.weight_sha256[:10] if res_r.weight_sha256 else None}")

# 2. GuideCane: no weights, strict allow_fallback=False
g = GuideCaneAdapter()
res_g = g.predict_v2(dummy, allow_fallback=False)
print(f"GuideCane strict status: {res_g.status}, error: {res_g.error_message}")

# 3. GuideCane: with allow_fallback=True
g2 = GuideCaneAdapter()
res_g2 = g2.predict_v2(dummy, allow_fallback=True)
print(f"GuideCane fallback status: {res_g2.status}, exec_type: {res_g2.execution_type}, detections: {len(res_g2.detections)}")

# 4. CitySurfaces: no weights, strict allow_fallback=False
c = CitySurfacesAdapter()
res_c = c.predict_v2(dummy, allow_fallback=False)
print(f"CitySurfaces strict status: {res_c.status}, error: {res_c.error_message}")

# 5. SidewalkValidator: no DINO weights, strict allow_fallback=False
v = SidewalkValidatorAdapter()
res_v = v.predict_v2(dummy, allow_fallback=False)
print(f"Validator strict status: {res_v.status}, error: {res_v.error_message}")

print("All adapter checks completed successfully!")
