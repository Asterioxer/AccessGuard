import sys
from pathlib import Path
import time
import tracemalloc
import concurrent.futures
import json
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.cv.pipeline import CVPipeline
from app.cv.adapters.heuristic_adapter import HeuristicCVAdapter
from app.cv.adapters.dummy_adapter import DummyAdapter
from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token

def generate_test_image_bytes():
    w, h = 640, 480
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    for y in [240, 280, 320, 360, 400]:
        cv2.line(img, (100, y), (540, y), (40, 40, 40), 4)
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()

def generate_synthetic_test_video(path: str, duration_sec: int = 5, fps: int = 30):
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    total_frames = duration_sec * fps
    
    for i in range(total_frames):
        frame = np.full((h, w, 3), 160, dtype=np.uint8)
        # Moving horizontal steps
        offset = (i * 2) % 40
        for y in range(200 + offset, 440, 40):
            cv2.line(frame, (100, y), (540, y), (30, 30, 30), 4)
        out.write(frame)
    out.release()

def audit_performance():
    print("=== Phase 5: Reliability and Performance Benchmark ===")
    results = {}

    # 1. Inference Latency Benchmark
    print("\n1. Measuring Model Adapter Inference Latency...")
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    for y in [240, 280, 320, 360, 400]:
        cv2.line(img, (100, y), (540, y), (40, 40, 40), 4)

    heuristic = HeuristicCVAdapter()
    dummy = DummyAdapter()

    # Warmup
    heuristic.predict(img)
    dummy.predict(img)

    # Measure Heuristic
    heuristic_times = []
    for _ in range(25):
        t0 = time.perf_counter()
        heuristic.predict(img)
        t1 = time.perf_counter()
        heuristic_times.append((t1 - t0) * 1000.0) # ms

    # Measure Dummy
    dummy_times = []
    for _ in range(25):
        t0 = time.perf_counter()
        dummy.predict(img)
        t1 = time.perf_counter()
        dummy_times.append((t1 - t0) * 1000.0) # ms

    results["inference_latency"] = {
        "heuristic_cv_adapter": {
            "mean_ms": round(float(np.mean(heuristic_times)), 2),
            "median_ms": round(float(np.median(heuristic_times)), 2),
            "p95_ms": round(float(np.percentile(heuristic_times, 95)), 2),
            "min_ms": round(float(np.min(heuristic_times)), 2),
            "max_ms": round(float(np.max(heuristic_times)), 2)
        },
        "dummy_adapter": {
            "mean_ms": round(float(np.mean(dummy_times)), 2),
            "median_ms": round(float(np.median(dummy_times)), 2),
            "p95_ms": round(float(np.percentile(dummy_times, 95)), 2)
        }
    }
    print(f"   Heuristic Adapter: Mean {results['inference_latency']['heuristic_cv_adapter']['mean_ms']} ms (P95: {results['inference_latency']['heuristic_cv_adapter']['p95_ms']} ms)")
    print(f"   Dummy Adapter: Mean {results['inference_latency']['dummy_adapter']['mean_ms']} ms")

    # 2. Video Processing Latency & Memory Benchmark
    print("\n2. Measuring Video Frame Processing & Peak Memory...")
    temp_video_path = Path(__file__).parent / "test_benchmark_video.mp4"
    generate_synthetic_test_video(str(temp_video_path), duration_sec=5, fps=30) # 150 frames

    pipeline = CVPipeline(adapter_name="heuristic")

    tracemalloc.start()
    t_start = time.perf_counter()
    video_barriers = pipeline.process_video(str(temp_video_path), sample_fps=1.0)
    t_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    video_duration_ms = (t_end - t_start) * 1000.0
    results["video_benchmark"] = {
        "video_duration_sec": 5.0,
        "video_total_frames": 150,
        "processing_time_total_ms": round(video_duration_ms, 2),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "barriers_detected": len(video_barriers)
    }
    print(f"   5s Video (150 frames) processed in: {video_duration_ms:.1f} ms")
    print(f"   Peak memory allocation: {results['video_benchmark']['peak_memory_mb']} MB")

    if temp_video_path.exists():
        temp_video_path.unlink()

    # 3. Concurrent Uploads Stress Test
    print("\n3. Testing Concurrent Uploads Throughput (15 parallel requests)...")
    token = create_access_token(data={"sub": "benchmarker", "role": "contributor"})
    img_bytes = generate_test_image_bytes()

    client = TestClient(app)

    def upload_worker(idx):
        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/observations/upload",
            files={"file": (f"bench_{idx}.jpg", img_bytes, "image/jpeg")},
            data={"latitude": "37.7765", "longitude": "-122.4194", "adapter": "heuristic"},
            headers={"Authorization": f"Bearer {token}"}
        )
        t1 = time.perf_counter()
        return resp.status_code, (t1 - t0) * 1000.0

    upload_latencies = []
    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_worker, i) for i in range(15)]
        for f in concurrent.futures.as_completed(futures):
            status_code, latency = f.result()
            upload_latencies.append(latency)
            if status_code == 201:
                success_count += 1

    results["concurrent_uploads"] = {
        "total_requests": 15,
        "successful_requests": success_count,
        "success_rate_pct": round((success_count / 15.0) * 100.0, 1),
        "mean_latency_ms": round(float(np.mean(upload_latencies)), 2),
        "median_latency_ms": round(float(np.median(upload_latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(upload_latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(upload_latencies, 99)), 2)
    }
    print(f"   Concurrent uploads: {success_count}/15 succeeded")
    print(f"   Mean latency: {results['concurrent_uploads']['mean_latency_ms']} ms (P95: {results['concurrent_uploads']['p95_latency_ms']} ms)")

    out_path = Path(__file__).parent / "performance_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPerformance Benchmark complete. Results saved to {out_path}")
    return results

if __name__ == "__main__":
    audit_performance()
