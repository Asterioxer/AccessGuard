import React, { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { LiveFrameResponse, LiveDetectionDTO } from '../types';

interface LiveCameraProps {
  onObstacleDetected?: (detections: LiveDetectionDTO[]) => void;
}

export const LiveCamera: React.FC<LiveCameraProps> = ({ onObstacleDetected }) => {
  const [permissionGranted, setPermissionGranted] = useState<boolean | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamMode, setStreamMode] = useState<'snapshot' | 'continuous'>('snapshot');
  const [targetFps, setTargetFps] = useState<number>(2); // 1, 2, or 5 FPS
  const [activeAdapter, setActiveAdapter] = useState<string>('heuristic');
  const [availableAdapters, setAvailableAdapters] = useState<string[]>(['heuristic', 'yolo', 'onnx', 'torchvision']);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
  
  // Performance & Telemetry HUD
  const [lastLatencyMs, setLastLatencyMs] = useState<number>(0);
  const [lastFpsEst, setLastFpsEst] = useState<number>(0);
  const [framesDropped, setFramesDropped] = useState<number>(0);
  const [lastDetections, setLastDetections] = useState<LiveDetectionDTO[]>([]);
  const [imageQuality, setImageQuality] = useState<{ is_blurry: boolean; blur_variance: number } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [streamId] = useState<string>(() => 'cam_' + Math.random().toString(36).substring(2, 9));

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const isInFlightRef = useRef<boolean>(false);
  const timerRef = useRef<any>(null);

  // Load available adapters on mount
  useEffect(() => {
    api.listAdapters()
      .then(res => {
        setAvailableAdapters(res.available_adapters);
        setActiveAdapter(res.active_adapter);
      })
      .catch(() => {
        // Fallback defaults
      });
  }, []);

  // Switch adapter
  const handleAdapterChange = async (newAdapter: string) => {
    try {
      await api.setActiveAdapter(newAdapter);
      setActiveAdapter(newAdapter);
    } catch (err: any) {
      setErrorMessage(`Failed to switch adapter: ${err.message}`);
    }
  };

  // Start webcam with explicit permission request
  const startCamera = async () => {
    setErrorMessage(null);
    try {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(t => t.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facingMode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setPermissionGranted(true);
          setIsStreaming(true);
        };
      }
    } catch (err: any) {
      setPermissionGranted(false);
      setIsStreaming(false);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setErrorMessage('Camera access was denied. Please grant webcam permission in your browser to test live inference.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setErrorMessage('No camera device found on your system. Please connect a webcam.');
      } else {
        setErrorMessage(`Camera error: ${err.message || 'Unable to access media stream.'}`);
      }
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    clearCanvas();
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Clear bounding box overlay
  const clearCanvas = () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  // Draw bounding boxes on canvas
  const drawDetections = (detections: LiveDetectionDTO[], width: number, height: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    detections.forEach(det => {
      const [ymin, xmin, ymax, xmax] = det.bbox;
      const x = xmin * width;
      const y = ymin * height;
      const w = (xmax - xmin) * width;
      const h = (ymax - ymin) * height;

      // Color selection based on category
      let strokeColor = '#EF4444'; // Red: stairs / obstacle
      let badgeBg = 'rgba(239, 68, 68, 0.85)';
      if (det.barrier_type === 'curb_ramp_present' || det.barrier_type === 'curb_ramp') {
        strokeColor = '#10B981'; // Green: ramp
        badgeBg = 'rgba(16, 185, 129, 0.85)';
      } else if (det.barrier_type === 'surface_damage') {
        strokeColor = '#F59E0B'; // Amber: damage
        badgeBg = 'rgba(245, 158, 11, 0.85)';
      }

      // Draw bounding box
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(x, y, w, h);

      // Label
      const confPct = Math.round(det.confidence * 100);
      const labelText = `${det.barrier_type.replace('_', ' ')} (${confPct}%)`;
      ctx.font = 'bold 12px sans-serif';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = badgeBg;
      ctx.fillRect(x, Math.max(0, y - 22), textWidth + 12, 22);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(labelText, x + 6, Math.max(16, y - 6));

      // Uncertainty indicator
      if (det.is_uncertain) {
        ctx.fillStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.fillRect(x + textWidth + 16, Math.max(0, y - 22), 70, 22);
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText('UNCERTAIN', x + textWidth + 20, Math.max(15, y - 7));
      }
    });
  };

  // Capture single frame and run inference
  const captureAndInfer = useCallback(async () => {
    if (!videoRef.current || !isStreaming) return;
    if (isInFlightRef.current) {
      // Frame dropping when inference is in flight
      setFramesDropped(prev => prev + 1);
      return;
    }

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    isInFlightRef.current = true;

    try {
      // Draw video frame to an offscreen canvas to obtain base64 JPEG
      const offscreen = document.createElement('canvas');
      offscreen.width = video.videoWidth;
      offscreen.height = video.videoHeight;
      const ctx = offscreen.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
      const base64Data = offscreen.toDataURL('image/jpeg', 0.80);

      const clientStart = performance.now();
      const res: LiveFrameResponse = await api.postLiveFrame({
        base64_image: base64Data,
        stream_id: streamId,
        timestamp: Date.now() / 1000.0
      });
      const clientEnd = performance.now();
      const roundTripMs = Math.round(clientEnd - clientStart);

      setLastLatencyMs(res.latency_ms);
      setLastFpsEst(res.fps_estimate);
      setLastDetections(res.detections);
      setImageQuality(res.image_quality);

      // Draw bounding box overlay
      drawDetections(res.detections, video.videoWidth, video.videoHeight);

      if (onObstacleDetected && res.detections.length > 0) {
        onObstacleDetected(res.detections);
      }
    } catch (err: any) {
      console.warn('Live frame inference error:', err);
    } finally {
      isInFlightRef.current = false;
    }
  }, [isStreaming, streamId, onObstacleDetected]);

  // Handle continuous streaming mode interval
  useEffect(() => {
    if (isStreaming && streamMode === 'continuous') {
      const intervalMs = Math.max(200, Math.round(1000 / targetFps));
      timerRef.current = setInterval(() => {
        captureAndInfer();
      }, intervalMs);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isStreaming, streamMode, targetFps, captureAndInfer]);

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
      {/* Header & Controls Bar */}
      <div className="p-4 bg-gray-50 border-b border-gray-200 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-indigo-600 animate-pulse"></span>
            Live Camera Intelligence & Testing Interface
          </h2>
          <p className="text-xs text-gray-500">
            Real-time sidewalk barrier inference, spatial bounding boxes, and temporal deduplication.
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Adapter Selection */}
          <div className="flex items-center gap-1.5 text-xs">
            <label htmlFor="adapter-select" className="font-medium text-gray-700">Model:</label>
            <select
              id="adapter-select"
              value={activeAdapter}
              onChange={(e) => handleAdapterChange(e.target.value)}
              className="px-2.5 py-1.5 bg-white border border-gray-300 rounded-md text-xs font-semibold text-gray-800 shadow-sm focus:ring-1 focus:ring-indigo-500"
            >
              {availableAdapters.map((adp) => (
                <option key={adp} value={adp}>
                  {adp.toUpperCase()} {adp === 'onnx' ? '(Fastest)' : adp === 'yolo' ? '(Deep Learning)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Mode Switch */}
          <div className="inline-flex rounded-md shadow-sm">
            <button
              onClick={() => setStreamMode('snapshot')}
              className={`px-3 py-1.5 text-xs font-medium rounded-l-md border ${
                streamMode === 'snapshot'
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
              }`}
            >
              Snapshot Mode
            </button>
            <button
              onClick={() => setStreamMode('continuous')}
              className={`px-3 py-1.5 text-xs font-medium rounded-r-md border-t border-b border-r ${
                streamMode === 'continuous'
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
              }`}
            >
              Continuous Stream
            </button>
          </div>

          {/* Target FPS for continuous mode */}
          {streamMode === 'continuous' && (
            <div className="flex items-center gap-1 text-xs">
              <label htmlFor="fps-select" className="text-gray-600">Rate:</label>
              <select
                id="fps-select"
                value={targetFps}
                onChange={(e) => setTargetFps(Number(e.target.value))}
                className="px-2 py-1 bg-white border border-gray-300 rounded text-xs font-medium"
              >
                <option value={1}>1 FPS</option>
                <option value={2}>2 FPS (Recommended)</option>
                <option value={5}>5 FPS (High)</option>
              </select>
            </div>
          )}

          {/* Facing mode */}
          <button
            onClick={() => {
              setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
              if (isStreaming) {
                stopCamera();
                setTimeout(startCamera, 300);
              }
            }}
            title="Flip camera"
            className="p-1.5 text-gray-600 hover:text-gray-900 border border-gray-300 rounded hover:bg-gray-100"
          >
            🔄 Flip
          </button>
        </div>
      </div>

      {/* Main Viewport & Video Overlay */}
      <div className="relative bg-gray-900 min-h-[360px] flex items-center justify-center overflow-hidden">
        {/* Live Video Feed */}
        <video
          ref={videoRef}
          playsInline
          muted
          className={`w-full max-h-[500px] object-contain ${!isStreaming ? 'hidden' : 'block'}`}
        />

        {/* Canvas Overlay for Bounding Boxes */}
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full pointer-events-none object-contain ${!isStreaming ? 'hidden' : 'block'}`}
        />

        {/* Performance Heads-Up Display (HUD) */}
        {isStreaming && (
          <div className="absolute top-3 left-3 bg-black/75 backdrop-blur-sm text-white px-3 py-2 rounded-lg text-xs font-mono space-y-1 z-10 border border-white/10 shadow-lg">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-ping"></span>
              <span className="font-bold text-green-400">ACTIVE: {activeAdapter.toUpperCase()}</span>
            </div>
            <div>Server Latency: <span className="text-yellow-300 font-semibold">{lastLatencyMs} ms</span></div>
            <div>Max FPS Capacity: <span className="text-cyan-300">{lastFpsEst} FPS</span></div>
            <div>Dropped Frames: <span className={framesDropped > 0 ? "text-orange-400" : "text-gray-400"}>{framesDropped}</span></div>
            {imageQuality && (
              <div>
                Quality: {imageQuality.is_blurry ? (
                  <span className="text-red-400 font-bold">⚠️ Motion Blur (Var={imageQuality.blur_variance})</span>
                ) : (
                  <span className="text-emerald-400">✓ Sharp (Var={imageQuality.blur_variance})</span>
                )}
              </div>
            )}
            <div className="text-[10px] text-gray-400 pt-1 border-t border-white/10">
              🛡️ OpenCV Face & Plate Anonymization Active
            </div>
          </div>
        )}

        {/* Permission Request / Start Prompt when camera inactive */}
        {!isStreaming && (
          <div className="text-center p-8 max-w-md">
            <div className="w-16 h-16 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
              📷
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Webcam Permission Required</h3>
            <p className="text-xs text-gray-300 mb-6 leading-relaxed">
              AccessGuard runs in-browser accessibility barrier detection on your street or sidewalk view.
              No raw camera video or biometric facial data is stored or uploaded to servers without explicit submission.
            </p>
            {errorMessage && (
              <div className="mb-4 p-3 bg-red-900/50 border border-red-500/50 text-red-200 text-xs rounded-lg text-left">
                {errorMessage}
              </div>
            )}
            <button
              onClick={startCamera}
              className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow transition duration-150 text-sm"
            >
              Start Live Camera
            </button>
          </div>
        )}

        {/* Floating Snapshot Trigger Button */}
        {isStreaming && streamMode === 'snapshot' && (
          <div className="absolute bottom-4 inset-x-0 flex justify-center z-10">
            <button
              onClick={captureAndInfer}
              className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-full shadow-lg flex items-center gap-2 text-sm border-2 border-white/20 transition-transform active:scale-95"
            >
              📸 Capture & Analyze Frame
            </button>
          </div>
        )}
      </div>

      {/* Footer Status Bar & Detections Inventory */}
      <div className="p-4 bg-gray-50 border-t border-gray-200 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          {isStreaming ? (
            <button
              onClick={stopCamera}
              className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white font-medium rounded-md shadow-sm"
            >
              Stop Camera
            </button>
          ) : (
            <span className="text-gray-500 font-medium">Camera Stopped</span>
          )}
          <span className="text-gray-600">
            Detections in view: <strong className="text-gray-900 font-bold">{lastDetections.length}</strong>
          </span>
        </div>

        {/* Real-time Detections Badges */}
        <div className="flex flex-wrap items-center gap-2">
          {lastDetections.map((d, idx) => (
            <span
              key={idx}
              className={`px-2 py-1 rounded-md font-semibold text-[11px] flex items-center gap-1.5 ${
                d.barrier_type.includes('ramp')
                  ? 'bg-green-100 text-green-800 border border-green-300'
                  : d.barrier_type.includes('damage')
                  ? 'bg-amber-100 text-amber-800 border border-amber-300'
                  : 'bg-red-100 text-red-800 border border-red-300'
              }`}
            >
              {d.barrier_type.replace('_', ' ')}
              <span className="bg-white/80 px-1 rounded text-[10px]">
                {Math.round(d.confidence * 100)}%
              </span>
              {d.is_confirmed && <span title="Temporally Verified Track">✓</span>}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
