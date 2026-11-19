import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import {
  IngestionProviderInfo,
  IngestionJob,
  SegmentedRegion,
  VideoJobProgress,
  RealityScorecard,
} from '../types';

export const AutonomousDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'ingestion' | 'segmentation' | 'reality' | 'provenance'>('ingestion');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ingestion State
  const [providers, setProviders] = useState<IngestionProviderInfo[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [newJobProvider, setNewJobProvider] = useState('mapillary');
  const [bboxInput, setBboxInput] = useState('');
  const [intervalMin, setIntervalMin] = useState(60);
  const [priority, setPriority] = useState('normal');
  const [osmImportSummary, setOsmImportSummary] = useState<string | null>(null);

  // Segmentation & Video State
  const [segRegions, setSegRegions] = useState<SegmentedRegion[]>([]);
  const [segImageSrc, setSegImageSrc] = useState<string | null>(null);
  const [showMasks, setShowMasks] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.45);
  const [videoJob, setVideoJob] = useState<VideoJobProgress | null>(null);
  const [videoProcessing, setVideoProcessing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Reality Engine State
  const [scorecard, setScorecard] = useState<RealityScorecard | null>(null);
  const [evaluatingReality, setEvaluatingReality] = useState(false);

  useEffect(() => {
    loadProviders();
    loadJobs();
    loadScorecard();
  }, []);

  const loadProviders = async () => {
    try {
      const data = await api.getProviders();
      setProviders(data);
    } catch (e: any) {
      console.error(e);
    }
  };

  const loadJobs = async () => {
    try {
      const data = await api.getIngestionJobs();
      setJobs(data);
    } catch (e: any) {
      console.error(e);
    }
  };

  const loadScorecard = async () => {
    try {
      const data = await api.getRealityScorecard();
      setScorecard(data);
    } catch (e: any) {
      console.error(e);
    }
  };

  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);
      const coords = bboxInput.split(',').map(s => parseFloat(s.trim())) as [number, number, number, number];
      if (coords.length !== 4 || coords.some(isNaN)) {
        throw new Error('Please enter 4 valid comma-separated coordinates: min_lon, min_lat, max_lon, max_lat');
      }
      await api.createIngestionJob({
        provider: newJobProvider,
        bbox: coords,
        interval_minutes: intervalMin,
        priority,
      });
      await loadJobs();
    } catch (err: any) {
      setError(err.message || 'Failed to create job');
    } finally {
      setLoading(false);
    }
  };

  const handleImportOSM = async () => {
    try {
      setLoading(true);
      setError(null);
      setOsmImportSummary(null);
      const coords = bboxInput.split(',').map(s => parseFloat(s.trim())) as [number, number, number, number];
      if (coords.length !== 4 || coords.some(isNaN)) {
        throw new Error('Enter a bounding box: min_lon, min_lat, max_lon, max_lat');
      }
      const result = await api.importOSMPedestrianNetwork(coords);
      setOsmImportSummary(String(result.new_segments_inserted) + ' new OSM pedestrian ways imported (' + String(result.ways_returned) + ' returned). Imported ways remain unassessed until observations are collected.');
      window.dispatchEvent(new Event('accessguard:segments-updated'));
    } catch (err: any) {
      setError(err.message || 'OpenStreetMap import failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRunJob = async (jobId: string) => {
    try {
      setLoading(true);
      await api.runIngestionJob(jobId);
      await loadJobs();
    } catch (err: any) {
      setError(err.message || 'Failed to execute job');
    } finally {
      setLoading(false);
    }
  };

  const handleImageUploadForSegmentation = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setLoading(true);
      setError(null);
      const reader = new FileReader();
      reader.onload = (event) => setSegImageSrc(event.target?.result as string);
      reader.readAsDataURL(file);

      const resp = await api.predictSegmentation(file);
      setSegRegions(resp.regions || []);
    } catch (err: any) {
      setError(err.message || 'Segmentation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setVideoProcessing(true);
      setError(null);
      const resp = await api.uploadVideo(file);
      const jid = resp.job_id;

      // Poll video progress
      const interval = setInterval(async () => {
        try {
          const poll = await api.getVideoJob(jid);
          setVideoJob(poll);
          if (poll.status === 'COMPLETED' || poll.status === 'FAILED') {
            clearInterval(interval);
            setVideoProcessing(false);
          }
        } catch {
          clearInterval(interval);
          setVideoProcessing(false);
        }
      }, 1500);
    } catch (err: any) {
      setError(err.message || 'Video upload failed');
      setVideoProcessing(false);
    }
  };

  // Render canvas overlays for segmentation masks
  useEffect(() => {
    if (!canvasRef.current || !segImageSrc) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = segImageSrc;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Draw polygons
      if (showMasks) {
        segRegions.forEach(reg => {
          if (!reg.polygon || reg.polygon.length < 3) return;
          ctx.beginPath();
          const first = reg.polygon[0];
          ctx.moveTo(first[0] * img.width, first[1] * img.height);
          for (let i = 1; i < reg.polygon.length; i++) {
            const pt = reg.polygon[i];
            ctx.lineTo(pt[0] * img.width, pt[1] * img.height);
          }
          ctx.closePath();
          ctx.fillStyle = `${reg.color_hex}${Math.round(maskOpacity * 255).toString(16).padStart(2, '0')}`;
          ctx.fill();
          ctx.strokeStyle = reg.color_hex;
          ctx.lineWidth = 2;
          ctx.stroke();
        });
      }

      // Draw bounding boxes
      if (showBoxes) {
        segRegions.forEach(reg => {
          const [ymin, xmin, ymax, xmax] = reg.bbox;
          const x = xmin * img.width;
          const y = ymin * img.height;
          const w = (xmax - xmin) * img.width;
          const h = (ymax - ymin) * img.height;

          ctx.strokeStyle = reg.color_hex;
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, w, h);

          // Label
          ctx.fillStyle = reg.color_hex;
          ctx.fillRect(x, Math.max(0, y - 22), 180, 22);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 12px sans-serif';
          ctx.fillText(`${reg.class_name.replace('_', ' ')} (${Math.round(reg.confidence * 100)}%)`, x + 5, Math.max(14, y - 6));
        });
      }
    };
  }, [segImageSrc, segRegions, showMasks, showBoxes, maskOpacity]);

  const handleRunRealityTest = async () => {
    try {
      setEvaluatingReality(true);
      setError(null);
      const card = await api.triggerRealityEvaluation();
      setScorecard(card);
    } catch (err: any) {
      setError(err.message || 'Reality test evaluation failed');
    } finally {
      setEvaluatingReality(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Mode Telemetry */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 border border-slate-800 rounded-2xl p-6 text-white shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-emerald-500/20 text-emerald-300 text-xs font-semibold rounded-full border border-emerald-500/40">
                ● Acquisition requires a configured source
              </span>
              <span className="text-xs text-slate-400">Baseline v1.2.0 | Mission 005</span>
            </div>
            <h1 className="text-2xl font-bold mt-2 text-white">Autonomous Field Intelligence & Reality Engine</h1>
            <p className="text-slate-300 text-sm mt-1 max-w-2xl">
              Automated multi-source streetscape acquisition, pixel-level contour segmentation, video deduplication, and independent empirical reality verification.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1.5 bg-slate-800 border border-slate-700 text-xs rounded-lg text-slate-300">
              External imagery is only available when a provider is configured
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-800 mt-6 space-x-1">
          <button
            onClick={() => setActiveTab('ingestion')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'ingestion'
                ? 'border-indigo-400 text-indigo-300'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            Data Sources & Ingestion ({providers.length})
          </button>
          <button
            onClick={() => setActiveTab('segmentation')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'segmentation'
                ? 'border-indigo-400 text-indigo-300'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            Pixel Segmentation & Video Tracker
          </button>
          <button
            onClick={() => setActiveTab('reality')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'reality'
                ? 'border-indigo-400 text-indigo-300'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            Reality Scorecard (Mission 005)
          </button>
          <button
            onClick={() => setActiveTab('provenance')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'provenance'
                ? 'border-indigo-400 text-indigo-300'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            Geospatial Provenance & Freshness
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* TAB 1: DATA SOURCES & INGESTION */}
      {activeTab === 'ingestion' && (
        <div className="space-y-6">
          {/* Provider Cards */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {providers.map((p) => (
              <div key={p.provider} className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-white capitalize">{p.provider.replace('_', ' ')}</h3>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                        p.circuit_state === 'CLOSED' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                      }`}
                    >
                      {p.circuit_state}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-2 line-clamp-2">{p.attribution}</p>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800 text-xs text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>Limit:</span>
                    <span className="text-slate-300">{p.rate_limit_per_min}/min</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Authenticated:</span>
                    <span className={p.is_authenticated ? 'text-emerald-400' : 'text-amber-400'}>
                      {p.is_authenticated ? 'Yes (Live)' : 'No (Fixture)'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Create Ingestion Job Form */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <h2 className="text-lg font-bold text-white mb-4">Configure Autonomous Ingestion Schedule</h2>
            <form onSubmit={handleCreateJob} className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Provider</label>
                <select
                  value={newJobProvider}
                  onChange={(e) => setNewJobProvider(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                >
                  <option value="mapillary">Mapillary API v4</option>
                  <option value="kartaview">KartaView</option>
                  <option value="copernicus">Copernicus STAC</option>
                  <option value="live_stream">Authorized Live Stream</option>
                  <option value="pexels">Pexels (Supplementary)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Bounding Box (min_lon, min_lat, max_lon, max_lat)</label>
                <input
                  type="text"
                  value={bboxInput}
                  onChange={(e) => setBboxInput(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Interval (minutes)</label>
                <input
                  type="number"
                  min="5"
                  max="1440"
                  value={intervalMin}
                  onChange={(e) => setIntervalMin(parseInt(e.target.value))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                />
              </div>

              <div className="flex items-end">
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2 px-4 rounded-lg text-sm transition-colors"
                >
                  {loading ? 'Creating...' : '+ Schedule Job'}
                </button>
              </div>
            </form>
            <div className="mt-4 flex flex-col sm:flex-row sm:items-center gap-3">
              <button type="button" onClick={handleImportOSM} disabled={loading} className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white font-medium py-2 px-4 rounded-lg text-sm">
                {loading ? 'Importing...' : 'Import real OSM pedestrian ways'}
              </button>
              <p className="text-xs text-slate-400">Uses the bounding box above. Requires an administrator session and a small area (maximum 0.02 square degrees). Source: OpenStreetMap (ODbL).</p>
            </div>
            {osmImportSummary && <p className="mt-3 text-xs text-emerald-300">{osmImportSummary}</p>}
          </div>

          {/* Active Jobs Queue */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <h2 className="text-lg font-bold text-white mb-4">Acquisition Jobs & Queue</h2>
            {jobs.length === 0 ? (
              <p className="text-sm text-slate-400">No acquisition jobs configured yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-800/60 uppercase tracking-wider text-slate-400">
                    <tr>
                      <th className="p-3">Job ID</th>
                      <th className="p-3">Provider</th>
                      <th className="p-3">Status</th>
                      <th className="p-3">Priority</th>
                      <th className="p-3">Processed / Fetched</th>
                      <th className="p-3">Next Scheduled Run</th>
                      <th className="p-3">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {jobs.map((j) => (
                      <tr key={j.id} className="hover:bg-slate-800/30">
                        <td className="p-3 font-mono">{j.id.slice(0, 8)}</td>
                        <td className="p-3 capitalize font-medium text-white">{j.provider}</td>
                        <td className="p-3">
                          <span
                            className={`px-2 py-0.5 rounded-full font-semibold text-[10px] ${
                              j.status === 'completed'
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : j.status === 'running'
                                ? 'bg-indigo-500/20 text-indigo-400'
                                : 'bg-slate-700 text-slate-300'
                            }`}
                          >
                            {j.status}
                          </span>
                        </td>
                        <td className="p-3 uppercase text-[10px] font-semibold">{j.priority}</td>
                        <td className="p-3">
                          <span className="text-emerald-400 font-medium">{j.items_processed}</span> / {j.items_fetched}
                        </td>
                        <td className="p-3 text-slate-400">
                          {j.next_run_at ? new Date(j.next_run_at).toLocaleTimeString() : 'Immediate'}
                        </td>
                        <td className="p-3">
                          <button
                            onClick={() => handleRunJob(j.id)}
                            disabled={loading || j.status === 'running'}
                            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-200 hover:text-white"
                          >
                            Run Now
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: PIXEL SEGMENTATION & VIDEO TRACKER */}
      {activeTab === 'segmentation' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Image Segmentation Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-bold text-white">Pixel-Level Semantic Segmentation</h2>
              <p className="text-xs text-slate-400">
                Upload streetscape image to extract genuine non-rectangular polygon contours for traversable sidewalks, stairs, damaged pavement, and open drains.
              </p>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUploadForSegmentation}
                className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500"
              />

              {segImageSrc && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={showMasks} onChange={(e) => setShowMasks(e.target.checked)} />
                      Show Masks
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
                      Show Bounding Boxes
                    </label>
                    <div className="flex items-center gap-2">
                      <span>Opacity:</span>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.05"
                        value={maskOpacity}
                        onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                        className="w-20"
                      />
                    </div>
                  </div>

                  <div className="border border-slate-800 rounded-xl overflow-hidden bg-black flex justify-center">
                    <canvas ref={canvasRef} className="max-w-full max-h-96 object-contain" />
                  </div>

                  <div className="space-y-1">
                    <h4 className="text-xs font-semibold text-slate-400">Segmented Regions ({segRegions.length}):</h4>
                    <div className="flex flex-wrap gap-2">
                      {segRegions.map((r, i) => (
                        <span
                          key={i}
                          style={{ borderColor: r.color_hex, color: r.color_hex }}
                          className="px-2.5 py-1 text-xs rounded-lg border bg-slate-900/80 font-medium"
                        >
                          {r.class_name.replace('_', ' ')}: {Math.round(r.confidence * 100)}% ({r.polygon?.length || 0} pts)
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Video Processing & Temporal Deduplication */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-lg font-bold text-white">Video Intelligence & Temporal Deduplication</h2>
              <p className="text-xs text-slate-400">
                Upload a pedestrian walkway video to evaluate temporal barrier tracking. Multi-frame sightings of the same physical hazard are clustered into exactly 1 physical report.
              </p>
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                disabled={videoProcessing}
                className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500"
              />

              {videoProcessing && (
                <div className="p-4 bg-indigo-500/10 border border-indigo-500/30 rounded-xl space-y-2">
                  <div className="flex justify-between text-xs text-indigo-300">
                    <span>Processing video frames...</span>
                    <span>{videoJob?.progress_percent || 0}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-indigo-500 h-2 transition-all duration-300"
                      style={{ width: `${videoJob?.progress_percent || 0}%` }}
                    />
                  </div>
                </div>
              )}

              {videoJob && videoJob.status === 'COMPLETED' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-3 bg-slate-800/60 rounded-lg">
                      <div className="text-slate-400">Sampled Frames</div>
                      <div className="text-base font-bold text-white mt-1">{videoJob.sampled_frames}</div>
                    </div>
                    <div className="p-3 bg-slate-800/60 rounded-lg">
                      <div className="text-slate-400">Video Duration</div>
                      <div className="text-base font-bold text-white mt-1">{videoJob.duration_seconds}s</div>
                    </div>
                    <div className="p-3 bg-slate-800/60 rounded-lg">
                      <div className="text-slate-400">Unique Obstacles</div>
                      <div className="text-base font-bold text-emerald-400 mt-1">{videoJob.unique_obstacles_count}</div>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-800/60 text-slate-400 uppercase text-[10px]">
                        <tr>
                          <th className="p-2">Track ID</th>
                          <th className="p-2">Type</th>
                          <th className="p-2">Duration</th>
                          <th className="p-2">Sightings</th>
                          <th className="p-2">Confidence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800">
                        {videoJob.confirmed_tracks.map((t) => (
                          <tr key={t.track_id} className="hover:bg-slate-800/30">
                            <td className="p-2 font-mono">{t.track_id}</td>
                            <td className="p-2 font-medium text-white capitalize">{t.barrier_type.replace('_', ' ')}</td>
                            <td className="p-2">{t.duration_sec}s</td>
                            <td className="p-2">{t.frame_count} frames</td>
                            <td className="p-2 font-semibold text-emerald-400">{Math.round(t.max_confidence * 100)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: REALITY SCORECARD (MISSION 005 ADDENDUM) */}
      {activeTab === 'reality' && (
        <div className="space-y-6">
          {scorecard && (
            <div className="space-y-6">
              {/* Scorecard Hero KPIs */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
                  <div className="text-xs text-slate-400">Pipeline Execution</div>
                  <div className="text-2xl font-bold text-emerald-400 mt-1">{scorecard.pipeline_execution_rate}%</div>
                  <div className="text-[10px] text-slate-500 mt-1">25/25 Scenes Loaded</div>
                </div>
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
                  <div className="text-xs text-slate-400">Detection mAP@0.5</div>
                  <div className="text-2xl font-bold text-indigo-400 mt-1">{scorecard.detection_metrics.map_50}</div>
                  <div className="text-[10px] text-slate-500 mt-1">mAP@0.5:0.95: {scorecard.detection_metrics.map_50_95}</div>
                </div>
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
                  <div className="text-xs text-slate-400">Segmentation mIoU</div>
                  <div className="text-2xl font-bold text-purple-400 mt-1">{scorecard.segmentation_metrics.mean_iou}</div>
                  <div className="text-[10px] text-slate-500 mt-1">5 Infrastructure Classes</div>
                </div>
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
                  <div className="text-xs text-slate-400">Geospatial Error</div>
                  <div className="text-2xl font-bold text-amber-400 mt-1">
                    {scorecard.geospatial_accuracy?.mean_association_error_meters != null
                      ? `${scorecard.geospatial_accuracy.mean_association_error_meters}m`
                      : 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    {scorecard.geospatial_accuracy?.percent_correctly_snapped != null
                      ? `${scorecard.geospatial_accuracy.percent_correctly_snapped}% Snapped`
                      : (scorecard.geospatial_accuracy?.status || 'Unmeasured')}
                  </div>
                </div>
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
                  <div className="text-xs text-slate-400">Temporal Dedup</div>
                  <div className="text-2xl font-bold text-cyan-400 mt-1">
                    {scorecard.temporal_deduplication?.suppression_efficiency_percent != null
                      ? `${scorecard.temporal_deduplication.suppression_efficiency_percent}%`
                      : 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    {scorecard.temporal_deduplication?.frame_to_obstacle_reduction_ratio != null
                      ? `${scorecard.temporal_deduplication.frame_to_obstacle_reduction_ratio} Reduction`
                      : (scorecard.temporal_deduplication?.status || 'Unmeasured')}
                  </div>
                </div>
              </div>

              {/* Environmental Robustness Breakdown */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-base font-bold text-white">Environmental Robustness Slices</h3>
                    <p className="text-xs text-slate-400">Accuracy breakdown across real-world physical and illumination conditions.</p>
                  </div>
                  <button
                    onClick={handleRunRealityTest}
                    disabled={evaluatingReality}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg"
                  >
                    {evaluatingReality ? 'Evaluating...' : '↻ Re-Run Reality Test'}
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {Object.entries(scorecard.environmental_robustness).map(([cat, vals]) => (
                    <div key={cat} className="space-y-2 bg-slate-800/40 p-4 rounded-xl">
                      <h4 className="text-xs uppercase font-bold text-slate-300">{cat}</h4>
                      {Object.entries(vals).map(([sub, score]) => (
                        <div key={sub} className="flex justify-between items-center text-xs">
                          <span className="text-slate-400 capitalize">{sub.replace('_', ' ')}</span>
                          <span className="font-semibold text-white">{Math.round(score * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>

              {/* Failed Predictions Review Queue */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-base font-bold text-white mb-2">
                  Preserved Failed Predictions Audit Queue ({scorecard.failed_predictions_count})
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Transparent preservation of misclassified or low-IoU street scenes for human accessibility auditor review.
                </p>
                <div className="space-y-3">
                  {scorecard.failed_predictions_sample.map((f) => (
                    <div key={f.sample_id} className="p-3 bg-slate-800/40 border border-slate-800 rounded-xl flex items-center justify-between text-xs">
                      <div>
                        <div className="font-semibold text-white">{f.title}</div>
                        <div className="text-slate-400 mt-0.5">
                          Expected: <strong className="text-amber-400">{f.expected_hazard}</strong> | Reason: {f.diagnostic_reason}
                        </div>
                      </div>
                      <span className="px-2.5 py-1 bg-amber-500/10 text-amber-400 rounded border border-amber-500/30">
                        Needs Human Review
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: GEOSPATIAL PROVENANCE & FRESHNESS */}
      {activeTab === 'provenance' && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <h2 className="text-lg font-bold text-white">Provenance, Freshness & Routing Guardrails</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-4 bg-slate-800/60 rounded-xl space-y-2">
              <div className="font-bold text-emerald-400">Fresh Observations (&lt; 30 days)</div>
              <p className="text-slate-300">Given full 100% weighting in network accessibility routing calculations.</p>
            </div>
            <div className="p-4 bg-slate-800/60 rounded-xl space-y-2">
              <div className="font-bold text-amber-400">Aging Observations (30 - 180 days)</div>
              <p className="text-slate-300">Weighted with decay coefficient; flagged with inspection reminder.</p>
            </div>
            <div className="p-4 bg-slate-800/60 rounded-xl space-y-2">
              <div className="font-bold text-rose-400">Stale Observations (&gt; 180 days)</div>
              <p className="text-slate-300">
                Network routing penalty applied. <strong>Routing engine will NEVER certify an unassessed or stale route as accessible.</strong>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
