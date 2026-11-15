import React, { useState } from 'react';
import { Navigation, ShieldCheck, AlertTriangle, Clock, Compass, Layers, Check } from 'lucide-react';
import { api } from '../api/client';
import { RouteResponse } from '../types';

interface RoutePlannerProps {
  onRouteCalculated: (route: RouteResponse | null) => void;
  currentRoute: RouteResponse | null;
}

export const RoutePlanner: React.FC<RoutePlannerProps> = ({ onRouteCalculated, currentRoute }) => {
  const [startLat, setStartLat] = useState('37.7749');
  const [startLon, setStartLon] = useState('-122.4194');
  const [destLat, setDestLat] = useState('37.7788');
  const [destLon, setDestLon] = useState('-122.4194');
  const [profile, setProfile] = useState('manual_wheelchair');
  const [avoidUnverified, setAvoidUnverified] = useState(true);
  const [algorithm, setAlgorithm] = useState<'dijkstra' | 'astar'>('dijkstra');
  const [includeAlternatives, setIncludeAlternatives] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCalculateRoute = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const res = await api.calculateRoute({
        start: { latitude: parseFloat(startLat), longitude: parseFloat(startLon) },
        destination: { latitude: parseFloat(destLat), longitude: parseFloat(destLon) },
        wheelchair_profile: profile,
        avoid_unverified_barriers: avoidUnverified,
        algorithm,
        include_alternatives: includeAlternatives,
      });
      onRouteCalculated(res);
    } catch (err: any) {
      setError(err.message || 'Failed to compute accessible route');
      onRouteCalculated(null);
    } finally {
      setIsLoading(false);
    }
  };

  const setPreset = (sLat: string, sLon: string, dLat: string, dLon: string) => {
    setStartLat(sLat);
    setStartLon(sLon);
    setDestLat(dLat);
    setDestLon(dLon);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-5">
      <div className="flex items-center space-x-2.5">
        <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
          <Navigation className="w-5 h-5 text-cyan-400" />
        </div>
        <div>
          <h2 className="text-base font-bold text-white">Wheelchair Route Planner</h2>
          <p className="text-xs text-slate-400">Barrier-aware graph navigation avoiding stairs & hazards</p>
        </div>
      </div>

      {/* Preset Route Scenarios */}
      <div>
        <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
          Benchmark Demonstration Scenarios
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setPreset('37.7765', '-122.4194', '37.7778', '-122.4194')}
            className="text-left p-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/80 transition-colors text-xs"
          >
            <div className="font-semibold text-slate-200">Central Plaza Transit</div>
            <div className="text-[10px] text-amber-400">Tests stair avoidance via ramp bypass</div>
          </button>
          <button
            type="button"
            onClick={() => setPreset('37.7749', '-122.4194', '37.7788', '-122.4194')}
            className="text-left p-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-800 border border-slate-700/80 transition-colors text-xs"
          >
            <div className="font-semibold text-slate-200">Main St Full Corridor</div>
            <div className="text-[10px] text-emerald-400">Full corridor with verified crossings</div>
          </button>
        </div>
      </div>

      <form onSubmit={handleCalculateRoute} className="space-y-4">
        {/* Origin */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
            <span>Origin Coordinates</span>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <input
              type="number"
              step="any"
              value={startLat}
              onChange={(e) => setStartLat(e.target.value)}
              placeholder="Start Lat"
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
            <input
              type="number"
              step="any"
              value={startLon}
              onChange={(e) => setStartLon(e.target.value)}
              placeholder="Start Lon"
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>

        {/* Destination */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
            <span>Destination Coordinates</span>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <input
              type="number"
              step="any"
              value={destLat}
              onChange={(e) => setDestLat(e.target.value)}
              placeholder="Dest Lat"
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
            <input
              type="number"
              step="any"
              value={destLon}
              onChange={(e) => setDestLon(e.target.value)}
              placeholder="Dest Lon"
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
          </div>
        </div>

        {/* Profile & Algorithm */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Pedestrian Profile</label>
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-2 text-white text-xs focus:outline-none focus:ring-1 focus:ring-cyan-500"
              aria-label="Pedestrian mobility profile"
            >
              <option value="manual_wheelchair">Manual Wheelchair</option>
              <option value="powered_wheelchair">Power Wheelchair</option>
              <option value="stroller">Stroller / Walker</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">Graph Algorithm</label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as 'dijkstra' | 'astar')}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-2 text-white text-xs focus:outline-none focus:ring-1 focus:ring-cyan-500"
              aria-label="Pathfinding algorithm selection"
            >
              <option value="dijkstra">Dijkstra (Exact Cost)</option>
              <option value="astar">A* (Haversine Heuristic)</option>
            </select>
          </div>
        </div>

        {/* Options */}
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="avoidUnverified"
              checked={avoidUnverified}
              onChange={(e) => setAvoidUnverified(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <label htmlFor="avoidUnverified" className="text-xs text-slate-300 cursor-pointer">
              Penalize tentative community obstacles
            </label>
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="includeAlternatives"
              checked={includeAlternatives}
              onChange={(e) => setIncludeAlternatives(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <label htmlFor="includeAlternatives" className="text-xs text-slate-300 cursor-pointer">
              Compare with direct shortest route tradeoff
            </label>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-xl text-xs flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-emerald-600 hover:from-cyan-500 hover:to-emerald-500 text-white rounded-xl text-xs font-bold transition-all shadow-lg shadow-cyan-900/30 flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <span>Evaluating Graph Traversal...</span>
          ) : (
            <>
              <Compass className="w-4 h-4" />
              <span>Calculate Accessible Route</span>
            </>
          )}
        </button>
      </form>

      {/* Calculated Route Summary */}
      {currentRoute && currentRoute.success && (
        <div className="pt-4 border-t border-slate-800 space-y-4">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-slate-800/80 p-2.5 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 block">Distance</span>
              <span className="text-sm font-bold text-white">{currentRoute.total_distance_meters} m</span>
            </div>
            <div className="bg-slate-800/80 p-2.5 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 block">Duration</span>
              <span className="text-sm font-bold text-white">{currentRoute.estimated_duration_minutes} min</span>
            </div>
            <div className="bg-slate-800/80 p-2.5 rounded-xl border border-slate-700">
              <span className="text-[10px] text-slate-400 block">Score</span>
              <span className="text-sm font-bold text-emerald-400">
                {currentRoute.overall_accessibility_score}/100 ({currentRoute.route_grade})
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2 p-2.5 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 text-xs font-semibold">
            <ShieldCheck className="w-4 h-4 flex-shrink-0 text-emerald-400" />
            <span>{currentRoute.avoids_stairs ? 'Verified Stair-Free Accessible Route' : 'Warning: Unavoidable Obstacles'}</span>
          </div>

          {currentRoute.direct_route_comparison && (
            <div className="p-3 bg-slate-800/60 rounded-xl border border-slate-700/80 space-y-1.5 text-xs">
              <div className="flex items-center justify-between text-slate-300 font-semibold">
                <span>Route Efficiency Comparison</span>
                <span className="text-[10px] text-cyan-400 font-mono">Algorithm: {currentRoute.algorithm_used || 'Dijkstra'}</span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>Shortest Direct Distance:</span>
                <span className="text-white font-mono">{currentRoute.direct_route_comparison.direct_distance_meters} m</span>
              </div>
              {currentRoute.direct_route_comparison.accessible_detour_meters > 0 && (
                <div className="flex items-center justify-between text-[11px] text-amber-400">
                  <span>Accessible Ramp Bypass Detour:</span>
                  <span className="font-mono">+{currentRoute.direct_route_comparison.accessible_detour_meters} m</span>
                </div>
              )}
              {currentRoute.direct_route_comparison.direct_route_has_stairs && (
                <div className="flex items-center space-x-1.5 text-[11px] text-emerald-400 pt-1 border-t border-slate-700">
                  <Check className="w-3.5 h-3.5 flex-shrink-0" />
                  <span>Safely detours around {currentRoute.direct_route_comparison.stairs_avoided_by_accessible_route} direct staircase obstacle(s).</span>
                </div>
              )}
            </div>
          )}

          {currentRoute.routing_notes.length > 0 && (
            <div className="space-y-1 text-xs text-slate-300 bg-slate-800/40 p-3 rounded-xl border border-slate-800">
              {currentRoute.routing_notes.map((note, idx) => (
                <div key={idx} className="flex items-start space-x-1.5">
                  <span className="text-cyan-400 font-bold">•</span>
                  <span>{note}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
