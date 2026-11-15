import React, { useState, useEffect } from 'react';
import { BarChart3, ShieldCheck, CheckCircle2, AlertTriangle, EyeOff, Lock, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { AnalyticsSummary } from '../types';

export const AnalyticsDashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const data = await api.getAnalytics();
      setAnalytics(data);
    } catch (err) {
      console.error('Failed to load analytics:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (isLoading) {
    return (
      <div className="text-center py-24 text-slate-500 text-sm">
        Loading platform intelligence analytics...
      </div>
    );
  }

  if (!analytics) return null;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Top Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
            City Accessibility Index
          </span>
          <div className="text-3xl font-black text-white mt-1">
            {analytics.average_city_accessibility_score}
            <span className="text-sm font-normal text-slate-400"> / 100</span>
          </div>
          <p className="text-[11px] text-emerald-400 mt-1">Weighted pedestrian sidewalk score</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
            Total Obstacles Mapped
          </span>
          <div className="text-3xl font-black text-white mt-1">
            {analytics.total_observations}
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Street-level observations recorded</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
            Human Verified Rate
          </span>
          <div className="text-3xl font-black text-emerald-400 mt-1">
            {(analytics.verification_rate * 100).toFixed(0)}%
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Community peer consensus</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
            Tentative / Pending Review
          </span>
          <div className="text-3xl font-black text-amber-400 mt-1">
            {analytics.tentative_observations}
          </div>
          <p className="text-[11px] text-slate-400 mt-1">Awaiting multi-party confirmation</p>
        </div>
      </div>

      {/* Barrier Distribution & Segment Grades */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Barrier Classes */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Barrier Classification Breakdown</span>
          </h3>

          <div className="space-y-3">
            {Object.entries(analytics.barrier_distribution).length === 0 ? (
              <p className="text-xs text-slate-500 py-4">No barrier detections recorded yet.</p>
            ) : (
              Object.entries(analytics.barrier_distribution).map(([bType, count]) => (
                <div key={bType} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300 capitalize">{bType.replace(/_/g, ' ')}</span>
                    <span className="font-bold text-white">{count} detected</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-500"
                      style={{
                        width: `${Math.min(100, (count / (analytics.total_observations || 1)) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Segment Grade Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-emerald-400" />
            <span>Pedestrian Segment Grade Distribution</span>
          </h3>

          <div className="space-y-3">
            {['A', 'B', 'C', 'D', 'F'].map((grade) => {
              const count = analytics.segment_grade_distribution[grade] || 0;
              const colorClass =
                grade === 'A'
                  ? 'bg-emerald-500'
                  : grade === 'B'
                  ? 'bg-sky-500'
                  : grade === 'C'
                  ? 'bg-amber-500'
                  : grade === 'D'
                  ? 'bg-orange-500'
                  : 'bg-rose-500';

              return (
                <div key={grade} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300 font-semibold">Grade {grade}</span>
                    <span className="font-bold text-white">{count} segments</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${colorClass}`}
                      style={{ width: `${Math.min(100, (count / 10) * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Engineering Safeguards Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center space-x-2">
          <Lock className="w-4 h-4 text-cyan-400" />
          <span>Engineering Constraints & Privacy Safeguards</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs text-slate-400">
          <div className="bg-slate-800/60 p-3.5 rounded-xl border border-slate-700/60 space-y-1">
            <span className="font-semibold text-emerald-400 block">No Facial Recognition</span>
            <p>Faces and vehicle license plates are blurred by sanitization filters before storage.</p>
          </div>
          <div className="bg-slate-800/60 p-3.5 rounded-xl border border-slate-700/60 space-y-1">
            <span className="font-semibold text-emerald-400 block">Metadata Stripping</span>
            <p>GPS and camera serial EXIF metadata are stripped immediately after coordinate parsing.</p>
          </div>
          <div className="bg-slate-800/60 p-3.5 rounded-xl border border-slate-700/60 space-y-1">
            <span className="font-semibold text-emerald-400 block">Objective Calibration</span>
            <p>Obstacles require high evidence or consensus; borderlines are flagged explicitly as tentative.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
