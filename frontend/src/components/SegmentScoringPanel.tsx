import React, { useState } from 'react';
import { X, Layers, RefreshCw, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { PedestrianSegment } from '../types';
import { api } from '../api/client';

interface SegmentScoringPanelProps {
  segment: PedestrianSegment | null;
  onClose: () => void;
  onSegmentUpdated: (updated: PedestrianSegment) => void;
}

export const SegmentScoringPanel: React.FC<SegmentScoringPanelProps> = ({
  segment,
  onClose,
  onSegmentUpdated,
}) => {
  const [isRecalculating, setIsRecalculating] = useState(false);

  if (!segment) return null;

  const handleRecalculate = async () => {
    setIsRecalculating(true);
    try {
      const updated = await api.recalculateSegment(segment.id);
      onSegmentUpdated(updated);
    } catch (err) {
      console.error('Recalculation error:', err);
    } finally {
      setIsRecalculating(false);
    }
  };

  const breakdown = segment.score_breakdown;

  const getGradeBadge = (grade: string) => {
    switch (grade) {
      case 'A':
        return 'bg-emerald-950 text-emerald-400 border-emerald-800';
      case 'B':
        return 'bg-sky-950 text-sky-400 border-sky-800';
      case 'C':
        return 'bg-amber-950 text-amber-400 border-amber-800';
      case 'D':
        return 'bg-orange-950 text-orange-400 border-orange-800';
      case 'U':
        return 'bg-slate-900 text-slate-400 border-slate-700';
      default:
        return 'bg-rose-950 text-rose-400 border-rose-800';
    }
  };

  return (
    <div className="absolute top-6 right-6 z-[1000] w-96 max-w-[calc(100vw-3rem)] bg-slate-900/95 backdrop-blur-md border border-slate-700/80 rounded-2xl p-5 shadow-2xl space-y-4 max-h-[85vh] overflow-y-auto">
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-2">
          <div className="p-2 rounded-lg bg-slate-800 border border-slate-700">
            <Layers className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white leading-tight">{segment.name}</h3>
            <span className="text-[10px] text-slate-400 font-mono">ID: {segment.id}</span>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white p-1">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Big Score Card */}
      <div className="bg-slate-800/80 rounded-xl p-4 border border-slate-700 flex items-center justify-between">
        <div>
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
            Accessibility Score
          </span>
          <div className="text-3xl font-black text-white">
            {segment.accessibility_score}
            <span className="text-sm font-normal text-slate-400"> / 100</span>
          </div>
        </div>
        <div className="text-right">
          <span
            className={`text-xl font-black px-3.5 py-1 rounded-xl border inline-block ${getGradeBadge(
              segment.score_grade
            )}`}
          >
            Grade {segment.score_grade}
          </span>
          <span className="text-[10px] text-slate-400 block mt-1">
            Length: {segment.length_meters}m
          </span>
        </div>
      </div>

      {/* Transparent Breakdown Criteria */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-1.5">
            <Info className="w-3.5 h-3.5 text-cyan-400" />
            <span>Transparent Scoring Criteria</span>
          </h4>
          <button
            onClick={handleRecalculate}
            disabled={isRecalculating}
            className="text-[10px] text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 disabled:opacity-50"
            title="Recalculate score from linked observations"
          >
            <RefreshCw className={`w-3 h-3 ${isRecalculating ? 'animate-spin' : ''}`} />
            <span>Recalculate</span>
          </button>
        </div>

        <div className="bg-slate-950/70 rounded-xl p-3 border border-slate-800 text-xs space-y-2 font-mono">
          <div className="flex justify-between text-slate-300">
            <span>Base Benchmark:</span>
            <span className="text-emerald-400">+{breakdown.base_score} pts</span>
          </div>

          {breakdown.stairs_penalty > 0 && (
            <div className="flex justify-between text-rose-400">
              <span>Stairs / Unramped Steps:</span>
              <span>-{breakdown.stairs_penalty} pts</span>
            </div>
          )}

          {breakdown.curb_ramp_missing_penalty > 0 && (
            <div className="flex justify-between text-rose-400">
              <span>Missing Curb Ramp:</span>
              <span>-{breakdown.curb_ramp_missing_penalty} pts</span>
            </div>
          )}

          {breakdown.surface_damage_penalty > 0 && (
            <div className="flex justify-between text-amber-400">
              <span>Pavement Surface Damage:</span>
              <span>-{breakdown.surface_damage_penalty} pts</span>
            </div>
          )}

          {breakdown.obstruction_penalty > 0 && (
            <div className="flex justify-between text-amber-400">
              <span>Path Corridor Obstacles:</span>
              <span>-{breakdown.obstruction_penalty} pts</span>
            </div>
          )}

          {breakdown.curb_ramp_bonus > 0 && (
            <div className="flex justify-between text-emerald-400">
              <span>Accessible Curb Ramp:</span>
              <span>+{breakdown.curb_ramp_bonus} pts</span>
            </div>
          )}

          <div className="pt-2 border-t border-slate-800 flex justify-between font-bold text-white font-sans text-sm">
            <span>Net Calculated Score:</span>
            <span>{breakdown.net_score} pts</span>
          </div>
        </div>
      </div>

      {/* Penalties / Bonuses Explanations */}
      {breakdown.penalties_detail.length > 0 && (
        <div className="space-y-1">
          <span className="text-[10px] font-semibold text-rose-400 block uppercase tracking-wider">
            Detected Impairments:
          </span>
          <div className="space-y-1 text-[11px] text-slate-300">
            {breakdown.penalties_detail.map((item, idx) => (
              <div key={idx} className="flex items-start space-x-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400 flex-shrink-0 mt-0.5" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {breakdown.bonuses_detail.length > 0 && (
        <div className="space-y-1">
          <span className="text-[10px] font-semibold text-emerald-400 block uppercase tracking-wider">
            Accessibility Assets:
          </span>
          <div className="space-y-1 text-[11px] text-slate-300">
            {breakdown.bonuses_detail.map((item, idx) => (
              <div key={idx} className="flex items-start space-x-1.5">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
