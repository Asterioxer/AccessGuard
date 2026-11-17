import React, { useState } from 'react';
import { X, CheckCircle, AlertTriangle, HelpCircle, ThumbsUp, ThumbsDown, ShieldAlert, FileText, Check } from 'lucide-react';
import { Observation } from '../types';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';

interface ObservationDetailModalProps {
  observation: Observation | null;
  onClose: () => void;
  onVerificationUpdated: (updatedObs: Observation) => void;
}

export const ObservationDetailModal: React.FC<ObservationDetailModalProps> = ({
  observation,
  onClose,
  onVerificationUpdated,
}) => {
  const { user } = useAuth();
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [voteSubmitted, setVoteSubmitted] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [imageError, setImageError] = useState(false);

  if (!observation) return null;

  const handleVote = async (vote: 'confirm' | 'dispute' | 'reject') => {
    if (!user) {
      setActionError('Please sign in to submit community verifications.');
      return;
    }

    setIsSubmitting(true);
    setActionError(null);

    try {
      await api.submitVerification({
        observation_id: observation.id,
        vote,
        notes: notes || undefined,
      });
      setVoteSubmitted(vote);
      // Refresh observation data
      const refreshed = await api.getObservation(observation.id);
      onVerificationUpdated(refreshed);
    } catch (err: any) {
      setActionError(err.message || 'Verification submission failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full p-6 shadow-2xl relative max-h-[90vh] overflow-y-auto">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2.5 rounded-xl bg-slate-800 border border-slate-700">
            <ShieldAlert className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-bold text-white">Observation #{observation.id.slice(0, 8)}</h2>
              <span
                className={`text-[11px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${
                  observation.status === 'verified'
                    ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
                    : observation.status === 'disputed'
                    ? 'bg-rose-950/80 text-rose-300 border-rose-800'
                    : 'bg-amber-950/80 text-amber-300 border-amber-800'
                }`}
              >
                {observation.status}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Coordinates: {observation.latitude != null && observation.longitude != null ? `${observation.latitude.toFixed(5)}, ${observation.longitude.toFixed(5)}` : 'Unanchored (No GPS provided)'} • Recorded {new Date(observation.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>

        {/* Media with Bounding Boxes */}
        <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-950 mb-6 flex justify-center items-center max-h-[380px]">
          {observation.media_type === 'image' ? (
            <div className="relative inline-block w-full text-center">
              {imageError ? (
                <div className="p-8 text-center bg-slate-900 border border-rose-500/30 rounded-xl my-4">
                  <div className="text-rose-400 font-semibold text-sm">Failed to Load Street-Level Image</div>
                  <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                    Image media could not be retrieved. External API credentials (such as MAPILLARY_CLIENT_TOKEN) may be missing or the remote image URL is unreachable.
                  </p>
                </div>
              ) : (
                <img
                  src={observation.media_url}
                  alt="Sidewalk obstacle detection"
                  className="max-h-[360px] mx-auto object-contain rounded-lg"
                  onError={() => setImageError(true)}
                />
              )}
              {/* Overlaid Normalized Bounding Boxes */}
              {!imageError && (
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                {observation.detections.map((det) => {
                  const x = det.bbox.xmin * 100;
                  const y = det.bbox.ymin * 100;
                  const width = (det.bbox.xmax - det.bbox.xmin) * 100;
                  const height = (det.bbox.ymax - det.bbox.ymin) * 100;
                  const isKeypoint = !!det.evidence?.is_keypoint;
                  const isDisagreement = det.evidence?.consensus_status === 'DISAGREEMENT';
                  const strokeColor = isDisagreement ? '#f43f5e' : det.barrier_type === 'stairs' ? '#ef4444' : det.barrier_type === 'curb_ramp_present' ? '#10b981' : '#f59e0b';

                  return (
                    <g key={det.id}>
                      {isKeypoint ? (
                        <g>
                          <circle
                            cx={x + width / 2}
                            cy={y + height / 2}
                            r="3"
                            fill="#10b981"
                            stroke="#ffffff"
                            strokeWidth="1"
                          />
                          <circle
                            cx={x + width / 2}
                            cy={y + height / 2}
                            r="6"
                            fill="none"
                            stroke="#10b981"
                            strokeWidth="1"
                            strokeDasharray="2, 2"
                          />
                        </g>
                      ) : (
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          fill={strokeColor}
                          fillOpacity="0.15"
                          stroke={strokeColor}
                          strokeWidth={isDisagreement ? "2.5" : "1.5"}
                          strokeDasharray={det.is_uncertain ? '3, 2' : undefined}
                        />
                      )}
                    </g>
                  );
                })}
              </svg>
              )}
            </div>
          ) : (
            <video controls className="max-h-[360px] w-full object-contain">
              <source src={observation.media_url} type="video/mp4" />
            </video>
          )}
        </div>

        {/* Detected Barriers & Evidence Breakdown */}
        <div className="space-y-3 mb-6">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Detected Accessibility Barriers ({observation.detections.length})
          </h3>

          {observation.detections.map((det) => (
            <div
              key={det.id}
              className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 space-y-2.5"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-bold text-white capitalize">
                    {det.barrier_type.replace(/_/g, ' ')}
                  </span>
                  {det.is_confirmed ? (
                    <span className="text-[10px] font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded-full flex items-center space-x-1">
                      <CheckCircle className="w-3 h-3" />
                      <span>Confirmed by Evidence</span>
                    </span>
                  ) : det.is_uncertain ? (
                    <span className="text-[10px] font-semibold bg-amber-950 text-amber-400 border border-amber-800 px-2 py-0.5 rounded-full flex items-center space-x-1">
                      <HelpCircle className="w-3 h-3" />
                      <span>Uncertain Prediction</span>
                    </span>
                  ) : (
                    <span className="text-[10px] font-semibold bg-slate-700 text-slate-300 px-2 py-0.5 rounded-full">
                      Tentative Machine Flag
                    </span>
                  )}
                </div>

                <div className="text-right">
                  <span className="text-xs font-bold text-emerald-400">
                    {(det.confidence * 100).toFixed(0)}% Confidence
                  </span>
                  <span className="text-[10px] text-slate-500 block">
                    Uncertainty: {det.uncertainty_score.toFixed(2)}
                  </span>
                </div>
              </div>

              {det.evidence && (
                <div className="bg-slate-900/90 rounded-lg p-3 text-xs border border-slate-800 space-y-2">
                  {/* Multi-Model Consensus & Adapter Provenance Badges */}
                  <div className="flex flex-wrap items-center gap-1.5 pb-1 border-b border-slate-800">
                    {det.evidence.consensus_status === 'DISAGREEMENT' ? (
                      <span className="text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-700 px-2 py-0.5 rounded-full flex items-center space-x-1">
                        <AlertTriangle className="w-3 h-3 text-rose-400" />
                        <span>Multi-Model Disagreement</span>
                      </span>
                    ) : det.evidence.consensus_status === 'CONSENSUS' ? (
                      <span className="text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-700 px-2 py-0.5 rounded-full flex items-center space-x-1">
                        <CheckCircle className="w-3 h-3 text-emerald-400" />
                        <span>Multi-Model Consensus</span>
                      </span>
                    ) : null}

                    {det.evidence.verification_state === 'VERIFIED' && (
                      <span className="text-[10px] font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800 px-1.5 py-0.5 rounded">
                        Validator Verified
                      </span>
                    )}

                    {det.evidence.roughness_score !== undefined && (
                      <span className="text-[10px] font-medium bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded">
                        Roughness: {Number(det.evidence.roughness_score).toFixed(2)}
                      </span>
                    )}

                    {Array.isArray(det.evidence.contributing_adapters) && det.evidence.contributing_adapters.map((ad: string) => (
                      <span key={ad} className="text-[10px] font-mono bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded border border-slate-700">
                        {ad.replace('_adapter', '')}
                      </span>
                    ))}
                  </div>

                  {det.evidence.explanation && (
                    <div className="flex items-start space-x-2 text-slate-300">
                      <FileText className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                      <span><b>Evidence:</b> {det.evidence.explanation}</span>
                    </div>
                  )}

                  {det.evidence.feature_type && (
                    <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                      <div>Feature: <span className="text-slate-300">{det.evidence.feature_type}</span></div>
                      <div>Measured Value: <span className="text-slate-300">{det.evidence.measured_metric} (Threshold: {det.evidence.threshold_applied})</span></div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Community Verification & Peer Consensus Section */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-1.5">
              <Check className="w-4 h-4 text-emerald-400" />
              <span>Human Verification & Consensus</span>
            </h3>
            <span className="text-xs text-slate-400">
              Net Score: <b className="text-white">{observation.verification_score}</b>
            </span>
          </div>

          <p className="text-xs text-slate-400 mb-3">
            Pedestrian obstacles require multi-party community consensus before being marked as fully confirmed on accessible routing graphs.
          </p>

          {actionError && (
            <div className="mb-3 p-2 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-lg text-xs">
              {actionError}
            </div>
          )}

          {voteSubmitted ? (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-lg text-xs font-medium text-center">
              Thank you! Your verification review ({voteSubmitted}) has been recorded.
            </div>
          ) : (
            <div className="space-y-3">
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Optional field notes (e.g., 'Inspected on foot: 4 concrete steps, no ramp adjacent')"
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
              />

              <div className="flex space-x-2">
                <button
                  onClick={() => handleVote('confirm')}
                  disabled={isSubmitting}
                  className="flex-1 bg-emerald-600/90 hover:bg-emerald-500 text-white py-2 rounded-lg text-xs font-semibold flex items-center justify-center space-x-1 transition-colors"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                  <span>Confirm Barrier</span>
                </button>
                <button
                  onClick={() => handleVote('dispute')}
                  disabled={isSubmitting}
                  className="flex-1 bg-slate-700 hover:bg-slate-600 text-slate-200 py-2 rounded-lg text-xs font-semibold flex items-center justify-center space-x-1 transition-colors"
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                  <span>Dispute / Inaccurate</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
