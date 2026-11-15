import React, { useState, useEffect } from 'react';
import { CheckCircle2, AlertCircle, ThumbsUp, ThumbsDown, Eye, ShieldCheck, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { Observation } from '../types';
import { useAuth } from '../context/AuthContext';

interface VerificationQueueProps {
  onInspectObservation: (obs: Observation) => void;
  onRefreshData: () => void;
}

export const VerificationQueue: React.FC<VerificationQueueProps> = ({
  onInspectObservation,
  onRefreshData,
}) => {
  const { user } = useAuth();
  const [queue, setQueue] = useState<Observation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const loadQueue = async () => {
    setIsLoading(true);
    try {
      const items = await api.getVerificationQueue();
      setQueue(items);
    } catch (err) {
      console.error('Failed to load verification queue:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadQueue();
  }, []);

  const handleQuickVote = async (obsId: string, vote: 'confirm' | 'dispute') => {
    if (!user) {
      setActionMessage('Please sign in to submit community verifications.');
      return;
    }
    try {
      await api.submitVerification({
        observation_id: obsId,
        vote,
        notes: `Quick verification vote: ${vote}`,
      });
      setActionMessage(`Observation vote (${vote}) recorded successfully.`);
      loadQueue();
      onRefreshData();
    } catch (err: any) {
      setActionMessage(err.message || 'Verification submission failed');
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-emerald-500/10 rounded-xl border border-emerald-500/30">
            <CheckCircle2 className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Human Verification Queue</h2>
            <p className="text-xs text-slate-400">
              Community peer review: confirming and correcting AI-flagged sidewalk obstacles
            </p>
          </div>
        </div>

        <button
          onClick={loadQueue}
          disabled={isLoading}
          className="p-2 text-slate-400 hover:text-white bg-slate-800 rounded-lg border border-slate-700 transition-colors"
          title="Refresh Queue"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {actionMessage && (
        <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-xl text-xs flex items-center justify-between">
          <span>{actionMessage}</span>
          <button onClick={() => setActionMessage(null)} className="text-slate-400 hover:text-white ml-2">×</button>
        </div>
      )}

      {!user && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/30 text-amber-300 rounded-xl text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>You are browsing as a guest. Sign in to vote on pending obstacles.</span>
        </div>
      )}

      {isLoading ? (
        <div className="text-center py-12 text-slate-500 text-sm">
          Loading unverified submissions...
        </div>
      ) : queue.length === 0 ? (
        <div className="text-center py-12 space-y-2">
          <ShieldCheck className="w-12 h-12 text-emerald-400 mx-auto" />
          <h3 className="text-sm font-semibold text-white">All Clear!</h3>
          <p className="text-xs text-slate-400">No pending observations requiring human review right now.</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {queue.map((obs) => {
            const barrierTypes = obs.detections.map((d) => d.barrier_type.replace(/_/g, ' ')).join(', ');
            const topDet = obs.detections[0];

            return (
              <div
                key={obs.id}
                className="bg-slate-800/80 border border-slate-700 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 hover:border-slate-600 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-16 h-16 rounded-lg bg-slate-950 overflow-hidden border border-slate-700 flex-shrink-0">
                    <img
                      src={obs.media_url}
                      alt="Thumbnail"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-bold text-white capitalize">
                        {barrierTypes || 'Pending Obstacle'}
                      </span>
                      <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-amber-950 text-amber-400 border border-amber-800">
                        Tentative
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Coordinates: {obs.latitude?.toFixed(4)}, {obs.longitude?.toFixed(4)}
                    </p>
                    {topDet && (
                      <p className="text-[11px] text-slate-300 mt-1">
                        Confidence: <b className="text-emerald-400">{(topDet.confidence * 100).toFixed(0)}%</b> • Uncertainty: {topDet.uncertainty_score.toFixed(2)}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center space-x-2 self-end sm:self-center">
                  <button
                    onClick={() => onInspectObservation(obs)}
                    className="p-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg text-xs font-medium flex items-center space-x-1 transition-colors"
                    title="Inspect Evidence & Bounding Boxes"
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>Inspect</span>
                  </button>

                  <button
                    onClick={() => handleQuickVote(obs.id, 'confirm')}
                    disabled={!user}
                    className="p-2 bg-emerald-600/90 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors"
                    title="Confirm Obstacle Presence"
                  >
                    <ThumbsUp className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Confirm</span>
                  </button>

                  <button
                    onClick={() => handleQuickVote(obs.id, 'dispute')}
                    disabled={!user}
                    className="p-2 bg-slate-700 hover:bg-rose-600 disabled:opacity-50 text-slate-300 hover:text-white rounded-lg text-xs font-semibold flex items-center space-x-1 transition-colors"
                    title="Dispute AI Detection"
                  >
                    <ThumbsDown className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Dispute</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
