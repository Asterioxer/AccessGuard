import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { MapView } from './components/MapView';
import { UploadModal } from './components/UploadModal';
import { ObservationDetailModal } from './components/ObservationDetailModal';
import { RoutePlanner } from './components/RoutePlanner';
import { VerificationQueue } from './components/VerificationQueue';
import { SegmentScoringPanel } from './components/SegmentScoringPanel';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { LiveCamera } from './components/LiveCamera';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { AutonomousDashboard } from './components/AutonomousDashboard';
import { api } from './api/client';
import { Observation, PedestrianSegment, RouteResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'map' | 'route' | 'verification' | 'analytics' | 'live_camera' | 'evaluation' | 'autonomous'>('map');
  const [observations, setObservations] = useState<Observation[]>([]);
  const [segments, setSegments] = useState<PedestrianSegment[]>([]);
  const [selectedSegment, setSelectedSegment] = useState<PedestrianSegment | null>(null);
  const [selectedObservation, setSelectedObservation] = useState<Observation | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [activeRoute, setActiveRoute] = useState<RouteResponse | null>(null);

  const loadData = async () => {
    try {
      const [obsData, segData] = await Promise.all([
        api.getObservations(),
        api.getSegments(),
      ]);
      setObservations(obsData);
      setSegments(segData);
    } catch (err) {
      console.error('Failed to load initial data:', err);
    }
  };

  useEffect(() => {
    loadData();
    const refreshSegments = () => {
      api.getSegments().then(setSegments).catch(console.error);
    };
    window.addEventListener('accessguard:segments-updated', refreshSegments);
    return () => window.removeEventListener('accessguard:segments-updated', refreshSegments);
  }, []);

  const handleUploadSuccess = (newObs: Observation) => {
    setObservations((prev) => [newObs, ...prev]);
    // Refresh segments as score may have updated
    api.getSegments().then(setSegments).catch(console.error);
    setSelectedObservation(newObs);
  };

  const handleVerificationUpdated = (updatedObs: Observation) => {
    setObservations((prev) =>
      prev.map((o) => (o.id === updatedObs.id ? updatedObs : o))
    );
    setSelectedObservation(updatedObs);
    // Refresh segments
    api.getSegments().then(setSegments).catch(console.error);
  };

  const handleSegmentUpdated = (updatedSeg: PedestrianSegment) => {
    setSegments((prev) =>
      prev.map((s) => (s.id === updatedSeg.id ? updatedSeg : s))
    );
    setSelectedSegment(updatedSeg);
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenUpload={() => setIsUploadOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1 relative overflow-hidden">
        {/* Tab 1: Map Explorer */}
        {activeTab === 'map' && (
          <div className="relative w-full h-full">
            <MapView
              observations={observations}
              segments={segments}
              selectedSegment={selectedSegment}
              onSelectSegment={setSelectedSegment}
              onSelectObservation={setSelectedObservation}
              routeCoordinates={activeRoute?.path_coordinates}
            />

            {/* Segment Scoring Details Overlay */}
            <SegmentScoringPanel
              segment={selectedSegment}
              onClose={() => setSelectedSegment(null)}
              onSegmentUpdated={handleSegmentUpdated}
            />
          </div>
        )}

        {/* Tab 2: Route Planner */}
        {activeTab === 'route' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 h-full overflow-hidden">
            <div className="lg:col-span-4 p-4 overflow-y-auto border-r border-slate-800 bg-slate-950">
              <RoutePlanner
                onRouteCalculated={setActiveRoute}
                currentRoute={activeRoute}
              />
            </div>
            <div className="lg:col-span-8 h-full relative">
              <MapView
                observations={observations}
                segments={segments}
                selectedSegment={selectedSegment}
                onSelectSegment={setSelectedSegment}
                onSelectObservation={setSelectedObservation}
                routeCoordinates={activeRoute?.path_coordinates}
              />
            </div>
          </div>
        )}

        {/* Tab 3: Human Verification Queue */}
        {activeTab === 'verification' && (
          <div className="h-full overflow-y-auto p-6 bg-slate-950">
            <VerificationQueue
              onInspectObservation={setSelectedObservation}
              onRefreshData={loadData}
            />
          </div>
        )}

        {/* Tab 4: Platform Analytics */}
        {activeTab === 'analytics' && (
          <div className="h-full overflow-y-auto p-6 bg-slate-950">
            <AnalyticsDashboard />
          </div>
        )}

        {/* Tab 5: Live Camera Testing & Real-Time Inference */}
        {activeTab === 'live_camera' && (
          <div className="h-full overflow-y-auto p-6 bg-slate-950 flex justify-center">
            <div className="w-full max-w-4xl">
              <LiveCamera />
            </div>
          </div>
        )}

        {/* Tab 6: Model Evaluation & Datasets Intelligence */}
        {activeTab === 'evaluation' && (
          <div className="h-full overflow-y-auto p-6 bg-slate-950">
            <div className="max-w-7xl mx-auto">
              <EvaluationDashboard />
            </div>
          </div>
        )}

        {/* Tab 7: Autonomous Field Intelligence & Reality Scorecard */}
        {activeTab === 'autonomous' && (
          <div className="h-full overflow-y-auto p-6 bg-slate-950">
            <div className="max-w-7xl mx-auto">
              <AutonomousDashboard />
            </div>
          </div>
        )}
      </main>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />

      {/* Observation Inspection & Evidence Modal */}
      <ObservationDetailModal
        observation={selectedObservation}
        onClose={() => setSelectedObservation(null)}
        onVerificationUpdated={handleVerificationUpdated}
      />
    </div>
  );
};
