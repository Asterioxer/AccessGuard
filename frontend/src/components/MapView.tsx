import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { Observation, PedestrianSegment } from '../types';

interface MapViewProps {
  observations: Observation[];
  segments: PedestrianSegment[];
  selectedSegment: PedestrianSegment | null;
  onSelectSegment: (segment: PedestrianSegment | null) => void;
  onSelectObservation: (observation: Observation) => void;
  routeCoordinates?: [number, number][];
  onMapClickCoord?: (coord: { lat: number; lng: number }) => void;
}

const GRADE_COLORS: Record<string, string> = {
  A: '#10b981', // Emerald green
  B: '#0284c7', // Sky blue
  C: '#eab308', // Amber yellow
  D: '#f97316', // Orange
  F: '#ef4444', // Rose red
};

export const MapView: React.FC<MapViewProps> = ({
  observations,
  segments,
  selectedSegment,
  onSelectSegment,
  onSelectObservation,
  routeCoordinates,
  onMapClickCoord,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const segmentLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const observationLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const routeLayerGroupRef = useRef<L.LayerGroup | null>(null);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center on downtown innovation corridor (San Francisco baseline)
    const map = L.map(mapContainerRef.current, {
      center: [37.7765, -122.4194],
      zoom: 16,
      zoomControl: true,
    });

    // Dark Matter tile layer for high contrast accessibility visualization
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 20,
    }).addTo(map);

    segmentLayerGroupRef.current = L.layerGroup().addTo(map);
    observationLayerGroupRef.current = L.layerGroup().addTo(map);
    routeLayerGroupRef.current = L.layerGroup().addTo(map);

    map.on('click', (e: L.LeafletMouseEvent) => {
      if (onMapClickCoord) {
        onMapClickCoord({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
    });

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Pedestrian Segments Layer
  useEffect(() => {
    if (!mapInstanceRef.current || !segmentLayerGroupRef.current) return;
    segmentLayerGroupRef.current.clearLayers();

    segments.forEach((seg) => {
      try {
        let latlngs: [number, number][] = [];
        if (seg.geometry_geojson) {
          const parsed = JSON.parse(seg.geometry_geojson);
          latlngs = parsed.map((pt: [number, number]) => [pt[1], pt[0]]); // [lat, lon]
        } else {
          latlngs = [
            [seg.start_lat, seg.start_lon],
            [seg.end_lat, seg.end_lon],
          ];
        }

        const isSelected = selectedSegment?.id === seg.id;
        const color = GRADE_COLORS[seg.score_grade] || '#64748b';

        const polyline = L.polyline(latlngs, {
          color,
          weight: isSelected ? 8 : 5,
          opacity: isSelected ? 1.0 : 0.85,
        });

        polyline.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          onSelectSegment(seg);
        });

        polyline.bindTooltip(
          `<strong>${seg.name}</strong><br/>Score: <b>${seg.accessibility_score}</b> (Grade ${seg.score_grade})`,
          { sticky: true }
        );

        segmentLayerGroupRef.current?.addLayer(polyline);
      } catch (err) {
        console.error('Error rendering segment:', seg.id, err);
      }
    });
  }, [segments, selectedSegment, onSelectSegment]);

  // Update Observation Barrier Markers Layer
  useEffect(() => {
    if (!mapInstanceRef.current || !observationLayerGroupRef.current) return;
    observationLayerGroupRef.current.clearLayers();

    observations.forEach((obs) => {
      if (!obs.latitude || !obs.longitude) return;

      const hasStairs = obs.detections.some((d) => d.barrier_type === 'stairs');
      const hasMissingRamp = obs.detections.some((d) => d.barrier_type === 'curb_ramp_missing');
      const hasDamage = obs.detections.some((d) => d.barrier_type === 'surface_damage');
      const hasRamp = obs.detections.some((d) => d.barrier_type === 'curb_ramp_present');

      let iconBg = '#f59e0b'; // amber default
      let iconLabel = '⚠️';
      if (hasStairs || hasMissingRamp) {
        iconBg = '#ef4444'; // red danger
        iconLabel = hasStairs ? '🪜' : '🚫';
      } else if (hasRamp) {
        iconBg = '#10b981'; // green accessible
        iconLabel = '♿';
      } else if (hasDamage) {
        iconBg = '#f97316'; // orange surface crack
        iconLabel = '⚡';
      }

      const isVerified = obs.status === 'verified';
      const borderStyle = isVerified ? '2px solid #ffffff' : '2px dashed #94a3b8';

      const customIcon = L.divIcon({
        className: 'custom-barrier-icon',
        html: `
          <div style="
            background-color: ${iconBg};
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            border: ${borderStyle};
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            cursor: pointer;
          ">
            ${iconLabel}
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const marker = L.marker([obs.latitude, obs.longitude], { icon: customIcon });

      marker.on('click', () => {
        onSelectObservation(obs);
      });

      const barrierTypes = obs.detections.map((d) => d.barrier_type.replace(/_/g, ' ')).join(', ');
      marker.bindTooltip(
        `<div style="font-size: 12px;">
          <b>${barrierTypes || 'Observation'}</b><br/>
          Status: <span style="text-transform: capitalize;">${obs.status}</span><br/>
          Click to inspect evidence
        </div>`
      );

      observationLayerGroupRef.current?.addLayer(marker);
    });
  }, [observations, onSelectObservation]);

  // Update Route Polyline Layer
  useEffect(() => {
    if (!mapInstanceRef.current || !routeLayerGroupRef.current) return;
    routeLayerGroupRef.current.clearLayers();

    if (routeCoordinates && routeCoordinates.length >= 2) {
      // Glow background line
      const glowLine = L.polyline(routeCoordinates, {
        color: '#06b6d4',
        weight: 12,
        opacity: 0.35,
        lineCap: 'round',
      });

      // Sharp foreground polyline
      const routeLine = L.polyline(routeCoordinates, {
        color: '#22d3ee',
        weight: 6,
        opacity: 0.95,
        dashArray: '8, 8',
      });

      // Start marker
      const startPoint = routeCoordinates[0];
      const startMarker = L.circleMarker(startPoint, {
        radius: 8,
        color: '#ffffff',
        fillColor: '#10b981',
        fillOpacity: 1,
        weight: 3,
      }).bindTooltip('<b>Route Origin</b>');

      // End marker
      const endPoint = routeCoordinates[routeCoordinates.length - 1];
      const endMarker = L.circleMarker(endPoint, {
        radius: 8,
        color: '#ffffff',
        fillColor: '#6366f1',
        fillOpacity: 1,
        weight: 3,
      }).bindTooltip('<b>Destination</b>');

      routeLayerGroupRef.current.addLayer(glowLine);
      routeLayerGroupRef.current.addLayer(routeLine);
      routeLayerGroupRef.current.addLayer(startMarker);
      routeLayerGroupRef.current.addLayer(endMarker);

      // Pan to route bounds
      mapInstanceRef.current.fitBounds(routeLine.getBounds(), { padding: [50, 50] });
    }
  }, [routeCoordinates]);

  return (
    <div className="relative w-full h-full min-h-[500px]">
      <div ref={mapContainerRef} className="w-full h-full rounded-xl overflow-hidden" />

      {/* Map Legend */}
      <div className="absolute bottom-6 left-6 z-[1000] bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-xl p-3 shadow-xl max-w-xs text-xs">
        <div className="font-semibold text-slate-200 mb-2 flex items-center justify-between">
          <span>Sidewalk Accessibility Grade</span>
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1.5 rounded-full bg-emerald-500"></span>
            <span className="text-slate-300">Grade A (90-100) Fully Accessible</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1.5 rounded-full bg-sky-500"></span>
            <span className="text-slate-300">Grade B (75-89) Minor Slope/Friction</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1.5 rounded-full bg-amber-500"></span>
            <span className="text-slate-300">Grade C (55-74) Moderate Roughness</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1.5 rounded-full bg-orange-500"></span>
            <span className="text-slate-300">Grade D (35-54) High Barrier Risk</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-1.5 rounded-full bg-rose-500"></span>
            <span className="text-slate-300">Grade F (0-34) Impassable / Stairs</span>
          </div>
        </div>

        <div className="mt-3 pt-2 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
          <span>Solid circle: Verified</span>
          <span>Dashed: Tentative</span>
        </div>
      </div>
    </div>
  );
};
