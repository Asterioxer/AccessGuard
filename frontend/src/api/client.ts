import {
  Observation,
  PedestrianSegment,
  RouteResponse,
  User,
  AnalyticsSummary
} from '../types';

const API_BASE = '/api/v1';

export function getAuthToken(): string | null {
  return localStorage.getItem('accessguard_token');
}

export function setAuthToken(token: string) {
  localStorage.setItem('accessguard_token', token);
}

export function clearAuthToken() {
  localStorage.removeItem('accessguard_token');
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export const api = {
  // Observations
  getObservations: (params?: { barrier_type?: string; status?: string; segment_id?: string }) => {
    const query = new URLSearchParams();
    if (params?.barrier_type) query.set('barrier_type', params.barrier_type);
    if (params?.status) query.set('status', params.status);
    if (params?.segment_id) query.set('segment_id', params.segment_id);
    const qs = query.toString() ? `?${query.toString()}` : '';
    return request<Observation[]>(`/observations${qs}`);
  },

  getObservation: (id: string) => request<Observation>(`/observations/${id}`),

  uploadObservation: async (formData: FormData) => {
    const token = getAuthToken();
    const headers: HeadersInit = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch(`${API_BASE}/observations/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }
    return response.json() as Promise<Observation>;
  },

  // Segments
  getSegments: () => request<PedestrianSegment[]>('/segments'),
  importOSMPedestrianNetwork: (bbox: [number, number, number, number]) => {
    const [min_lon, min_lat, max_lon, max_lat] = bbox;
    const query = new URLSearchParams({ min_lon: String(min_lon), min_lat: String(min_lat), max_lon: String(max_lon), max_lat: String(max_lat) });
    return request('/segments/import-osm?' + query.toString(), { method: 'POST' });
  },
  getSegment: (id: string) => request<PedestrianSegment>(`/segments/${id}`),
  recalculateSegment: (id: string) =>
    request<PedestrianSegment>(`/segments/${id}/recalculate`, { method: 'POST' }),

  // Routing
  calculateRoute: (data: {
    start: { latitude: number; longitude: number };
    destination: { latitude: number; longitude: number };
    wheelchair_profile?: string;
    avoid_unverified_barriers?: boolean;
    algorithm?: string;
    include_alternatives?: boolean;
  }) =>
    request<RouteResponse>('/routing/accessible-route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  // Verification
  submitVerification: (data: {
    observation_id: string;
    vote: 'confirm' | 'dispute' | 'reject';
    suggested_barrier_type?: string;
    notes?: string;
  }) =>
    request<{
      review_id: string;
      observation_id: string;
      new_status: string;
      verification_score: number;
      consensus_reached: boolean;
      status_changed: boolean;
    }>('/verifications/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  getVerificationQueue: () => request<Observation[]>('/verifications/queue'),

  // Analytics
  getAnalytics: () => request<AnalyticsSummary>('/analytics/summary'),

  // Auth
  login: async (email: string, password: string) => {
    const data = await request<{
      access_token: string;
      token_type: string;
      user_id: string;
      email: string;
      role: string;
    }>('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    setAuthToken(data.access_token);
    return data;
  },

  register: async (email: string, password: string, role: string = 'contributor') => {
    return request<User>('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, role }),
    });
  },

  getProfile: () => request<User>('/auth/me'),

  // Live CV & Adapters
  listAdapters: () =>
    request<{
      available_adapters: string[];
      active_adapter: string;
      onnx_available: boolean;
    }>('/cv/adapters'),

  setActiveAdapter: (adapter_name: string) =>
    request<{ status: string; active_adapter: string }>('/cv/active-adapter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ adapter_name }),
    }),

  postLiveFrame: async (data: {
    base64_image?: string;
    stream_id?: string;
    timestamp?: number;
  }) => {
    const formData = new FormData();
    if (data.base64_image) formData.append('base64_image', data.base64_image);
    if (data.stream_id) formData.append('stream_id', data.stream_id);
    if (data.timestamp) formData.append('timestamp', data.timestamp.toString());

    const token = getAuthToken();
    const headers: HeadersInit = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE}/cv/live-frame`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Live inference failed: ${response.status}`);
    }
    return response.json();
  },

  // Mission 005: Autonomous Ingestion
  getProviders: () => request<any[]>('/ingestion/providers'),

  getIngestionJobs: () => request<any[]>('/ingestion/jobs'),

  createIngestionJob: (data: {
    provider: string;
    bbox: [number, number, number, number];
    interval_minutes?: number;
    priority?: string;
  }) =>
    request<any>('/ingestion/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  runIngestionJob: (jobId: string) =>
    request<any>(`/ingestion/jobs/${jobId}/run`, {
      method: 'POST',
    }),

  queryProviderImagery: (data: {
    provider: string;
    bbox: [number, number, number, number];
    limit?: number;
  }) =>
    request<any>('/ingestion/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  getObservationProvenance: (obsId: string) =>
    request<any>(`/ingestion/provenance/${obsId}`),

  // Mission 005: Pixel-Level Segmentation
  getSegmentationClasses: () => request<{ classes: any[]; total_classes: number }>('/cv/segmentation/classes'),

  predictSegmentation: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const token = getAuthToken();
    const headers: HeadersInit = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}/cv/segmentation/predict`, {
      method: 'POST',
      headers,
      body: formData,
    });
    if (!res.ok) throw new Error(`Segmentation failed: ${res.status}`);
    return res.json();
  },

  // Mission 005: Video Intelligence
  uploadVideo: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const token = getAuthToken();
    const headers: HeadersInit = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}/video/process`, {
      method: 'POST',
      headers,
      body: formData,
    });
    if (!res.ok) throw new Error(`Video upload failed: ${res.status}`);
    return res.json();
  },

  getVideoJob: (jobId: string) => request<any>(`/video/jobs/${jobId}`),

  // Mission 005 Addendum: Reality Test Engine
  getRealityScorecard: () => request<any>('/reality/scorecard'),

  triggerRealityEvaluation: () =>
    request<any>('/reality/evaluate', {
      method: 'POST',
    }),

  getRealityFailures: () => request<any>('/reality/failures'),
};
