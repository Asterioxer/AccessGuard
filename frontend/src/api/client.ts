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
};
