import type {
  PredictionResult,
  Alert,
  HealthStatus,
  OutageRecord,
  HistoricalQuery,
  RiskLevel,
} from './types';

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const config: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new ApiError(
        response.status,
        `API request failed: ${response.status} ${response.statusText}`,
        body
      );
    }

    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(0, `Network error: ${(error as Error).message}`);
  }
}

export async function getPredictions(
  region: string,
  params?: {
    min_risk?: number;
    risk_level?: RiskLevel;
    limit?: number;
    offset?: number;
  }
): Promise<PredictionResult[]> {
  const searchParams = new URLSearchParams();
  if (params?.min_risk !== undefined)
    searchParams.set('min_risk', String(params.min_risk));
  if (params?.risk_level) searchParams.set('risk_level', params.risk_level);
  if (params?.limit !== undefined)
    searchParams.set('limit', String(params.limit));
  if (params?.offset !== undefined)
    searchParams.set('offset', String(params.offset));

  const query = searchParams.toString();
  return request<PredictionResult[]>(
    `/predictions/${region}${query ? `?${query}` : ''}`
  );
}

export async function postRealtimePrediction(requestBody: {
  region: string;
  h3_cell: string;
  weather_data?: Record<string, number>;
  infrastructure_data?: Record<string, number>;
}): Promise<PredictionResult> {
  return request<PredictionResult>('/predictions/realtime', {
    method: 'POST',
    body: JSON.stringify(requestBody),
  });
}

export async function getActiveAlerts(
  region: string,
  severity?: RiskLevel
): Promise<Alert[]> {
  const params = new URLSearchParams();
  if (severity) params.set('severity', severity);
  const query = params.toString();
  return request<Alert[]>(
    `/alerts/active/${region}${query ? `?${query}` : ''}`
  );
}

export async function acknowledgeAlert(
  alertId: string,
  acknowledgedBy: string
): Promise<Alert> {
  return request<Alert>(`/alerts/${alertId}/acknowledge`, {
    method: 'POST',
    body: JSON.stringify({ acknowledged_by: acknowledgedBy }),
  });
}

export async function getHistoricalOutages(
  query: HistoricalQuery
): Promise<OutageRecord[]> {
  const params = new URLSearchParams();
  if (query.region) params.set('region', query.region);
  params.set('start_date', query.start_date);
  params.set('end_date', query.end_date);
  if (query.min_severity) params.set('min_severity', query.min_severity);
  if (query.limit !== undefined) params.set('limit', String(query.limit));
  if (query.offset !== undefined) params.set('offset', String(query.offset));

  return request<OutageRecord[]>(`/outages/historical?${params.toString()}`);
}

export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/health');
}

export { ApiError };
