export type RiskLevel = 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED';

export interface UncertaintyEstimate {
  lower: number;
  upper: number;
  aleatoric: number;
  epistemic: number;
  confidence_level: number;
}

export interface PredictionResult {
  prediction_id: string;
  h3_cell: string;
  region: string;
  risk_probability: number;
  uncertainty: UncertaintyEstimate;
  risk_level: RiskLevel;
  model_version: string;
  top_features: FeatureContribution[];
  computed_at: string;
}

export interface FeatureContribution {
  name: string;
  value: number;
  direction: 'up' | 'down';
}

export interface Alert {
  id: string;
  severity: RiskLevel;
  region_code: string;
  h3_cell: string;
  risk_probability: number;
  uncertainty_range: [number, number];
  description: string;
  recommended_actions: string[];
  created_at: string;
  expires_at: string;
  acknowledged: boolean;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  db_connected: boolean;
  redis_connected: boolean;
  model_loaded: boolean;
  active_models: string[];
  uptime_seconds: number;
}

export interface OutageRecord {
  id: string;
  region: string;
  h3_cell: string;
  started_at: string;
  resolved_at: string | null;
  duration_minutes: number | null;
  customers_affected: number;
  cause: string;
  predicted_risk: number | null;
  was_predicted: boolean;
}

export interface HistoricalQuery {
  region?: string;
  start_date: string;
  end_date: string;
  min_severity?: RiskLevel;
  limit?: number;
  offset?: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  auc_roc: number;
  f1_score: number;
  precision: number;
  recall: number;
  status: 'active' | 'shadow' | 'retired';
  deployed_at: string;
}

export interface RiskThresholds {
  green_max: number;
  yellow_max: number;
  orange_max: number;
}

export interface RegionOption {
  code: string;
  name: string;
}

export const REGIONS: RegionOption[] = [
  { code: 'TX', name: 'Texas' },
  { code: 'CA', name: 'California' },
  { code: 'FL', name: 'Florida' },
];

export const RISK_COLORS: Record<RiskLevel, string> = {
  GREEN: '#22c55e',
  YELLOW: '#eab308',
  ORANGE: '#f97316',
  RED: '#ef4444',
};

export const RISK_BG_COLORS: Record<RiskLevel, string> = {
  GREEN: 'rgba(34, 197, 94, 0.15)',
  YELLOW: 'rgba(234, 179, 8, 0.15)',
  ORANGE: 'rgba(249, 115, 22, 0.15)',
  RED: 'rgba(239, 68, 68, 0.15)',
};
