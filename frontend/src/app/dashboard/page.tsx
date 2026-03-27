'use client';

import { useState, useCallback } from 'react';
import {
  AlertTriangle,
  Shield,
  TrendingUp,
  MapPin,
  Activity,
} from 'lucide-react';
import Header from '@/components/Layout/Header';
import AlertFeed from '@/components/Alerts/AlertFeed';
import RiskTimeline from '@/components/Charts/RiskTimeline';
import FeatureImportance from '@/components/Charts/FeatureImportance';
import type { Alert, PredictionResult, RiskLevel, FeatureContribution } from '@/lib/types';
import { RISK_COLORS } from '@/lib/types';

// --- Mock Data ---

const MOCK_PREDICTIONS: PredictionResult[] = [
  {
    prediction_id: 'pred-001',
    h3_cell: '8a2a1072b59ffff',
    region: 'TX',
    risk_probability: 0.87,
    uncertainty: { lower: 0.79, upper: 0.94, aleatoric: 0.05, epistemic: 0.03, confidence_level: 0.95 },
    risk_level: 'RED',
    model_version: 'v2.3.1',
    top_features: [
      { name: 'wind_speed_sustained', value: 0.234, direction: 'up' },
      { name: 'tree_density_proximity', value: 0.189, direction: 'up' },
      { name: 'infrastructure_age_years', value: 0.156, direction: 'up' },
    ],
    computed_at: new Date(Date.now() - 120000).toISOString(),
  },
  {
    prediction_id: 'pred-002',
    h3_cell: '8a2a1072b5bffff',
    region: 'TX',
    risk_probability: 0.72,
    uncertainty: { lower: 0.63, upper: 0.81, aleatoric: 0.06, epistemic: 0.04, confidence_level: 0.95 },
    risk_level: 'ORANGE',
    model_version: 'v2.3.1',
    top_features: [
      { name: 'precipitation_rate_mm', value: 0.201, direction: 'up' },
      { name: 'soil_moisture_index', value: 0.178, direction: 'up' },
      { name: 'grid_redundancy_score', value: 0.145, direction: 'down' },
    ],
    computed_at: new Date(Date.now() - 180000).toISOString(),
  },
  {
    prediction_id: 'pred-003',
    h3_cell: '8a2a1072b4fffff',
    region: 'TX',
    risk_probability: 0.61,
    uncertainty: { lower: 0.52, upper: 0.70, aleatoric: 0.04, epistemic: 0.05, confidence_level: 0.95 },
    risk_level: 'ORANGE',
    model_version: 'v2.3.1',
    top_features: [
      { name: 'temperature_delta_24h', value: 0.167, direction: 'up' },
      { name: 'demand_load_ratio', value: 0.143, direction: 'up' },
      { name: 'maintenance_recency_days', value: 0.121, direction: 'up' },
    ],
    computed_at: new Date(Date.now() - 240000).toISOString(),
  },
  {
    prediction_id: 'pred-004',
    h3_cell: '8a2a1072b6fffff',
    region: 'TX',
    risk_probability: 0.45,
    uncertainty: { lower: 0.36, upper: 0.54, aleatoric: 0.05, epistemic: 0.03, confidence_level: 0.95 },
    risk_level: 'YELLOW',
    model_version: 'v2.3.1',
    top_features: [
      { name: 'wind_gust_max', value: 0.134, direction: 'up' },
      { name: 'equipment_failure_rate', value: 0.112, direction: 'up' },
      { name: 'underground_cable_pct', value: 0.098, direction: 'down' },
    ],
    computed_at: new Date(Date.now() - 300000).toISOString(),
  },
  {
    prediction_id: 'pred-005',
    h3_cell: '8a2a1072b53ffff',
    region: 'TX',
    risk_probability: 0.38,
    uncertainty: { lower: 0.29, upper: 0.47, aleatoric: 0.04, epistemic: 0.04, confidence_level: 0.95 },
    risk_level: 'YELLOW',
    model_version: 'v2.3.1',
    top_features: [
      { name: 'humidity_pct', value: 0.098, direction: 'up' },
      { name: 'vegetation_health_ndvi', value: 0.087, direction: 'down' },
      { name: 'crew_availability_index', value: 0.076, direction: 'down' },
    ],
    computed_at: new Date(Date.now() - 360000).toISOString(),
  },
  {
    prediction_id: 'pred-006',
    h3_cell: '8a2a1072b51ffff',
    region: 'TX',
    risk_probability: 0.15,
    uncertainty: { lower: 0.08, upper: 0.22, aleatoric: 0.03, epistemic: 0.02, confidence_level: 0.95 },
    risk_level: 'GREEN',
    model_version: 'v2.3.1',
    top_features: [
      { name: 'wind_speed_sustained', value: 0.045, direction: 'down' },
      { name: 'grid_redundancy_score', value: 0.089, direction: 'down' },
      { name: 'recent_maintenance', value: 0.067, direction: 'down' },
    ],
    computed_at: new Date(Date.now() - 420000).toISOString(),
  },
];

const MOCK_ALERTS: Alert[] = [
  {
    id: 'alert-001',
    severity: 'RED',
    region_code: 'TX',
    h3_cell: '8a2a1072b59ffff',
    risk_probability: 0.87,
    uncertainty_range: [0.79, 0.94],
    description: 'Critical outage risk detected in Houston metro area. Sustained winds exceeding 65 mph with aging infrastructure in sector.',
    recommended_actions: [
      'Deploy emergency response crews to sector H-14',
      'Pre-position mobile generators at critical facilities',
      'Issue public advisory for affected zip codes',
      'Activate mutual aid agreements with neighboring utilities',
    ],
    created_at: new Date(Date.now() - 300000).toISOString(),
    expires_at: new Date(Date.now() + 3600000).toISOString(),
    acknowledged: false,
  },
  {
    id: 'alert-002',
    severity: 'ORANGE',
    region_code: 'TX',
    h3_cell: '8a2a1072b5bffff',
    risk_probability: 0.72,
    uncertainty_range: [0.63, 0.81],
    description: 'Elevated risk in Dallas-Fort Worth corridor. Heavy precipitation and saturated soil conditions increasing tree-fall probability.',
    recommended_actions: [
      'Increase patrol frequency along major distribution lines',
      'Clear known hazard trees in high-risk corridors',
      'Verify backup power at hospitals and emergency services',
    ],
    created_at: new Date(Date.now() - 600000).toISOString(),
    expires_at: new Date(Date.now() + 7200000).toISOString(),
    acknowledged: false,
  },
  {
    id: 'alert-003',
    severity: 'ORANGE',
    region_code: 'TX',
    h3_cell: '8a2a1072b4fffff',
    risk_probability: 0.61,
    uncertainty_range: [0.52, 0.70],
    description: 'Moderate-high risk in Austin area. Rapid temperature swing causing demand surge with limited reserve margin.',
    recommended_actions: [
      'Request voluntary conservation from commercial customers',
      'Prepare load shedding protocols for non-critical circuits',
    ],
    created_at: new Date(Date.now() - 900000).toISOString(),
    expires_at: new Date(Date.now() + 5400000).toISOString(),
    acknowledged: true,
  },
  {
    id: 'alert-004',
    severity: 'YELLOW',
    region_code: 'TX',
    h3_cell: '8a2a1072b6fffff',
    risk_probability: 0.45,
    uncertainty_range: [0.36, 0.54],
    description: 'Watch condition in San Antonio region. Wind gusts approaching threshold with some equipment nearing maintenance windows.',
    recommended_actions: [
      'Monitor weather radar for deteriorating conditions',
      'Review staffing levels for evening shift',
    ],
    created_at: new Date(Date.now() - 1800000).toISOString(),
    expires_at: new Date(Date.now() + 10800000).toISOString(),
    acknowledged: false,
  },
];

const MOCK_TIMELINE = Array.from({ length: 24 }, (_, i) => {
  const time = new Date(Date.now() - (23 - i) * 3600000);
  const baseRisk = 0.3 + 0.15 * Math.sin(i / 4) + 0.1 * Math.sin(i / 2);
  const risk = Math.min(0.95, Math.max(0.05, baseRisk + (Math.random() - 0.5) * 0.1));
  const spread = 0.06 + Math.random() * 0.04;
  return {
    timestamp: time.toISOString(),
    risk,
    ci_lower: Math.max(0, risk - spread),
    ci_upper: Math.min(1, risk + spread),
    actual_outage: i === 18,
  };
});

const MOCK_FEATURES: FeatureContribution[] = [
  { name: 'wind_speed_sustained', value: 0.234, direction: 'up' as const },
  { name: 'tree_density_proximity', value: 0.189, direction: 'up' as const },
  { name: 'infrastructure_age_years', value: 0.156, direction: 'up' as const },
  { name: 'precipitation_rate_mm', value: 0.134, direction: 'up' as const },
  { name: 'grid_redundancy_score', value: 0.121, direction: 'down' as const },
  { name: 'soil_moisture_index', value: 0.109, direction: 'up' as const },
  { name: 'demand_load_ratio', value: 0.098, direction: 'up' as const },
  { name: 'underground_cable_pct', value: 0.087, direction: 'down' as const },
  { name: 'recent_maintenance', value: 0.076, direction: 'down' as const },
  { name: 'temperature_delta_24h', value: 0.065, direction: 'up' as const },
];

// --- Component ---

const riskLevelBg: Record<RiskLevel, string> = {
  GREEN: 'bg-risk-green/10 border-risk-green/30',
  YELLOW: 'bg-risk-yellow/10 border-risk-yellow/30',
  ORANGE: 'bg-risk-orange/10 border-risk-orange/30',
  RED: 'bg-risk-red/10 border-risk-red/30',
};

const riskLevelText: Record<RiskLevel, string> = {
  GREEN: 'text-risk-green',
  YELLOW: 'text-risk-yellow',
  ORANGE: 'text-risk-orange',
  RED: 'text-risk-red',
};

export default function DashboardPage() {
  const [region, setRegion] = useState('TX');
  const [alerts, setAlerts] = useState<Alert[]>(MOCK_ALERTS);
  const predictions = MOCK_PREDICTIONS;

  const handleAcknowledge = useCallback((alertId: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a))
    );
  }, []);

  const cellsAtRisk = predictions.filter(
    (p) => p.risk_level !== 'GREEN'
  ).length;
  const activeAlertCount = alerts.filter((a) => !a.acknowledged).length;
  const avgConfidence =
    predictions.reduce((acc, p) => acc + p.uncertainty.confidence_level, 0) /
    predictions.length;

  const riskDistribution = predictions.reduce(
    (acc, p) => {
      acc[p.risk_level] = (acc[p.risk_level] || 0) + 1;
      return acc;
    },
    {} as Record<RiskLevel, number>
  );

  return (
    <div className="flex h-full flex-col">
      <Header
        title="Dashboard"
        selectedRegion={region}
        onRegionChange={setRegion}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Main content area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Summary stats row */}
          <div className="grid grid-cols-4 gap-4">
            <div className="card flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-risk-red/15">
                <AlertTriangle className="h-5 w-5 text-risk-red" />
              </div>
              <div>
                <p className="stat-value text-white">{activeAlertCount}</p>
                <p className="stat-label">Active Alerts</p>
              </div>
            </div>

            <div className="card flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-risk-orange/15">
                <MapPin className="h-5 w-5 text-risk-orange" />
              </div>
              <div>
                <p className="stat-value text-white">{cellsAtRisk}</p>
                <p className="stat-label">Cells at Risk</p>
              </div>
            </div>

            <div className="card flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15">
                <Shield className="h-5 w-5 text-accent" />
              </div>
              <div>
                <p className="stat-value text-white">
                  {(avgConfidence * 100).toFixed(0)}%
                </p>
                <p className="stat-label">Avg Confidence</p>
              </div>
            </div>

            <div className="card flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-risk-green/15">
                <TrendingUp className="h-5 w-5 text-risk-green" />
              </div>
              <div>
                <p className="stat-value text-white">
                  {predictions.length}
                </p>
                <p className="stat-label">Monitored Cells</p>
              </div>
            </div>
          </div>

          {/* Map placeholder + risk cards grid */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-300">
                Regional Risk Overview
              </h3>
              <div className="flex items-center gap-3">
                {(['GREEN', 'YELLOW', 'ORANGE', 'RED'] as RiskLevel[]).map(
                  (level) => (
                    <span
                      key={level}
                      className="flex items-center gap-1 text-xs text-gray-500"
                    >
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: RISK_COLORS[level] }}
                      />
                      {riskDistribution[level] || 0}
                    </span>
                  )
                )}
              </div>
            </div>

            <div
              id="map-container"
              className="relative mb-4 flex h-48 items-center justify-center rounded-lg border border-dashed border-surface-border bg-surface/50"
            >
              <div className="text-center">
                <Activity className="mx-auto mb-2 h-8 w-8 text-gray-600" />
                <p className="text-sm text-gray-500">
                  Interactive map requires Mapbox token
                </p>
                <p className="text-xs text-gray-600">
                  Set NEXT_PUBLIC_MAPBOX_TOKEN to enable deck.gl visualization
                </p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {predictions.map((pred) => (
                <div
                  key={pred.prediction_id}
                  className={`rounded-lg border p-3 transition-all hover:shadow-md ${riskLevelBg[pred.risk_level]}`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <code className="text-xs text-gray-500 font-mono">
                      {pred.h3_cell.slice(0, 12)}...
                    </code>
                    <span
                      className={`text-xs font-bold ${riskLevelText[pred.risk_level]}`}
                    >
                      {pred.risk_level}
                    </span>
                  </div>

                  <div className="mb-2">
                    <span
                      className={`text-2xl font-bold tabular-nums ${riskLevelText[pred.risk_level]}`}
                    >
                      {(pred.risk_probability * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Risk bar */}
                  <div className="mb-2 h-1.5 w-full rounded-full bg-surface">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${pred.risk_probability * 100}%`,
                        backgroundColor: RISK_COLORS[pred.risk_level],
                      }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>
                      CI: {(pred.uncertainty.lower * 100).toFixed(0)}-
                      {(pred.uncertainty.upper * 100).toFixed(0)}%
                    </span>
                    <span>{pred.model_version}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-2 gap-4">
            <RiskTimeline data={MOCK_TIMELINE} />
            <FeatureImportance features={MOCK_FEATURES} />
          </div>
        </div>

        {/* Sidebar: Alert feed */}
        <div className="w-96 shrink-0 border-l border-surface-border bg-surface-raised p-4 overflow-hidden flex flex-col">
          <AlertFeed alerts={alerts} onAcknowledge={handleAcknowledge} />
        </div>
      </div>
    </div>
  );
}
