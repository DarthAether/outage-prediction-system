'use client';

import { useState, useCallback } from 'react';
import {
  Sliders,
  Server,
  Database,
  Cpu,
  CheckCircle,
  XCircle,
  ArrowUpCircle,
  Clock,
  Activity,
  Shield,
} from 'lucide-react';
import Header from '@/components/Layout/Header';
import type { ModelInfo, RiskThresholds, HealthStatus, RiskLevel } from '@/lib/types';
import { RISK_COLORS } from '@/lib/types';

const MOCK_MODELS: ModelInfo[] = [
  {
    name: 'XGBoost Ensemble',
    version: 'v2.3.1',
    auc_roc: 0.934,
    f1_score: 0.891,
    precision: 0.912,
    recall: 0.871,
    status: 'active',
    deployed_at: '2025-03-10T08:00:00Z',
  },
  {
    name: 'LightGBM Stacked',
    version: 'v2.4.0-rc1',
    auc_roc: 0.941,
    f1_score: 0.898,
    precision: 0.905,
    recall: 0.891,
    status: 'shadow',
    deployed_at: '2025-03-20T12:00:00Z',
  },
  {
    name: 'Random Forest Baseline',
    version: 'v1.8.2',
    auc_roc: 0.897,
    f1_score: 0.842,
    precision: 0.878,
    recall: 0.809,
    status: 'retired',
    deployed_at: '2024-12-01T10:00:00Z',
  },
  {
    name: 'Neural Network Hybrid',
    version: 'v0.9.0-beta',
    auc_roc: 0.928,
    f1_score: 0.879,
    precision: 0.894,
    recall: 0.865,
    status: 'shadow',
    deployed_at: '2025-03-18T14:00:00Z',
  },
];

const MOCK_HEALTH: HealthStatus = {
  status: 'healthy',
  db_connected: true,
  redis_connected: true,
  model_loaded: true,
  active_models: ['XGBoost Ensemble v2.3.1'],
  uptime_seconds: 432000,
};

const statusBadge: Record<string, { bg: string; text: string; label: string }> = {
  active: { bg: 'bg-risk-green/15', text: 'text-risk-green', label: 'Active' },
  shadow: { bg: 'bg-risk-yellow/15', text: 'text-risk-yellow', label: 'Shadow' },
  retired: { bg: 'bg-gray-800', text: 'text-gray-500', label: 'Retired' },
};

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

export default function AdminPage() {
  const [region, setRegion] = useState('TX');
  const [thresholds, setThresholds] = useState<RiskThresholds>({
    green_max: 0.25,
    yellow_max: 0.5,
    orange_max: 0.75,
  });
  const [models] = useState<ModelInfo[]>(MOCK_MODELS);
  const [health] = useState<HealthStatus>(MOCK_HEALTH);
  const [promotingModel, setPromotingModel] = useState<string | null>(null);

  const handleThresholdChange = useCallback(
    (key: keyof RiskThresholds, value: number) => {
      setThresholds((prev) => {
        const updated = { ...prev, [key]: value };
        if (key === 'green_max' && value >= updated.yellow_max) {
          updated.yellow_max = Math.min(value + 0.05, updated.orange_max - 0.05);
        }
        if (key === 'yellow_max') {
          if (value <= updated.green_max) updated.green_max = Math.max(0, value - 0.05);
          if (value >= updated.orange_max) updated.orange_max = Math.min(1, value + 0.05);
        }
        if (key === 'orange_max' && value <= updated.yellow_max) {
          updated.yellow_max = Math.max(updated.green_max + 0.05, value - 0.05);
        }
        return updated;
      });
    },
    []
  );

  const handlePromote = useCallback((modelName: string) => {
    setPromotingModel(modelName);
    setTimeout(() => setPromotingModel(null), 2000);
  }, []);

  const thresholdConfig: {
    key: keyof RiskThresholds;
    label: string;
    from: string;
    toLevel: RiskLevel;
    color: string;
  }[] = [
    {
      key: 'green_max',
      label: 'GREEN upper bound',
      from: 'GREEN',
      toLevel: 'YELLOW',
      color: RISK_COLORS.GREEN,
    },
    {
      key: 'yellow_max',
      label: 'YELLOW upper bound',
      from: 'YELLOW',
      toLevel: 'ORANGE',
      color: RISK_COLORS.YELLOW,
    },
    {
      key: 'orange_max',
      label: 'ORANGE upper bound',
      from: 'ORANGE',
      toLevel: 'RED',
      color: RISK_COLORS.ORANGE,
    },
  ];

  return (
    <div className="flex h-full flex-col">
      <Header
        title="Administration"
        selectedRegion={region}
        onRegionChange={setRegion}
      />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Risk Thresholds */}
        <div className="card">
          <div className="flex items-center gap-2 mb-6">
            <Sliders className="h-4.5 w-4.5 text-accent" />
            <h3 className="text-sm font-semibold text-gray-300">
              Risk Level Thresholds
            </h3>
          </div>

          <div className="mb-6">
            <div className="flex h-8 w-full overflow-hidden rounded-lg">
              <div
                className="flex items-center justify-center text-xs font-bold text-white/80 transition-all"
                style={{
                  width: `${thresholds.green_max * 100}%`,
                  backgroundColor: RISK_COLORS.GREEN,
                }}
              >
                GREEN
              </div>
              <div
                className="flex items-center justify-center text-xs font-bold text-white/80 transition-all"
                style={{
                  width: `${(thresholds.yellow_max - thresholds.green_max) * 100}%`,
                  backgroundColor: RISK_COLORS.YELLOW,
                }}
              >
                YELLOW
              </div>
              <div
                className="flex items-center justify-center text-xs font-bold text-white/80 transition-all"
                style={{
                  width: `${(thresholds.orange_max - thresholds.yellow_max) * 100}%`,
                  backgroundColor: RISK_COLORS.ORANGE,
                }}
              >
                ORANGE
              </div>
              <div
                className="flex items-center justify-center text-xs font-bold text-white/80 transition-all"
                style={{
                  width: `${(1 - thresholds.orange_max) * 100}%`,
                  backgroundColor: RISK_COLORS.RED,
                }}
              >
                RED
              </div>
            </div>
            <div className="mt-1 flex justify-between text-xs text-gray-600 tabular-nums">
              <span>0.00</span>
              <span>0.25</span>
              <span>0.50</span>
              <span>0.75</span>
              <span>1.00</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-6">
            {thresholdConfig.map((cfg) => (
              <div key={cfg.key} className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-gray-400">
                    {cfg.label}
                  </label>
                  <span
                    className="text-sm font-bold tabular-nums"
                    style={{ color: cfg.color }}
                  >
                    {thresholds[cfg.key].toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  value={thresholds[cfg.key]}
                  onChange={(e) =>
                    handleThresholdChange(cfg.key, parseFloat(e.target.value))
                  }
                  className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, ${cfg.color} 0%, ${cfg.color} ${
                      ((thresholds[cfg.key] - 0.05) / 0.9) * 100
                    }%, #2a2d3e ${
                      ((thresholds[cfg.key] - 0.05) / 0.9) * 100
                    }%, #2a2d3e 100%)`,
                  }}
                />
                <p className="text-xs text-gray-600">
                  {cfg.from} &rarr; {cfg.toLevel} boundary
                </p>
              </div>
            ))}
          </div>

          <div className="mt-4 flex justify-end">
            <button className="btn-primary">Save Thresholds</button>
          </div>
        </div>

        {/* Model Comparison */}
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="h-4.5 w-4.5 text-accent" />
            <h3 className="text-sm font-semibold text-gray-300">
              Model Registry
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border">
                  <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Model
                  </th>
                  <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Version
                  </th>
                  <th className="pb-3 pr-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                    AUC-ROC
                  </th>
                  <th className="pb-3 pr-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                    F1 Score
                  </th>
                  <th className="pb-3 pr-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                    Precision
                  </th>
                  <th className="pb-3 pr-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                    Recall
                  </th>
                  <th className="pb-3 pr-4 text-center text-xs font-medium uppercase tracking-wider text-gray-500">
                    Status
                  </th>
                  <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Deployed
                  </th>
                  <th className="pb-3 text-center text-xs font-medium uppercase tracking-wider text-gray-500">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border/50">
                {models.map((model) => {
                  const badge = statusBadge[model.status];
                  return (
                    <tr
                      key={`${model.name}-${model.version}`}
                      className="transition-colors hover:bg-surface-overlay/30"
                    >
                      <td className="py-3 pr-4 font-medium text-gray-200">
                        {model.name}
                      </td>
                      <td className="py-3 pr-4">
                        <code className="text-xs font-mono text-gray-500">
                          {model.version}
                        </code>
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums text-gray-300">
                        <span
                          className={
                            model.auc_roc >= 0.93
                              ? 'text-risk-green'
                              : model.auc_roc >= 0.9
                                ? 'text-risk-yellow'
                                : 'text-gray-400'
                          }
                        >
                          {model.auc_roc.toFixed(3)}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums text-gray-400">
                        {model.f1_score.toFixed(3)}
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums text-gray-400">
                        {model.precision.toFixed(3)}
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums text-gray-400">
                        {model.recall.toFixed(3)}
                      </td>
                      <td className="py-3 pr-4 text-center">
                        <span className={`badge ${badge.bg} ${badge.text}`}>
                          {badge.label}
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-xs text-gray-500">
                        {new Date(model.deployed_at).toLocaleDateString(
                          'en-US',
                          { month: 'short', day: 'numeric', year: 'numeric' }
                        )}
                      </td>
                      <td className="py-3 text-center">
                        {model.status === 'shadow' && (
                          <button
                            onClick={() => handlePromote(model.name)}
                            disabled={promotingModel === model.name}
                            className="inline-flex items-center gap-1 rounded-md bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent transition-colors hover:bg-accent/20 disabled:opacity-50"
                          >
                            {promotingModel === model.name ? (
                              <>
                                <Activity className="h-3 w-3 animate-spin" />
                                Promoting...
                              </>
                            ) : (
                              <>
                                <ArrowUpCircle className="h-3 w-3" />
                                Promote
                              </>
                            )}
                          </button>
                        )}
                        {model.status === 'active' && (
                          <span className="text-xs text-risk-green/60">
                            Production
                          </span>
                        )}
                        {model.status === 'retired' && (
                          <span className="text-xs text-gray-600">
                            Archived
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* System Health */}
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Shield className="h-4.5 w-4.5 text-accent" />
            <h3 className="text-sm font-semibold text-gray-300">
              System Health
            </h3>
          </div>

          <div className="grid grid-cols-5 gap-4">
            <div className="rounded-lg border border-surface-border bg-surface p-4 text-center">
              <div
                className={`mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full ${
                  health.status === 'healthy'
                    ? 'bg-risk-green/15'
                    : health.status === 'degraded'
                      ? 'bg-risk-yellow/15'
                      : 'bg-risk-red/15'
                }`}
              >
                <Activity
                  className={`h-5 w-5 ${
                    health.status === 'healthy'
                      ? 'text-risk-green'
                      : health.status === 'degraded'
                        ? 'text-risk-yellow'
                        : 'text-risk-red'
                  }`}
                />
              </div>
              <p className="text-sm font-semibold capitalize text-gray-200">
                {health.status}
              </p>
              <p className="text-xs text-gray-500">Overall</p>
            </div>

            <div className="rounded-lg border border-surface-border bg-surface p-4 text-center">
              <div
                className={`mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full ${
                  health.db_connected ? 'bg-risk-green/15' : 'bg-risk-red/15'
                }`}
              >
                {health.db_connected ? (
                  <Database className="h-5 w-5 text-risk-green" />
                ) : (
                  <XCircle className="h-5 w-5 text-risk-red" />
                )}
              </div>
              <p className="text-sm font-semibold text-gray-200">
                {health.db_connected ? 'Connected' : 'Down'}
              </p>
              <p className="text-xs text-gray-500">Database</p>
            </div>

            <div className="rounded-lg border border-surface-border bg-surface p-4 text-center">
              <div
                className={`mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full ${
                  health.redis_connected ? 'bg-risk-green/15' : 'bg-risk-red/15'
                }`}
              >
                {health.redis_connected ? (
                  <Server className="h-5 w-5 text-risk-green" />
                ) : (
                  <XCircle className="h-5 w-5 text-risk-red" />
                )}
              </div>
              <p className="text-sm font-semibold text-gray-200">
                {health.redis_connected ? 'Connected' : 'Down'}
              </p>
              <p className="text-xs text-gray-500">Redis Cache</p>
            </div>

            <div className="rounded-lg border border-surface-border bg-surface p-4 text-center">
              <div
                className={`mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full ${
                  health.model_loaded ? 'bg-risk-green/15' : 'bg-risk-red/15'
                }`}
              >
                {health.model_loaded ? (
                  <CheckCircle className="h-5 w-5 text-risk-green" />
                ) : (
                  <XCircle className="h-5 w-5 text-risk-red" />
                )}
              </div>
              <p className="text-sm font-semibold text-gray-200">
                {health.model_loaded ? 'Loaded' : 'Not Loaded'}
              </p>
              <p className="text-xs text-gray-500">ML Model</p>
            </div>

            <div className="rounded-lg border border-surface-border bg-surface p-4 text-center">
              <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-accent/15">
                <Clock className="h-5 w-5 text-accent" />
              </div>
              <p className="text-sm font-semibold text-gray-200">
                {formatUptime(health.uptime_seconds)}
              </p>
              <p className="text-xs text-gray-500">Uptime</p>
            </div>
          </div>

          {health.active_models.length > 0 && (
            <div className="mt-4 rounded-lg border border-surface-border bg-surface p-3">
              <p className="mb-1 text-xs font-medium text-gray-500">
                Active Models
              </p>
              <div className="flex flex-wrap gap-2">
                {health.active_models.map((model) => (
                  <span
                    key={model}
                    className="inline-flex items-center gap-1 rounded-full bg-risk-green/10 px-2.5 py-1 text-xs font-medium text-risk-green"
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-risk-green animate-pulse-slow" />
                    {model}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
