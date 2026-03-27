'use client';

import { useState, useMemo } from 'react';
import {
  Calendar,
  Download,
  Search,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Clock,
  Users,
  Target,
} from 'lucide-react';
import Header from '@/components/Layout/Header';
import RiskTimeline from '@/components/Charts/RiskTimeline';
import type { RiskLevel, OutageRecord } from '@/lib/types';
import { RISK_COLORS } from '@/lib/types';

const MOCK_OUTAGES: OutageRecord[] = [
  {
    id: 'out-001',
    region: 'TX',
    h3_cell: '8a2a1072b59ffff',
    started_at: '2025-03-15T14:30:00Z',
    resolved_at: '2025-03-15T18:45:00Z',
    duration_minutes: 255,
    customers_affected: 12400,
    cause: 'Severe thunderstorm - tree contact',
    predicted_risk: 0.82,
    was_predicted: true,
  },
  {
    id: 'out-002',
    region: 'TX',
    h3_cell: '8a2a1072b5bffff',
    started_at: '2025-03-14T09:15:00Z',
    resolved_at: '2025-03-14T11:30:00Z',
    duration_minutes: 135,
    customers_affected: 5600,
    cause: 'Equipment failure - transformer',
    predicted_risk: 0.45,
    was_predicted: false,
  },
  {
    id: 'out-003',
    region: 'TX',
    h3_cell: '8a2a1072b4fffff',
    started_at: '2025-03-12T22:00:00Z',
    resolved_at: '2025-03-13T06:30:00Z',
    duration_minutes: 510,
    customers_affected: 28900,
    cause: 'Ice storm - line galloping',
    predicted_risk: 0.91,
    was_predicted: true,
  },
  {
    id: 'out-004',
    region: 'TX',
    h3_cell: '8a2a1072b6fffff',
    started_at: '2025-03-10T16:45:00Z',
    resolved_at: '2025-03-10T19:00:00Z',
    duration_minutes: 135,
    customers_affected: 3200,
    cause: 'Vehicle accident - pole damage',
    predicted_risk: null,
    was_predicted: false,
  },
  {
    id: 'out-005',
    region: 'TX',
    h3_cell: '8a2a1072b53ffff',
    started_at: '2025-03-08T11:20:00Z',
    resolved_at: '2025-03-08T13:45:00Z',
    duration_minutes: 145,
    customers_affected: 7800,
    cause: 'High winds - conductor slap',
    predicted_risk: 0.67,
    was_predicted: true,
  },
  {
    id: 'out-006',
    region: 'TX',
    h3_cell: '8a2a1072b51ffff',
    started_at: '2025-03-05T03:10:00Z',
    resolved_at: '2025-03-05T04:50:00Z',
    duration_minutes: 100,
    customers_affected: 2100,
    cause: 'Overload - demand surge',
    predicted_risk: 0.73,
    was_predicted: true,
  },
  {
    id: 'out-007',
    region: 'TX',
    h3_cell: '8a2a1072b5dffff',
    started_at: '2025-03-03T19:30:00Z',
    resolved_at: '2025-03-04T02:15:00Z',
    duration_minutes: 405,
    customers_affected: 18500,
    cause: 'Tornado damage - multiple structures',
    predicted_risk: 0.88,
    was_predicted: true,
  },
  {
    id: 'out-008',
    region: 'TX',
    h3_cell: '8a2a1072b55ffff',
    started_at: '2025-03-01T08:00:00Z',
    resolved_at: '2025-03-01T09:20:00Z',
    duration_minutes: 80,
    customers_affected: 1500,
    cause: 'Animal contact - substation',
    predicted_risk: 0.12,
    was_predicted: false,
  },
];

const MOCK_HISTORICAL_TIMELINE = Array.from({ length: 30 }, (_, i) => {
  const time = new Date(Date.now() - (29 - i) * 86400000);
  const baseRisk = 0.35 + 0.2 * Math.sin(i / 5) + 0.1 * Math.sin(i / 3);
  const risk = Math.min(0.95, Math.max(0.05, baseRisk + (Math.random() - 0.5) * 0.1));
  const spread = 0.08 + Math.random() * 0.05;
  return {
    timestamp: time.toISOString(),
    risk,
    ci_lower: Math.max(0, risk - spread),
    ci_upper: Math.min(1, risk + spread),
    actual_outage: [5, 12, 18, 24].includes(i),
  };
});

function getRiskLevel(prob: number | null): RiskLevel {
  if (prob === null) return 'GREEN';
  if (prob >= 0.75) return 'RED';
  if (prob >= 0.5) return 'ORANGE';
  if (prob >= 0.25) return 'YELLOW';
  return 'GREEN';
}

function formatDuration(minutes: number | null): string {
  if (minutes === null) return 'Ongoing';
  if (minutes < 60) return `${minutes}m`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function formatNumber(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

export default function HistoricalPage() {
  const [region, setRegion] = useState('TX');
  const [startDate, setStartDate] = useState('2025-03-01');
  const [endDate, setEndDate] = useState('2025-03-31');
  const [page, setPage] = useState(0);
  const pageSize = 5;

  const outages = MOCK_OUTAGES;
  const paginatedOutages = outages.slice(
    page * pageSize,
    (page + 1) * pageSize
  );
  const totalPages = Math.ceil(outages.length / pageSize);

  const stats = useMemo(() => {
    const totalOutages = outages.length;
    const totalCustomers = outages.reduce(
      (sum, o) => sum + o.customers_affected,
      0
    );
    const avgDuration =
      outages.reduce((sum, o) => sum + (o.duration_minutes || 0), 0) /
      totalOutages;
    const predictionRate =
      outages.filter((o) => o.was_predicted).length / totalOutages;
    return { totalOutages, totalCustomers, avgDuration, predictionRate };
  }, [outages]);

  return (
    <div className="flex h-full flex-col">
      <Header
        title="Historical Analysis"
        selectedRegion={region}
        onRegionChange={setRegion}
      />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Filters row */}
        <div className="card">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-gray-500" />
              <label className="text-sm text-gray-400">From</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="input-field w-40"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400">To</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="input-field w-40"
              />
            </div>
            <button className="btn-primary flex items-center gap-2">
              <Search className="h-3.5 w-3.5" />
              Query
            </button>
            <div className="flex-1" />
            <button className="btn-secondary flex items-center gap-2">
              <Download className="h-3.5 w-3.5" />
              Export CSV
            </button>
          </div>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-4 gap-4">
          <div className="card flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15">
              <BarChart3 className="h-5 w-5 text-accent" />
            </div>
            <div>
              <p className="stat-value text-white">{stats.totalOutages}</p>
              <p className="stat-label">Total Outages</p>
            </div>
          </div>

          <div className="card flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-risk-orange/15">
              <Users className="h-5 w-5 text-risk-orange" />
            </div>
            <div>
              <p className="stat-value text-white">
                {formatNumber(stats.totalCustomers)}
              </p>
              <p className="stat-label">Customers Affected</p>
            </div>
          </div>

          <div className="card flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-risk-yellow/15">
              <Clock className="h-5 w-5 text-risk-yellow" />
            </div>
            <div>
              <p className="stat-value text-white">
                {formatDuration(Math.round(stats.avgDuration))}
              </p>
              <p className="stat-label">Avg Duration</p>
            </div>
          </div>

          <div className="card flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-risk-green/15">
              <Target className="h-5 w-5 text-risk-green" />
            </div>
            <div>
              <p className="stat-value text-white">
                {(stats.predictionRate * 100).toFixed(0)}%
              </p>
              <p className="stat-label">Prediction Rate</p>
            </div>
          </div>
        </div>

        {/* Timeline chart */}
        <RiskTimeline
          data={MOCK_HISTORICAL_TIMELINE}
          title="30-Day Risk Trend"
        />

        {/* Outages table */}
        <div className="card">
          <h3 className="mb-4 text-sm font-semibold text-gray-300">
            Past Outages
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border">
                  <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Date
                  </th>
                  <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    H3 Cell
                  </th>
                  <th className="pb-3 pr-4 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Cause
                  </th>
                  <th className="pb-3 pr-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                    Duration
                  </th>
                  <th className="pb-3 pr-4 text-right text-xs font-medium uppercase tracking-wider text-gray-500">
                    Affected
                  </th>
                  <th className="pb-3 pr-4 text-center text-xs font-medium uppercase tracking-wider text-gray-500">
                    Predicted Risk
                  </th>
                  <th className="pb-3 text-center text-xs font-medium uppercase tracking-wider text-gray-500">
                    Predicted?
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border/50">
                {paginatedOutages.map((outage) => {
                  const riskLevel = getRiskLevel(outage.predicted_risk);
                  return (
                    <tr
                      key={outage.id}
                      className="transition-colors hover:bg-surface-overlay/30"
                    >
                      <td className="py-3 pr-4 text-gray-300">
                        {new Date(outage.started_at).toLocaleDateString(
                          'en-US',
                          {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          }
                        )}
                      </td>
                      <td className="py-3 pr-4">
                        <code className="text-xs font-mono text-gray-500">
                          {outage.h3_cell.slice(0, 12)}...
                        </code>
                      </td>
                      <td className="py-3 pr-4 text-gray-300 max-w-xs truncate">
                        {outage.cause}
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums text-gray-400">
                        {formatDuration(outage.duration_minutes)}
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums text-gray-400">
                        {outage.customers_affected.toLocaleString()}
                      </td>
                      <td className="py-3 pr-4 text-center">
                        {outage.predicted_risk !== null ? (
                          <span
                            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                            style={{
                              backgroundColor: `${RISK_COLORS[riskLevel]}20`,
                              color: RISK_COLORS[riskLevel],
                            }}
                          >
                            <span
                              className="h-1.5 w-1.5 rounded-full"
                              style={{
                                backgroundColor: RISK_COLORS[riskLevel],
                              }}
                            />
                            {(outage.predicted_risk * 100).toFixed(0)}%
                          </span>
                        ) : (
                          <span className="text-xs text-gray-600">N/A</span>
                        )}
                      </td>
                      <td className="py-3 text-center">
                        {outage.was_predicted ? (
                          <span className="badge-green">Yes</span>
                        ) : (
                          <span className="badge bg-gray-800 text-gray-500">
                            No
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="mt-4 flex items-center justify-between border-t border-surface-border pt-4">
            <p className="text-xs text-gray-500">
              Showing {page * pageSize + 1}-
              {Math.min((page + 1) * pageSize, outages.length)} of{' '}
              {outages.length} outages
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(Math.max(0, page - 1))}
                disabled={page === 0}
                className="rounded-md p-1.5 text-gray-500 transition-colors hover:bg-surface-overlay hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              {Array.from({ length: totalPages }, (_, i) => (
                <button
                  key={i}
                  onClick={() => setPage(i)}
                  className={`h-7 w-7 rounded-md text-xs font-medium transition-colors ${
                    i === page
                      ? 'bg-accent text-white'
                      : 'text-gray-500 hover:bg-surface-overlay hover:text-white'
                  }`}
                >
                  {i + 1}
                </button>
              ))}
              <button
                onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                disabled={page >= totalPages - 1}
                className="rounded-md p-1.5 text-gray-500 transition-colors hover:bg-surface-overlay hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
