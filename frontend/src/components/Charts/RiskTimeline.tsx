'use client';

import {
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceDot,
  ComposedChart,
} from 'recharts';

interface TimelineDataPoint {
  timestamp: string;
  risk: number;
  ci_lower: number;
  ci_upper: number;
  actual_outage?: boolean;
}

interface RiskTimelineProps {
  data: TimelineDataPoint[];
  title?: string;
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDate(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

interface TooltipPayloadEntry {
  value: number;
  dataKey: string;
  payload: TimelineDataPoint;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;

  const data = payload[0].payload;
  const risk = (data.risk * 100).toFixed(1);
  const lower = (data.ci_lower * 100).toFixed(1);
  const upper = (data.ci_upper * 100).toFixed(1);

  return (
    <div className="rounded-lg border border-surface-border bg-surface-overlay px-3 py-2 shadow-xl">
      <p className="mb-1 text-xs text-gray-500">{formatDate(data.timestamp)}</p>
      <p className="text-sm font-semibold text-white">Risk: {risk}%</p>
      <p className="text-xs text-gray-400">
        CI: {lower}% - {upper}%
      </p>
      {data.actual_outage && (
        <p className="mt-1 text-xs font-medium text-risk-red">
          Outage occurred
        </p>
      )}
    </div>
  );
}

export default function RiskTimeline({
  data,
  title = 'Risk Probability Over Time',
}: RiskTimelineProps) {
  const outagePoints = data.filter((d) => d.actual_outage);

  return (
    <div className="card space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">{title}</h3>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 5, right: 10, left: -10, bottom: 5 }}
          >
            <defs>
              <linearGradient id="uncertaintyGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#6366f1" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#2a2d3e"
              vertical={false}
            />

            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTime}
              stroke="#4a4d5e"
              tick={{ fontSize: 11, fill: '#6b7280' }}
              axisLine={{ stroke: '#2a2d3e' }}
            />

            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              stroke="#4a4d5e"
              tick={{ fontSize: 11, fill: '#6b7280' }}
              axisLine={{ stroke: '#2a2d3e' }}
            />

            <Tooltip content={<CustomTooltip />} />

            <Area
              dataKey="ci_upper"
              stroke="none"
              fill="url(#uncertaintyGrad)"
              fillOpacity={1}
              isAnimationActive={false}
            />
            <Area
              dataKey="ci_lower"
              stroke="none"
              fill="#0f1117"
              fillOpacity={1}
              isAnimationActive={false}
            />

            <Line
              type="monotone"
              dataKey="risk"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              activeDot={{
                r: 4,
                fill: '#6366f1',
                stroke: '#1c1f2e',
                strokeWidth: 2,
              }}
            />

            {outagePoints.map((point, i) => (
              <ReferenceDot
                key={i}
                x={point.timestamp}
                y={point.risk}
                r={5}
                fill="#ef4444"
                stroke="#1c1f2e"
                strokeWidth={2}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-accent" />
          Predicted Risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-4 rounded bg-accent/20" />
          Uncertainty Band
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-risk-red" />
          Actual Outage
        </span>
      </div>
    </div>
  );
}
