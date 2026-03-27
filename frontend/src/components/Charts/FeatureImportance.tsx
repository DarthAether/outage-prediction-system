'use client';

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';

interface Feature {
  name: string;
  value: number;
  direction: 'up' | 'down';
}

interface FeatureImportanceProps {
  features: Feature[];
  title?: string;
  maxFeatures?: number;
}

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .slice(0, 25);
}

interface TooltipPayloadEntry {
  value: number;
  payload: Feature;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;

  const data = payload[0].payload;
  return (
    <div className="rounded-lg border border-surface-border bg-surface-overlay px-3 py-2 shadow-xl">
      <p className="mb-1 text-sm font-medium text-white">
        {formatFeatureName(data.name)}
      </p>
      <p className="text-xs text-gray-400">
        Importance: {Math.abs(data.value).toFixed(4)}
      </p>
      <p
        className={`text-xs font-medium ${
          data.direction === 'up' ? 'text-risk-red' : 'text-risk-green'
        }`}
      >
        {data.direction === 'up' ? 'Increases' : 'Decreases'} risk
      </p>
    </div>
  );
}

export default function FeatureImportance({
  features,
  title = 'Top Feature Contributions',
  maxFeatures = 10,
}: FeatureImportanceProps) {
  const sortedFeatures = [...features]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, maxFeatures)
    .reverse();

  return (
    <div className="card space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">{title}</h3>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedFeatures}
            layout="vertical"
            margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#2a2d3e"
              horizontal={false}
            />

            <XAxis
              type="number"
              stroke="#4a4d5e"
              tick={{ fontSize: 11, fill: '#6b7280' }}
              axisLine={{ stroke: '#2a2d3e' }}
              tickFormatter={(v: number) => Math.abs(v).toFixed(3)}
            />

            <YAxis
              dataKey="name"
              type="category"
              width={130}
              stroke="#4a4d5e"
              tick={{ fontSize: 11, fill: '#9ca3af' }}
              axisLine={{ stroke: '#2a2d3e' }}
              tickFormatter={formatFeatureName}
            />

            <Tooltip content={<CustomTooltip />} cursor={false} />

            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
              {sortedFeatures.map((feature, index) => (
                <Cell
                  key={index}
                  fill={
                    feature.direction === 'up'
                      ? 'rgba(239, 68, 68, 0.7)'
                      : 'rgba(34, 197, 94, 0.7)'
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-risk-red/70" />
          Increases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-risk-green/70" />
          Decreases risk
        </span>
      </div>
    </div>
  );
}
