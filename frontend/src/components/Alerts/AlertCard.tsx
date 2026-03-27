'use client';

import { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Clock,
  MapPin,
} from 'lucide-react';
import type { Alert, RiskLevel } from '@/lib/types';

interface AlertCardProps {
  alert: Alert;
  onAcknowledge?: (alertId: string) => void;
}

const borderClass: Record<RiskLevel, string> = {
  GREEN: 'risk-border-green',
  YELLOW: 'risk-border-yellow',
  ORANGE: 'risk-border-orange',
  RED: 'risk-border-red',
};

const badgeClass: Record<RiskLevel, string> = {
  GREEN: 'badge-green',
  YELLOW: 'badge-yellow',
  ORANGE: 'badge-orange',
  RED: 'badge-red',
};

function timeAgo(dateString: string): string {
  const now = new Date();
  const date = new Date(dateString);
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function AlertCard({ alert, onAcknowledge }: AlertCardProps) {
  const [expanded, setExpanded] = useState(false);

  const riskPercent = (alert.risk_probability * 100).toFixed(1);
  const uncertaintyLow = (alert.uncertainty_range[0] * 100).toFixed(1);
  const uncertaintyHigh = (alert.uncertainty_range[1] * 100).toFixed(1);

  return (
    <div
      className={`${borderClass[alert.severity]} card animate-fade-in space-y-2`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={badgeClass[alert.severity]}>
              {alert.severity}
            </span>
            <span className="text-lg font-bold text-white tabular-nums">
              {riskPercent}%
            </span>
          </div>

          <p className="text-sm text-gray-300 line-clamp-2">
            {alert.description}
          </p>
        </div>

        {!alert.acknowledged && onAcknowledge && (
          <button
            onClick={() => onAcknowledge(alert.id)}
            className="shrink-0 rounded-md p-1.5 text-gray-500 transition-colors hover:bg-surface-overlay hover:text-risk-green"
            title="Acknowledge alert"
          >
            <CheckCircle className="h-4 w-4" />
          </button>
        )}
        {alert.acknowledged && (
          <CheckCircle className="h-4 w-4 shrink-0 text-risk-green/50" />
        )}
      </div>

      <div className="flex items-center gap-3 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <MapPin className="h-3 w-3" />
          {alert.h3_cell.slice(0, 10)}...
        </span>
        <span className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {timeAgo(alert.created_at)}
        </span>
        <span className="tabular-nums">
          CI: {uncertaintyLow}%-{uncertaintyHigh}%
        </span>
      </div>

      {alert.recommended_actions.length > 0 && (
        <div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-xs font-medium text-accent transition-colors hover:text-accent-hover"
          >
            {expanded ? (
              <>
                <ChevronUp className="h-3 w-3" />
                Hide actions
              </>
            ) : (
              <>
                <ChevronDown className="h-3 w-3" />
                {alert.recommended_actions.length} recommended actions
              </>
            )}
          </button>

          {expanded && (
            <ul className="mt-2 space-y-1.5 animate-slide-up">
              {alert.recommended_actions.map((action, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 rounded-md bg-surface/50 px-2.5 py-1.5 text-xs text-gray-400"
                >
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/60" />
                  {action}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
