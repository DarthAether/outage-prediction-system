'use client';

import { useEffect, useRef, useState } from 'react';
import { Filter } from 'lucide-react';
import type { Alert, RiskLevel } from '@/lib/types';
import AlertCard from './AlertCard';

interface AlertFeedProps {
  alerts: Alert[];
  onAcknowledge?: (alertId: string) => void;
}

const severityOrder: Record<RiskLevel, number> = {
  RED: 0,
  ORANGE: 1,
  YELLOW: 2,
  GREEN: 3,
};

export default function AlertFeed({ alerts, onAcknowledge }: AlertFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(alerts.length);
  const [severityFilter, setSeverityFilter] = useState<RiskLevel | 'ALL'>(
    'ALL'
  );
  const [filterOpen, setFilterOpen] = useState(false);

  useEffect(() => {
    if (alerts.length > prevCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
    prevCountRef.current = alerts.length;
  }, [alerts.length]);

  const filteredAlerts = alerts
    .filter((a) => severityFilter === 'ALL' || a.severity === severityFilter)
    .sort((a, b) => {
      const sevDiff = severityOrder[a.severity] - severityOrder[b.severity];
      if (sevDiff !== 0) return sevDiff;
      return (
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
    });

  const severityOptions: (RiskLevel | 'ALL')[] = [
    'ALL',
    'RED',
    'ORANGE',
    'YELLOW',
    'GREEN',
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-1 pb-3">
        <h3 className="text-sm font-semibold text-gray-300">
          Active Alerts
          <span className="ml-2 rounded-full bg-surface-overlay px-2 py-0.5 text-xs font-normal text-gray-500">
            {filteredAlerts.length}
          </span>
        </h3>

        <div className="relative">
          <button
            onClick={() => setFilterOpen(!filterOpen)}
            className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors ${
              severityFilter !== 'ALL'
                ? 'bg-accent/10 text-accent-hover'
                : 'text-gray-500 hover:bg-surface-overlay hover:text-gray-300'
            }`}
          >
            <Filter className="h-3 w-3" />
            {severityFilter === 'ALL' ? 'Filter' : severityFilter}
          </button>

          {filterOpen && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setFilterOpen(false)}
              />
              <div className="absolute right-0 top-full z-20 mt-1 w-32 rounded-md border border-surface-border bg-surface-overlay py-1 shadow-xl">
                {severityOptions.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => {
                      setSeverityFilter(opt);
                      setFilterOpen(false);
                    }}
                    className={`flex w-full items-center px-3 py-1.5 text-xs transition-colors ${
                      opt === severityFilter
                        ? 'bg-accent/10 text-accent-hover'
                        : 'text-gray-400 hover:bg-surface-border/50 hover:text-white'
                    }`}
                  >
                    {opt === 'ALL' ? 'All Severities' : opt}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 space-y-2 overflow-y-auto pr-1 gradient-mask"
      >
        {filteredAlerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="mb-2 rounded-full bg-surface-overlay p-3">
              <Filter className="h-5 w-5 text-gray-600" />
            </div>
            <p className="text-sm text-gray-500">No alerts found</p>
            <p className="text-xs text-gray-600">
              {severityFilter !== 'ALL'
                ? 'Try changing the severity filter'
                : 'System is operating normally'}
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <AlertCard
              key={alert.id}
              alert={alert}
              onAcknowledge={onAcknowledge}
            />
          ))
        )}
      </div>
    </div>
  );
}
