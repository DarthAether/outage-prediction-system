'use client';

import { useEffect, useState, useCallback } from 'react';
import { Activity, Clock, ChevronDown } from 'lucide-react';
import { REGIONS, type RegionOption } from '@/lib/types';
import { getHealth } from '@/lib/api';

interface HeaderProps {
  title: string;
  selectedRegion: string;
  onRegionChange: (region: string) => void;
}

export default function Header({
  title,
  selectedRegion,
  onRegionChange,
}: HeaderProps) {
  const [healthStatus, setHealthStatus] = useState<
    'healthy' | 'degraded' | 'unhealthy' | 'unknown'
  >('unknown');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const health = await getHealth();
      setHealthStatus(health.status);
      setLastUpdated(new Date());
    } catch {
      setHealthStatus('unknown');
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 60000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const statusColor: Record<string, string> = {
    healthy: 'bg-risk-green',
    degraded: 'bg-risk-yellow',
    unhealthy: 'bg-risk-red',
    unknown: 'bg-gray-500',
  };

  const selectedRegionData = REGIONS.find(
    (r: RegionOption) => r.code === selectedRegion
  );

  const formatTime = (date: Date | null) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <header className="flex h-14 items-center justify-between border-b border-surface-border bg-surface-raised px-6">
      <h1 className="text-lg font-semibold text-white">{title}</h1>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Clock className="h-3.5 w-3.5" />
          <span>Updated {formatTime(lastUpdated)}</span>
        </div>

        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 rounded-md border border-surface-border bg-surface px-3 py-1.5 text-sm text-gray-300 transition-colors hover:border-gray-600 hover:text-white"
          >
            <span className="font-medium">{selectedRegionData?.code}</span>
            <span className="text-gray-500">
              {selectedRegionData?.name}
            </span>
            <ChevronDown className="h-3.5 w-3.5 text-gray-500" />
          </button>

          {dropdownOpen && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setDropdownOpen(false)}
              />
              <div className="absolute right-0 top-full z-20 mt-1 w-48 rounded-md border border-surface-border bg-surface-overlay py-1 shadow-xl">
                {REGIONS.map((region: RegionOption) => (
                  <button
                    key={region.code}
                    onClick={() => {
                      onRegionChange(region.code);
                      setDropdownOpen(false);
                    }}
                    className={`flex w-full items-center gap-2 px-3 py-2 text-sm transition-colors ${
                      region.code === selectedRegion
                        ? 'bg-accent/10 text-accent-hover'
                        : 'text-gray-300 hover:bg-surface-border/50 hover:text-white'
                    }`}
                  >
                    <span className="font-medium">{region.code}</span>
                    <span className="text-gray-500">{region.name}</span>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${statusColor[healthStatus]}`}
            title={`System: ${healthStatus}`}
          />
          <Activity className="h-3.5 w-3.5 text-gray-500" />
        </div>
      </div>
    </header>
  );
}
