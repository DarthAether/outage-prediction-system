'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import type { PredictionResult, RiskLevel } from '@/lib/types';
import { getPredictions } from '@/lib/api';

interface UsePredictionsOptions {
  minRisk?: number;
  riskLevel?: RiskLevel;
  limit?: number;
}

interface UsePredictionsResult {
  predictions: PredictionResult[];
  isLoading: boolean;
  error: Error | null;
  lastUpdated: Date | null;
  refetch: () => Promise<void>;
}

const DEFAULT_REFRESH_INTERVAL = 30000;

export function usePredictions(
  region: string,
  refreshInterval: number = DEFAULT_REFRESH_INTERVAL,
  options?: UsePredictionsOptions
): UsePredictionsResult {
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchPredictions = useCallback(async () => {
    try {
      setError(null);
      const data = await getPredictions(region, {
        min_risk: options?.minRisk,
        risk_level: options?.riskLevel,
        limit: options?.limit,
      });
      setPredictions(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(
        err instanceof Error ? err : new Error('Failed to fetch predictions')
      );
    } finally {
      setIsLoading(false);
    }
  }, [region, options?.minRisk, options?.riskLevel, options?.limit]);

  useEffect(() => {
    setIsLoading(true);
    fetchPredictions();

    if (refreshInterval > 0) {
      intervalRef.current = setInterval(fetchPredictions, refreshInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchPredictions, refreshInterval]);

  return {
    predictions,
    isLoading,
    error,
    lastUpdated,
    refetch: fetchPredictions,
  };
}
