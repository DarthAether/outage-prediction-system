'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import type { Alert } from '@/lib/types';

type ConnectionStatus = 'connected' | 'disconnected' | 'reconnecting';

const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/v1';
const MAX_ALERTS = 100;
const MAX_RECONNECT_DELAY = 30000;
const BASE_RECONNECT_DELAY = 1000;

export function useAlertStream(region: string): {
  alerts: Alert[];
  connectionStatus: ConnectionStatus;
} {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const regionRef = useRef(region);

  regionRef.current = region;

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    const url = `${WS_BASE}/alerts/stream?region=${regionRef.current}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      reconnectAttemptRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const alert: Alert = JSON.parse(event.data);
        setAlerts((prev) => {
          const updated = [alert, ...prev];
          return updated.slice(0, MAX_ALERTS);
        });
      } catch {
        console.error('Failed to parse alert message');
      }
    };

    ws.onclose = (event) => {
      if (event.code === 1000) {
        setConnectionStatus('disconnected');
        return;
      }

      setConnectionStatus('reconnecting');
      const delay = Math.min(
        BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttemptRef.current),
        MAX_RECONNECT_DELAY
      );
      reconnectAttemptRef.current += 1;

      clearReconnectTimeout();
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, delay);
    };

    ws.onerror = () => {
      setConnectionStatus('reconnecting');
    };
  }, [clearReconnectTimeout]);

  useEffect(() => {
    connect();

    return () => {
      clearReconnectTimeout();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [region, connect, clearReconnectTimeout]);

  return { alerts, connectionStatus };
}
