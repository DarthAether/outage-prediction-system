'use client';

import { useState } from 'react';

interface ModelInfo {
  name: string;
  version: string;
  region: string;
  auc_roc: number;
  auc_pr: number;
  f1: number;
  ece: number;
  is_active: boolean;
}

const MOCK_MODELS: ModelInfo[] = [
  { name: 'XGBoost', version: 'v1.0.0', region: 'TX', auc_roc: 0.847, auc_pr: 0.812, f1: 0.783, ece: 0.089, is_active: true },
  { name: 'LightGBM', version: 'v1.0.0', region: 'TX', auc_roc: 0.852, auc_pr: 0.819, f1: 0.791, ece: 0.092, is_active: true },
  { name: 'LSTM+Attention', version: 'v1.0.0', region: 'TX', auc_roc: 0.838, auc_pr: 0.798, f1: 0.769, ece: 0.103, is_active: true },
  { name: 'Ensemble (Stacking)', version: 'v1.0.0', region: 'TX', auc_roc: 0.893, auc_pr: 0.861, f1: 0.824, ece: 0.067, is_active: true },
];

export default function AdminPage() {
  const [thresholds, setThresholds] = useState({
    yellow: 0.25,
    orange: 0.55,
    red: 0.80,
  });
  const [selectedRegion, setSelectedRegion] = useState('TX');

  const handleThresholdChange = (level: string, value: number) => {
    setThresholds(prev => ({ ...prev, [level]: value }));
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl font-bold">System Administration</h1>
          <p className="text-gray-400 mt-1">Model management, thresholds, and system health</p>
        </div>

        {/* Region Selector */}
        <div className="flex items-center gap-4">
          <label className="text-sm text-gray-400">Region:</label>
          <select
            value={selectedRegion}
            onChange={e => setSelectedRegion(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          >
            <option value="TX">Texas</option>
            <option value="CA">California</option>
            <option value="FL">Florida</option>
          </select>
        </div>

        {/* Risk Thresholds */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Risk Thresholds</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(thresholds).map(([level, value]) => {
              const colors: Record<string, string> = {
                yellow: 'text-yellow-400',
                orange: 'text-orange-400',
                red: 'text-red-400',
              };
              return (
                <div key={level} className="space-y-2">
                  <div className="flex justify-between">
                    <label className={`text-sm font-medium capitalize ${colors[level]}`}>
                      {level} Threshold
                    </label>
                    <span className="text-sm text-gray-400">{(value * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={value * 100}
                    onChange={e => handleThresholdChange(level, parseInt(e.target.value) / 100)}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              );
            })}
          </div>
          <button className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors">
            Save Thresholds
          </button>
        </div>

        {/* Model Comparison */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Model Comparison — {selectedRegion}</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-800">
                  <th className="text-left py-3 px-4">Model</th>
                  <th className="text-left py-3 px-4">Version</th>
                  <th className="text-right py-3 px-4">AUC-ROC</th>
                  <th className="text-right py-3 px-4">AUC-PR</th>
                  <th className="text-right py-3 px-4">F1</th>
                  <th className="text-right py-3 px-4">ECE</th>
                  <th className="text-center py-3 px-4">Status</th>
                  <th className="text-center py-3 px-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {MOCK_MODELS.map((model, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-3 px-4 font-medium">{model.name}</td>
                    <td className="py-3 px-4 text-gray-400">{model.version}</td>
                    <td className="py-3 px-4 text-right font-mono">{model.auc_roc.toFixed(3)}</td>
                    <td className="py-3 px-4 text-right font-mono">{model.auc_pr.toFixed(3)}</td>
                    <td className="py-3 px-4 text-right font-mono">{model.f1.toFixed(3)}</td>
                    <td className="py-3 px-4 text-right font-mono">{model.ece.toFixed(3)}</td>
                    <td className="py-3 px-4 text-center">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                        model.is_active ? 'bg-green-900/50 text-green-400' : 'bg-gray-800 text-gray-500'
                      }`}>
                        {model.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <button className="text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded transition-colors">
                        Promote
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* System Health */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
          <h2 className="text-lg font-semibold mb-4">System Health</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { name: 'PostgreSQL + TimescaleDB', status: 'healthy' },
              { name: 'Redis Streams', status: 'healthy' },
              { name: 'MLflow Tracking', status: 'healthy' },
              { name: 'Prediction Service', status: 'healthy' },
            ].map(service => (
              <div key={service.name} className="flex items-center gap-3 p-3 bg-gray-800/50 rounded">
                <div className={`w-2.5 h-2.5 rounded-full ${
                  service.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <div>
                  <p className="text-sm font-medium">{service.name}</p>
                  <p className="text-xs text-gray-500 capitalize">{service.status}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
