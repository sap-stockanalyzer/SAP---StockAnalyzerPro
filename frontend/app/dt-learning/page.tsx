"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface LearningMetrics {
  performance_7d: {
    win_rate: number;
    accuracy: number;
    total_trades: number;
    profit_factor: number;
    sharpe_ratio: number;
    avg_win: number;
    avg_loss: number;
    consecutive_wins: number;
    consecutive_losses: number;
  };
  performance_30d: {
    win_rate: number;
    accuracy: number;
    total_trades: number;
  };
  baseline: {
    win_rate: number;
    accuracy: number;
    profit_factor: number;
  };
  model_health: {
    days_since_retrain: number;
    confidence_calibration: number;
    next_retrain_estimate: string;
  };
  missed_opportunities: {
    total_evaluated: number;
    profitable_missed_pct: number;
    missed_pnl_usd: number;
    suggestions: string[];
  };
  dt_brain_knobs: Record<string, {
    current: number;
    default: number;
    range: [number, number];
    diff: number;
    diff_pct: number;
  }>;
  trade_quality: {
    total_trades_7d: number;
    win_rate_7d: number;
    profit_factor_7d: number;
    sharpe_ratio_7d: number;
  };
}

function MetricCard({ 
  label, 
  value, 
  format = "number",
  change,
  changeLabel,
}: { 
  label: string; 
  value: number | string | undefined;
  format?: "number" | "percent" | "currency" | "decimal";
  change?: number;
  changeLabel?: string;
}) {
  const formatValue = (val: number | string | undefined) => {
    if (val === undefined || val === null) return "N/A";
    if (typeof val === "string") return val;
    
    switch (format) {
      case "percent":
        return `${(val * 100).toFixed(1)}%`;
      case "currency":
        return `$${val.toFixed(2)}`;
      case "decimal":
        return val.toFixed(2);
      default:
        return val.toString();
    }
  };

  const getChangeColor = (change: number | undefined) => {
    if (!change) return "";
    return change > 0 ? "text-green-600" : "text-red-600";
  };

  return (
    <div className="space-y-1">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-2xl font-bold">{formatValue(value)}</div>
      {change !== undefined && (
        <div className={`text-xs ${getChangeColor(change)}`}>
          {change > 0 ? "+" : ""}{formatValue(change)} {changeLabel}
        </div>
      )}
    </div>
  );
}

function KnobTable({ knobs }: { knobs: LearningMetrics["dt_brain_knobs"] | undefined }) {
  if (!knobs) return <div>No knob data available</div>;

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b">
            <th className="text-left p-2">Knob</th>
            <th className="text-right p-2">Current</th>
            <th className="text-right p-2">Default</th>
            <th className="text-right p-2">Change</th>
            <th className="text-right p-2">Range</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(knobs).map(([name, data]) => (
            <tr key={name} className="border-b hover:bg-gray-50">
              <td className="p-2 font-mono text-xs">{name}</td>
              <td className="text-right p-2 font-semibold">{data.current.toFixed(3)}</td>
              <td className="text-right p-2 text-gray-500">{data.default.toFixed(3)}</td>
              <td className={`text-right p-2 ${data.diff > 0 ? 'text-green-600' : data.diff < 0 ? 'text-red-600' : ''}`}>
                {data.diff > 0 ? '+' : ''}{data.diff.toFixed(3)} ({data.diff_pct.toFixed(1)}%)
              </td>
              <td className="text-right p-2 text-xs text-gray-500">
                [{data.range[0]} - {data.range[1]}]
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DTLearningDashboard() {
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/backend/api/dt/learning/metrics");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setMetrics(data);
        setError(null);
      } catch (e) {
        console.error("Error fetching learning metrics:", e);
        setError(e instanceof Error ? e.message : "Failed to fetch metrics");
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    
    // Refresh every 60 seconds
    const interval = setInterval(fetchMetrics, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !metrics) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold mb-6">Day Trading Learning</h1>
        <div className="text-center py-12">Loading learning metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold mb-6">Day Trading Learning</h1>
        <Card>
          <CardContent className="p-6">
            <div className="text-red-600">Error: {error}</div>
            <div className="mt-4 text-sm text-gray-500">
              Make sure the dt_backend API is running and accessible.
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Day Trading Learning</h1>
        <div className="text-sm text-gray-500">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* Performance Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Trends (7 Days)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <MetricCard 
              label="Win Rate" 
              value={metrics?.performance_7d?.win_rate} 
              format="percent"
              change={metrics?.performance_7d?.win_rate && metrics?.baseline?.win_rate ? 
                metrics.performance_7d.win_rate - metrics.baseline.win_rate : undefined}
              changeLabel="vs baseline"
            />
            <MetricCard 
              label="Accuracy" 
              value={metrics?.performance_7d?.accuracy} 
              format="percent"
            />
            <MetricCard 
              label="Profit Factor" 
              value={metrics?.performance_7d?.profit_factor} 
              format="decimal"
            />
            <MetricCard 
              label="Sharpe Ratio" 
              value={metrics?.performance_7d?.sharpe_ratio} 
              format="decimal"
            />
          </div>
          
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-6 pt-6 border-t">
            <MetricCard 
              label="Total Trades" 
              value={metrics?.performance_7d?.total_trades} 
            />
            <MetricCard 
              label="Avg Win" 
              value={metrics?.performance_7d?.avg_win} 
              format="percent"
            />
            <MetricCard 
              label="Avg Loss" 
              value={metrics?.performance_7d?.avg_loss} 
              format="percent"
            />
            <MetricCard 
              label="Streak" 
              value={
                (metrics?.performance_7d?.consecutive_wins || 0) > 0 
                  ? `${metrics?.performance_7d?.consecutive_wins}W`
                  : `${metrics?.performance_7d?.consecutive_losses}L`
              } 
            />
          </div>
        </CardContent>
      </Card>

      {/* Model Health */}
      <Card>
        <CardHeader>
          <CardTitle>Model Health</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Days since retrain:</span>
              <span className="font-semibold text-lg">
                {metrics?.model_health?.days_since_retrain || 0}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Confidence calibration:</span>
              <span className="font-semibold text-lg">
                {(metrics?.model_health?.confidence_calibration || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Next retrain trigger:</span>
              <span className="font-semibold">
                {metrics?.model_health?.next_retrain_estimate || "Unknown"}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Missed Opportunities */}
      <Card>
        <CardHeader>
          <CardTitle>Missed Opportunities</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-6 mb-4">
            <MetricCard 
              label="Total Evaluated" 
              value={metrics?.missed_opportunities?.total_evaluated || 0} 
            />
            <MetricCard 
              label="Would-Have-Won Rate" 
              value={metrics?.missed_opportunities?.profitable_missed_pct} 
              format="percent"
            />
            <MetricCard 
              label="Missed PnL" 
              value={metrics?.missed_opportunities?.missed_pnl_usd} 
              format="currency"
            />
          </div>
          
          {metrics?.missed_opportunities?.suggestions && 
           metrics.missed_opportunities.suggestions.length > 0 && (
            <div className="mt-6 pt-6 border-t">
              <div className="text-sm font-semibold mb-2">Suggestions:</div>
              <ul className="space-y-2">
                {metrics.missed_opportunities.suggestions.map((suggestion, i) => (
                  <li key={i} className="text-sm text-gray-700 flex items-start">
                    <span className="mr-2">•</span>
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {/* DT Brain Status */}
      <Card>
        <CardHeader>
          <CardTitle>DT Brain - Auto Knobs</CardTitle>
        </CardHeader>
        <CardContent>
          <KnobTable knobs={metrics?.dt_brain_knobs} />
          <div className="mt-4 text-xs text-gray-500">
            Knobs auto-adjust based on performance using EMA smoothing (20% per update)
          </div>
        </CardContent>
      </Card>

      {/* Baseline Comparison */}
      {metrics?.baseline && Object.keys(metrics.baseline).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Baseline Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-6">
              <div>
                <div className="text-sm text-gray-500">Baseline Win Rate</div>
                <div className="text-xl font-bold">
                  {((metrics.baseline.win_rate || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Baseline Accuracy</div>
                <div className="text-xl font-bold">
                  {((metrics.baseline.accuracy || 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Baseline Profit Factor</div>
                <div className="text-xl font-bold">
                  {(metrics.baseline.profit_factor || 0).toFixed(2)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
