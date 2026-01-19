"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { X, Download, TrendingUp, TrendingDown } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ReportDetails {
  id: string;
  type: "backtest" | "model" | "performance" | "analysis";
  name: string;
  description?: string;
  created_at: string;
  metrics: Record<string, any>;
  summary: {
    total_trades?: number;
    win_rate?: number;
    profit_factor?: number;
    max_drawdown?: number;
    sharpe_ratio?: number;
    return_pct?: number;
  };
  charts: Array<{
    id: string;
    title: string;
    type: string;
    data: any;
  }>;
}

interface ReportViewerProps {
  report: ReportDetails;
  onClose: () => void;
  onDownload: (reportId: string, format: "pdf" | "csv" | "json") => void;
  className?: string;
}

export function ReportViewer({ report, onClose, onDownload, className }: ReportViewerProps) {
  const [activeTab, setActiveTab] = useState("summary");
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateString;
    }
  };

  const formatPercent = (value?: number) => {
    if (value === undefined || value === null) return "N/A";
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatNumber = (value?: number, decimals = 2) => {
    if (value === undefined || value === null) return "N/A";
    return value.toFixed(decimals);
  };

  return (
    <div className={`fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50 ${className || ""}`}>
      <Card className="border-white/10 bg-slate-900 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <CardHeader className="border-b border-white/10">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl">{report.name}</CardTitle>
              {report.description && (
                <p className="text-sm text-white/60 mt-1">{report.description}</p>
              )}
              <p className="text-xs text-white/40 mt-1">Created: {formatDate(report.created_at)}</p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => onDownload(report.id, "pdf")}
                className="border-white/10"
              >
                <Download size={14} className="mr-1" />
                PDF
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="border-white/10"
              >
                <X size={16} />
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto p-6">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-white/5">
              <TabsTrigger value="summary">Summary</TabsTrigger>
              <TabsTrigger value="metrics">Metrics</TabsTrigger>
              <TabsTrigger value="charts">Charts</TabsTrigger>
            </TabsList>

            <TabsContent value="summary" className="space-y-4 mt-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {report.summary.total_trades !== undefined && (
                  <MetricBox
                    label="Total Trades"
                    value={report.summary.total_trades.toString()}
                  />
                )}
                {report.summary.win_rate !== undefined && (
                  <MetricBox
                    label="Win Rate"
                    value={formatPercent(report.summary.win_rate)}
                    isPositive={report.summary.win_rate > 0.5}
                  />
                )}
                {report.summary.return_pct !== undefined && (
                  <MetricBox
                    label="Return"
                    value={formatPercent(report.summary.return_pct)}
                    isPositive={report.summary.return_pct > 0}
                  />
                )}
                {report.summary.sharpe_ratio !== undefined && (
                  <MetricBox
                    label="Sharpe Ratio"
                    value={formatNumber(report.summary.sharpe_ratio)}
                    isPositive={report.summary.sharpe_ratio > 1}
                  />
                )}
                {report.summary.profit_factor !== undefined && (
                  <MetricBox
                    label="Profit Factor"
                    value={formatNumber(report.summary.profit_factor)}
                    isPositive={report.summary.profit_factor > 1}
                  />
                )}
                {report.summary.max_drawdown !== undefined && (
                  <MetricBox
                    label="Max Drawdown"
                    value={formatPercent(report.summary.max_drawdown)}
                    isPositive={report.summary.max_drawdown > -0.1}
                  />
                )}
              </div>
            </TabsContent>

            <TabsContent value="metrics" className="mt-4">
              <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                <pre className="text-sm text-white/80 overflow-x-auto">
                  {JSON.stringify(report.metrics, null, 2)}
                </pre>
              </div>
            </TabsContent>

            <TabsContent value="charts" className="mt-4">
              {report.charts && report.charts.length > 0 ? (
                <div className="space-y-4">
                  {report.charts.map((chart) => (
                    <div key={chart.id} className="rounded-lg border border-white/10 bg-white/5 p-4">
                      <h3 className="text-lg font-semibold mb-2">{chart.title}</h3>
                      <p className="text-sm text-white/60 mb-2">Type: {chart.type}</p>
                      <div className="text-xs text-white/40">
                        Chart data: {JSON.stringify(chart.data).substring(0, 100)}...
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-white/60">No charts available</div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricBox({
  label,
  value,
  isPositive,
}: {
  label: string;
  value: string;
  isPositive?: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-3">
      <div className="text-xs text-white/60 mb-1">{label}</div>
      <div className={`text-lg font-semibold flex items-center gap-1 ${
        isPositive === undefined ? "text-white" :
        isPositive ? "text-green-400" : "text-red-400"
      }`}>
        {isPositive !== undefined && (
          isPositive ? <TrendingUp size={16} /> : <TrendingDown size={16} />
        )}
        {value}
      </div>
    </div>
  );
}
