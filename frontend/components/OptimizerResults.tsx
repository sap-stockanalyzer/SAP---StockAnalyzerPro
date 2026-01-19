"use client";

import { EfficientFrontierChart } from "./EfficientFrontierChart";
import { AllocationTable } from "./AllocationTable";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TrendingUp, Target, AlertTriangle } from "lucide-react";

interface OptimizerResultsProps {
  results: {
    efficient_frontier: Array<{ risk: number; return: number }>;
    recommended_portfolio: Array<{
      symbol: string;
      allocation_pct: number;
      shares?: number;
      dollar_amount?: number;
    }>;
    expected_return: number;
    expected_risk: number;
    sharpe_ratio: number;
    status: "success" | "partial" | "failed";
    warnings?: string[];
    computation_time_ms: number;
  };
  className?: string;
}

export function OptimizerResults({ results, className }: OptimizerResultsProps) {
  return (
    <div className={`space-y-6 ${className || ""}`}>
      {/* Status and Warnings */}
      {results.status !== "success" && (
        <div
          className={`rounded-xl border p-4 ${
            results.status === "failed"
              ? "border-red-500/40 bg-red-500/10"
              : "border-yellow-500/40 bg-yellow-500/10"
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle
              className={results.status === "failed" ? "text-red-400" : "text-yellow-400"}
              size={20}
            />
            <h3 className="font-semibold">
              {results.status === "failed" ? "Optimization Failed" : "Partial Results"}
            </h3>
          </div>
          {results.warnings && results.warnings.length > 0 && (
            <ul className="text-sm space-y-1 text-white/80">
              {results.warnings.map((warning, i) => (
                <li key={i}>• {warning}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-white/10 bg-white/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp size={16} className="text-blue-400" />
              Expected Return
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {(results.expected_return * 100).toFixed(2)}%
            </div>
            <p className="text-xs text-white/60 mt-1">Annualized</p>
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-white/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Target size={16} className="text-orange-400" />
              Expected Risk
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {(results.expected_risk * 100).toFixed(2)}%
            </div>
            <p className="text-xs text-white/60 mt-1">Volatility</p>
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-white/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp size={16} className="text-green-400" />
              Sharpe Ratio
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {results.sharpe_ratio.toFixed(3)}
            </div>
            <p className="text-xs text-white/60 mt-1">Risk-adjusted return</p>
          </CardContent>
        </Card>
      </div>

      {/* Efficient Frontier Chart */}
      <EfficientFrontierChart
        data={results.efficient_frontier}
        currentPortfolio={{ risk: results.expected_risk, return: results.expected_return }}
      />

      {/* Allocation Table */}
      <AllocationTable allocations={results.recommended_portfolio} />

      {/* Computation Time */}
      <div className="text-xs text-white/40 text-right">
        Computed in {results.computation_time_ms.toFixed(0)}ms
      </div>
    </div>
  );
}
