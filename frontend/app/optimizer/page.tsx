"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { OptimizerResults } from "@/components/OptimizerResults";
import { Skeleton } from "@/components/ui/skeleton";
import { Play, AlertCircle } from "lucide-react";
import { api } from "@/lib/api";
import type { OptimizerParams, OptimizerResults as OptimizerResultsType } from "@/lib/types";

export default function OptimizerPage() {
  const [params, setParams] = useState<OptimizerParams>({
    target_horizon: "3m",
    risk_tolerance: 0.15,
    max_positions: 10,
  });
  const [results, setResults] = useState<OptimizerResultsType | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runOptimizer = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.optimizer.run(params);
      setResults(data);
    } catch (err: any) {
      setError(err.message || "Failed to run optimization");
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold mb-2">Portfolio Optimizer</h1>
          <p className="text-white/60">
            Run portfolio optimization to find efficient allocations based on risk-return trade-offs
          </p>
        </div>

        {/* Parameters Card */}
        <Card className="border-white/10 bg-white/5">
          <CardHeader>
            <CardTitle className="text-lg">Optimization Parameters</CardTitle>
            <CardDescription className="text-white/60">
              Configure constraints and preferences for portfolio optimization
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Target Horizon */}
              <div className="space-y-2">
                <Label htmlFor="horizon">Target Horizon</Label>
                <Select
                  value={params.target_horizon}
                  onValueChange={(value: any) =>
                    setParams({ ...params, target_horizon: value })
                  }
                >
                  <SelectTrigger id="horizon" className="bg-slate-900 border-white/10">
                    <SelectValue placeholder="Select horizon" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-900 border-white/10">
                    <SelectItem value="1w">1 Week</SelectItem>
                    <SelectItem value="1m">1 Month</SelectItem>
                    <SelectItem value="3m">3 Months</SelectItem>
                    <SelectItem value="6m">6 Months</SelectItem>
                    <SelectItem value="1y">1 Year</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Risk Tolerance */}
              <div className="space-y-2">
                <Label htmlFor="risk">
                  Risk Tolerance: {(params.risk_tolerance * 100).toFixed(0)}%
                </Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="risk"
                    type="range"
                    min="0.01"
                    max="0.50"
                    step="0.01"
                    value={params.risk_tolerance}
                    onChange={(e) =>
                      setParams({ ...params, risk_tolerance: parseFloat(e.target.value) })
                    }
                    className="flex-1"
                  />
                </div>
                <p className="text-xs text-white/40">
                  {params.risk_tolerance < 0.1
                    ? "Conservative"
                    : params.risk_tolerance < 0.25
                    ? "Moderate"
                    : "Aggressive"}
                </p>
              </div>

              {/* Max Positions */}
              <div className="space-y-2">
                <Label htmlFor="positions">Max Positions</Label>
                <Input
                  id="positions"
                  type="number"
                  min="1"
                  max="50"
                  value={params.max_positions}
                  onChange={(e) =>
                    setParams({ ...params, max_positions: parseInt(e.target.value) })
                  }
                  className="bg-slate-900 border-white/10"
                />
              </div>
            </div>

            {/* Run Button */}
            <div className="flex items-center gap-4">
              <Button
                onClick={runOptimizer}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-500"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white mr-2" />
                    Running Optimization...
                  </>
                ) : (
                  <>
                    <Play size={16} className="mr-2" />
                    Run Optimization
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Loading State */}
        {loading && (
          <div className="space-y-6">
            <Skeleton className="h-[300px] w-full" />
            <Skeleton className="h-[400px] w-full" />
          </div>
        )}

        {/* Error State */}
        {error && (
          <Card className="border-red-500/40 bg-red-500/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-red-200">
                <AlertCircle size={20} />
                <div>
                  <h3 className="font-semibold mb-1">Optimization Failed</h3>
                  <p className="text-sm">{error}</p>
                  <p className="text-xs mt-2 text-red-300/80">
                    Make sure the backend optimizer endpoint is available at{" "}
                    <code className="bg-red-500/20 px-1 py-0.5 rounded">/api/optimizer/run</code>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {results && !loading && <OptimizerResults results={results} />}

        {/* Empty State */}
        {!results && !loading && !error && (
          <Card className="border-white/10 bg-white/5">
            <CardContent className="py-12">
              <div className="text-center text-white/60">
                <Play size={48} className="mx-auto mb-4 opacity-40" />
                <p className="text-lg font-medium mb-2">Ready to Optimize</p>
                <p className="text-sm">
                  Configure the parameters above and click "Run Optimization" to find the optimal
                  portfolio allocation
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}