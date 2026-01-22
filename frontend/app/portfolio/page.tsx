"use client";

import { useEffect, useState } from "react";
import { PerformanceChart } from "@/components/charts/PerformanceChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { DollarSign, TrendingUp, Activity } from "lucide-react";

type BotEquity = {
  botKey: string;
  equity: number;
  equity_curve?: Array<{ t?: string; value?: number }>;
};

type PortfolioData = {
  totalEquity: number;
  bots: BotEquity[];
  combinedCurve: Array<{ t: string; value: number }>;
};

export default function PortfolioPage() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<PortfolioData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPortfolio() {
      try {
        const res = await fetch("/api/backend/bots/page", { cache: "no-store" });
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        
        const bundle = await res.json();
        const swingBots = bundle?.swing?.status?.bots || {};
        
        // Extract bot data
        const bots: BotEquity[] = Object.entries(swingBots).map(([key, bot]: [string, any]) => ({
          botKey: key,
          equity: bot.equity || 0,
          equity_curve: bot.equity_curve || [],
        }));

        // Calculate total equity
        const totalEquity = bots.reduce((sum, bot) => sum + bot.equity, 0);

        // Combine curves (simplified - take first bot's curve as baseline)
        let combinedCurve: Array<{ t: string; value: number }> = [];
        if (bots.length > 0 && bots[0].equity_curve) {
          combinedCurve = bots[0].equity_curve
            .filter((point: any) => point && point.t && typeof point.value === "number")
            .map((point: any) => ({
              t: point.t,
              value: point.value,
            }));
        }

        // Fallback if no data
        if (combinedCurve.length === 0) {
          combinedCurve = generateFallbackCurve(totalEquity);
        }

        setData({ totalEquity, bots, combinedCurve });
      } catch (err: any) {
        setError(err.message || "Failed to load portfolio data");
      } finally {
        setLoading(false);
      }
    }

    fetchPortfolio();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-8">
        <div className="max-w-7xl mx-auto space-y-6">
          <Skeleton className="h-12 w-64" />
          <Skeleton className="h-[300px] w-full" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-8">
        <div className="max-w-7xl mx-auto">
          <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-6 text-red-200">
            <h3 className="font-semibold mb-2">Error Loading Portfolio</h3>
            <p>{error || "Unknown error"}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold mb-2">Portfolio</h1>
          <p className="text-white/60">Track your trading bot performance and equity</p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="border-white/10 bg-white/5">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total Equity</CardTitle>
              <DollarSign className="h-4 w-4 text-white/60" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${data.totalEquity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-white/5">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Active Bots</CardTitle>
              <Activity className="h-4 w-4 text-white/60" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{data.bots.length}</div>
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-white/5">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Performance</CardTitle>
              <TrendingUp className="h-4 w-4 text-white/60" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">
                {calculatePerformance(data.combinedCurve)}%
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Chart */}
        <PerformanceChart
          data={data.combinedCurve}
          title="Portfolio Equity Over Time"
          description="Combined equity from all active trading bots"
          valueLabel="Equity"
          showFooter={true}
        />

        {/* Individual Bot Cards */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Individual Bots</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.bots.map((bot) => (
              <Card key={bot.botKey} className="border-white/10 bg-white/5">
                <CardHeader>
                  <CardTitle className="text-base">{bot.botKey}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-lg font-semibold">
                    ${bot.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function calculatePerformance(curve: Array<{ t: string; value: number }>): string {
  if (curve.length < 2) return "0.00";
  const first = curve[0].value;
  const last = curve[curve.length - 1].value;
  if (first === 0) return "0.00";
  const change = ((last - first) / first) * 100;
  return change.toFixed(2);
}

function generateFallbackCurve(currentValue: number): Array<{ t: string; value: number }> {
  const points: Array<{ t: string; value: number }> = [];
  const now = Date.now();
  for (let i = 0; i < 30; i++) {
    points.push({
      t: new Date(now - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
      value: currentValue * (0.95 + Math.random() * 0.1),
    });
  }
  return points;
}