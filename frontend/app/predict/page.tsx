"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, AlertCircle } from "lucide-react";
import { getApiBaseUrl } from "@/lib/api";

// API endpoints via Next.js proxy
const API_BASE = getApiBaseUrl();

// Type definitions
type LivePrice = {
  symbol?: string;
  price?: number;
  change?: number;
  change_percent?: number;
  volume?: number;
  market_cap?: number;
  pe_ratio?: number;
  sector?: string;
  industry?: string;
  name?: string;
};

type PredictionHorizon = {
  predicted_return?: number;
  confidence?: number;
  target_price?: number;
  direction?: string;
};

type ModelPrediction = {
  symbol?: string;
  predictions?: {
    "1d"?: PredictionHorizon;
    "1w"?: PredictionHorizon;
    "2w"?: PredictionHorizon;
    "4w"?: PredictionHorizon;
    "13w"?: PredictionHorizon;
    "26w"?: PredictionHorizon;
    "52w"?: PredictionHorizon;
  };
};

type IntradaySignal = {
  score?: number;
  action?: string;
  position?: string;
  confidence?: number;
};

// Utility functions
function formatPrice(price?: number): string {
  if (price === undefined || price === null || isNaN(price)) return "—";
  return `$${price.toFixed(2)}`;
}

function formatPercent(value?: number): string {
  if (value === undefined || value === null || isNaN(value)) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function formatLargeNumber(value?: number): string {
  if (value === undefined || value === null || isNaN(value)) return "—";
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toFixed(0)}`;
}

function formatVolume(volume?: number): string {
  if (volume === undefined || volume === null || isNaN(volume)) return "—";
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toFixed(0);
}

// Main component
function PredictPageContent() {
  const searchParams = useSearchParams();
  const ticker = searchParams.get("ticker")?.toUpperCase() || "";

  const [livePrice, setLivePrice] = useState<LivePrice | null>(null);
  const [predictions, setPredictions] = useState<ModelPrediction | null>(null);
  const [intradaySignal, setIntradaySignal] = useState<IntradaySignal | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) {
      setError("No ticker symbol provided");
      setLoading(false);
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch live prices
        const priceRes = await fetch(
          `${API_BASE}/api/live-prices/prices?symbols=${ticker}`,
          { cache: "no-store" }
        );

        if (!priceRes.ok) {
          throw new Error(`Failed to fetch price data (${priceRes.status})`);
        }

        const priceData = await priceRes.json();
        const priceInfo = priceData?.prices?.[ticker] || priceData?.[ticker] || null;
        setLivePrice(priceInfo);

        // Fetch predictions
        try {
          const predRes = await fetch(
            `${API_BASE}/api/models/predict?symbols=${ticker}`,
            { cache: "no-store" }
          );

          if (predRes.ok) {
            const predData = await predRes.json();
            const predInfo = predData?.predictions?.[ticker] || predData?.[ticker] || null;
            setPredictions(predInfo);
          }
        } catch (e) {
          console.warn("Failed to fetch predictions:", e);
        }

        // Fetch intraday signals (optional)
        try {
          const intradayRes = await fetch(
            `${API_BASE}/api/intraday/symbol/${ticker}`,
            { cache: "no-store" }
          );

          if (intradayRes.ok) {
            const intradayData = await intradayRes.json();
            setIntradaySignal(intradayData);
          }
        } catch (e) {
          console.warn("Failed to fetch intraday signals:", e);
        }
      } catch (e: any) {
        setError(e?.message || "Failed to load ticker data");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [ticker]);

  if (!ticker) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <Card className="border-red-500/40 bg-red-500/10">
            <CardContent className="py-8">
              <div className="flex items-center gap-3 text-red-200">
                <AlertCircle className="h-5 w-5" />
                <p>No ticker symbol provided. Please search for a ticker.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="h-32 bg-gray-800/50 rounded-2xl" />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="h-64 bg-gray-800/50 rounded-2xl" />
              <div className="h-64 bg-gray-800/50 rounded-2xl" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !livePrice) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <Card className="border-red-500/40 bg-red-500/10">
            <CardContent className="py-8">
              <div className="flex items-center gap-3 text-red-200">
                <AlertCircle className="h-5 w-5" />
                <p>{error || "Failed to load ticker data"}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  const priceChange = livePrice.change || 0;
  const priceChangePercent = livePrice.change_percent || 0;
  const isPositive = priceChange >= 0;

  const horizons = ["1d", "1w", "2w", "4w", "13w", "26w", "52w"];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white px-4 py-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header Section */}
        <Card className="border-gray-800">
          <CardContent className="py-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-3xl font-bold">{ticker}</h1>
                  {livePrice.name && (
                    <span className="text-xl text-gray-400">{livePrice.name}</span>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-3">
                  <div className="text-4xl font-bold">{formatPrice(livePrice.price)}</div>
                  <div
                    className={`flex items-center gap-1 text-lg ${
                      isPositive ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {isPositive ? (
                      <TrendingUp className="h-5 w-5" />
                    ) : (
                      <TrendingDown className="h-5 w-5" />
                    )}
                    <span>
                      {formatPrice(Math.abs(priceChange))} ({formatPercent(priceChangePercent)})
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap gap-4">
                {livePrice.market_cap && (
                  <div>
                    <div className="text-xs text-gray-400 uppercase">Market Cap</div>
                    <div className="text-lg font-semibold">
                      {formatLargeNumber(livePrice.market_cap)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ML Predictions Card */}
          <Card className="border-gray-800">
            <CardHeader>
              <CardTitle className="text-base">ML Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              {predictions?.predictions ? (
                <div className="space-y-3">
                  {horizons.map((horizon) => {
                    const pred = predictions.predictions?.[horizon as keyof typeof predictions.predictions];
                    if (!pred) return null;

                    const isUp = (pred.predicted_return || 0) >= 0;

                    return (
                      <div
                        key={horizon}
                        className="flex items-center justify-between p-3 rounded-lg border border-gray-700 bg-gray-800/30"
                      >
                        <div className="flex items-center gap-3">
                          <Badge variant="outline" className="min-w-[48px] justify-center">
                            {horizon.toUpperCase()}
                          </Badge>
                          <div>
                            <div className="text-sm font-medium">
                              {formatPercent(pred.predicted_return)}
                            </div>
                            <div className="text-xs text-gray-400">
                              Target: {formatPrice(pred.target_price)}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div
                            className={`text-sm font-semibold ${
                              isUp ? "text-green-400" : "text-red-400"
                            }`}
                          >
                            {isUp ? "↑" : "↓"} {pred.direction || ""}
                          </div>
                          <div className="text-xs text-gray-400">
                            Conf: {formatPercent(pred.confidence)}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-sm text-gray-400 py-6 text-center">
                  No predictions available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Fundamentals Card */}
          <Card className="border-gray-800">
            <CardHeader>
              <CardTitle className="text-base">Fundamentals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 rounded-lg border border-gray-700 bg-gray-800/30">
                    <div className="text-xs text-gray-400 uppercase mb-1">P/E Ratio</div>
                    <div className="text-lg font-semibold">
                      {livePrice.pe_ratio ? livePrice.pe_ratio.toFixed(2) : "—"}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg border border-gray-700 bg-gray-800/30">
                    <div className="text-xs text-gray-400 uppercase mb-1">Volume</div>
                    <div className="text-lg font-semibold">
                      {formatVolume(livePrice.volume)}
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-lg border border-gray-700 bg-gray-800/30">
                  <div className="text-xs text-gray-400 uppercase mb-1">Sector</div>
                  <div className="text-sm font-medium">{livePrice.sector || "—"}</div>
                </div>

                <div className="p-3 rounded-lg border border-gray-700 bg-gray-800/30">
                  <div className="text-xs text-gray-400 uppercase mb-1">Industry</div>
                  <div className="text-sm font-medium">{livePrice.industry || "—"}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Intraday Signals Card */}
          {intradaySignal && (
            <Card className="border-gray-800 lg:col-span-2">
              <CardHeader>
                <CardTitle className="text-base">Intraday Signals</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 rounded-lg border border-gray-700 bg-gray-800/30">
                    <div className="text-xs text-gray-400 uppercase mb-2">Score</div>
                    <div className="text-2xl font-bold">
                      {intradaySignal.score !== undefined
                        ? intradaySignal.score.toFixed(2)
                        : "—"}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border border-gray-700 bg-gray-800/30">
                    <div className="text-xs text-gray-400 uppercase mb-2">Action</div>
                    <div className="text-xl font-semibold">
                      {intradaySignal.action || "—"}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border border-gray-700 bg-gray-800/30">
                    <div className="text-xs text-gray-400 uppercase mb-2">Position</div>
                    <div className="text-xl font-semibold">
                      {intradaySignal.position || "—"}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// Export with Suspense wrapper
export default function PredictPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white px-4 py-8">
          <div className="max-w-7xl mx-auto">
            <div className="animate-pulse space-y-6">
              <div className="h-32 bg-gray-800/50 rounded-2xl" />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="h-64 bg-gray-800/50 rounded-2xl" />
                <div className="h-64 bg-gray-800/50 rounded-2xl" />
              </div>
            </div>
          </div>
        </div>
      }
    >
      <PredictPageContent />
    </Suspense>
  );
}