"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";

// Use Next.js route handler proxy so we never hardcode ports in the UI.
const API_BASE = "";

type HorizonKey = "1w" | "2w" | "4w" | "52w";
const HORIZONS: HorizonKey[] = ["1w", "2w", "4w", "52w"];

type ViewMode = "long" | "mixed";

type TargetInfo = {
  expected_return?: number;
  target_price?: number;
  confidence?: number;
  direction?: "up" | "down" | "flat" | string;
};

type DisagreementInfo = {
  max_disagreement?: number;
  label?: "agree" | "weak" | "disagree" | string;
};

type SentimentInfo = {
  news_sentiment?: number;
  social_sentiment?: number;
};

type SymbolEntry = {
  symbol: string;
  name?: string;
  sector?: string;
  price?: number;
  targets?: Record<string, TargetInfo>;
  disagreement?: DisagreementInfo;
  sentiment?: SentimentInfo;
};

type PredictionFeed = {
  timestamp?: string;
  symbols?: Record<string, SymbolEntry>;
};

function formatPct(x?: number | null, digits = 1) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

function formatPrice(x?: number | null) {
  if (x == null || Number.isNaN(x)) return "—";
  return `$${x.toFixed(2)}`;
}

function disagreementBadge(info?: DisagreementInfo) {
  const label = (info?.label ?? "").toLowerCase();
  const level = info?.max_disagreement ?? 0;

  if (label === "disagree" || level > 0.5) return <Badge>High Disagreement</Badge>;
  if (label === "weak" || level > 0.25) return <Badge variant="outline">Mixed Views</Badge>;

  return <Badge variant="secondary">Aligned</Badge>;
}

export default function InsightsPage() {
  const [feed, setFeed] = useState<PredictionFeed | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [horizon, setHorizon] = useState<HorizonKey>("1w");
  const [viewMode, setViewMode] = useState<ViewMode>("long");
  const [priceBucket, setPriceBucket] = useState<string>("all");
  const [sector, setSector] = useState<string>("all");

  useEffect(() => {
    let mounted = true;

    const fetchFeed = async () => {
      try {
        setLoading(true);
        setError(null);

        const res = await fetch(
          `${API_BASE}/api/backend/api/insights/predictions/latest`,
          { cache: "no-store" }
        );

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = (await res.json()) as PredictionFeed;

        if (!mounted) return;
        setFeed(json);
      } catch (e: any) {
        if (!mounted) return;
        setError(e?.message ?? "Failed to load predictions");
        setFeed(null);
      } finally {
        if (!mounted) return;
        setLoading(false);
      }
    };

    fetchFeed();
    return () => {
      mounted = false;
    };
  }, []);

  const allSymbols = useMemo(() => {
    if (!feed?.symbols) return [];
    return Object.values(feed.symbols)
      .map((s) => ({
        ...s,
        _target: s.targets?.[horizon],
      }))
      .filter((s) => typeof (s as any)._target?.expected_return === "number") as Array<
      SymbolEntry & { _target: TargetInfo }
    >;
  }, [feed, horizon]);

  const sectors = useMemo(() => {
    const set = new Set<string>();
    allSymbols.forEach((s) => s.sector && set.add(s.sector));
    return Array.from(set).sort();
  }, [allSymbols]);

  const filtered = useMemo(() => {
    return allSymbols
      .filter((s) => {
        const exp = s._target.expected_return ?? 0;
        if (viewMode === "long" && exp <= 0) return false;
        return true;
      })
      .filter((s) => {
        const p = s.price ?? 0;
        if (priceBucket === "<1") return p < 1;
        if (priceBucket === "1-10") return p >= 1 && p < 10;
        if (priceBucket === "10-50") return p >= 10 && p < 50;
        if (priceBucket === "50-100") return p >= 50 && p < 100;
        if (priceBucket === ">100") return p >= 100;
        return true;
      })
      .filter((s) => (sector === "all" ? true : s.sector === sector))
      .sort((a, b) =>
        viewMode === "mixed"
          ? Math.abs((b._target.expected_return ?? 0)) - Math.abs((a._target.expected_return ?? 0))
          : (b._target.expected_return ?? 0) - (a._target.expected_return ?? 0)
      )
      .slice(0, 50);
  }, [allSymbols, viewMode, priceBucket, sector]);

  const viewModeOptions = useMemo(
    () => [
      { label: "Top Long Only", value: "long" },
      { label: "Mixed / Edge (Advanced)", value: "mixed" },
    ],
    []
  );

  const priceOptions = useMemo(
    () => [
      { label: "All Prices", value: "all" },
      { label: "< $1", value: "<1" },
      { label: "$1–$10", value: "1-10" },
      { label: "$10–$50", value: "10-50" },
      { label: "$50–$100", value: "50-100" },
      { label: "> $100", value: ">100" },
    ],
    []
  );

  const sectorOptions = useMemo(() => {
    return [
      { label: "All Sectors", value: "all" },
      ...sectors.map((s) => ({ label: s, value: s })),
    ];
  }, [sectors]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">AI Insights</h1>
          <p className="text-sm text-muted-foreground">
            Ranked price targets with confidence and regime-aware disagreement
          </p>
          {error && (
            <p className="mt-2 text-sm text-red-500">
              {error}
            </p>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <Select
            value={viewMode}
            onChange={(v) => setViewMode(v as ViewMode)}
            options={viewModeOptions}
            className="w-[190px]"
          />

          <Select
            value={priceBucket}
            onChange={setPriceBucket}
            options={priceOptions}
            className="w-[140px]"
          />

          <Select
            value={sector}
            onChange={setSector}
            options={sectorOptions}
            className="w-[160px]"
          />
        </div>
      </div>

      <Tabs value={horizon} onValueChange={(v) => setHorizon(v as HorizonKey)}>
        <TabsList>
          {HORIZONS.map((h) => (
            <TabsTrigger key={h} value={h}>
              {h.toUpperCase()}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value={horizon}>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-sm">
                Top 50 Signals {loading ? "(Loading…)" : `(${horizon.toUpperCase()})`}
              </CardTitle>
            </CardHeader>

            <CardContent className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-2">Symbol</th>
                    <th className="text-left hidden md:table-cell">Name</th>
                    <th className="text-left hidden md:table-cell">Sector</th>
                    <th className="text-right">Price</th>
                    <th className="text-right">Exp. Return</th>
                    <th className="text-right">Target</th>
                    <th className="text-right">Confidence</th>
                    <th className="text-center">Disagreement</th>
                  </tr>
                </thead>

                <tbody>
                  {filtered.map((row) => {
                    const t = row._target;
                    return (
                      <tr key={row.symbol} className="border-b hover:bg-muted/40">
                        <td className="font-semibold py-1.5">{row.symbol}</td>
                        <td className="hidden md:table-cell truncate">{row.name ?? "—"}</td>
                        <td className="hidden md:table-cell">{row.sector ?? "—"}</td>
                        <td className="text-right">{formatPrice(row.price)}</td>
                        <td className="text-right">{formatPct(t.expected_return)}</td>
                        <td className="text-right">{formatPrice(t.target_price)}</td>
                        <td className="text-right">{formatPct(t.confidence, 0)}</td>
                        <td className="text-center">{disagreementBadge(row.disagreement)}</td>
                      </tr>
                    );
                  })}

                  {filtered.length === 0 && !loading && (
                    <tr>
                      <td colSpan={8} className="py-4 text-center text-muted-foreground">
                        No signals match the current filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
