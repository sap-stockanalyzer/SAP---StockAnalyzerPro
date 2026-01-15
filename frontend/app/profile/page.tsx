"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

/**
 * AION Analytics — Profile / Portfolio page
 * Matches your sketch:
 *  - Big equity curve with range tabs
 *  - Right-side quick stats
 *  - Allocation donut + sector filters
 *  - Current holdings table
 *
 * Uses mock data for now so you can polish UI while the server migrates.
 */

type RangeKey = "1D" | "1W" | "1M" | "3M" | "6M" | "1Y";

type EquityPoint = {
  t: string; // label
  equity: number;
};

type Holding = {
  symbol: string;
  sector: string;
  qty: number;
  avg: number;
  last: number;
};

type AllocationSlice = {
  name: string;
  value: number; // %
};

const RANGES: { key: RangeKey; label: string; points: number }[] = [
  { key: "1D", label: "1D", points: 48 }, // ~30min-ish bars
  { key: "1W", label: "1W", points: 35 },
  { key: "1M", label: "1M", points: 30 },
  { key: "3M", label: "3M", points: 45 },
  { key: "6M", label: "6M", points: 60 },
  { key: "1Y", label: "1Y", points: 80 },
];

const SECTORS = ["Healthcare", "Energy", "Real Estate", "Financials", "Tech"] as const;

function fmtMoney(x: number) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x);
  return `${sign}$${v.toLocaleString(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  })}`;
}

function fmtPct(x: number) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}

// Deterministic-ish PRNG so UI doesn't "dance" on refresh
function makeRng(seed = 1337) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function buildMockEquity(range: RangeKey): EquityPoint[] {
  const meta = RANGES.find((r) => r.key === range) ?? RANGES[2];
  const n = meta.points;

  const rng = makeRng(
    range === "1D"
      ? 11
      : range === "1W"
      ? 22
      : range === "1M"
      ? 33
      : range === "3M"
      ? 44
      : range === "6M"
      ? 55
      : 66
  );

  // Base equity near your sketch number
  let equity = 67028.64;

  // Volatility per range
  const vol =
    range === "1D"
      ? 0.0015
      : range === "1W"
      ? 0.003
      : range === "1M"
      ? 0.004
      : range === "3M"
      ? 0.006
      : range === "6M"
      ? 0.008
      : 0.01;

  // Drift upward (your sketch trends up)
  const drift =
    range === "1D"
      ? 0.0002
      : range === "1W"
      ? 0.00035
      : range === "1M"
      ? 0.00045
      : range === "3M"
      ? 0.00055
      : range === "6M"
      ? 0.00065
      : 0.00075;

  const out: EquityPoint[] = [];
  for (let i = 0; i < n; i++) {
    const shock = (rng() - 0.5) * 2;
    const ret = drift + shock * vol;
    equity = equity * (1 + ret);

    const label =
      range === "1D"
        ? `${i}`
        : range === "1W"
        ? `D${i + 1}`
        : range === "1M"
        ? `D${i + 1}`
        : `W${i + 1}`;

    out.push({ t: label, equity: Math.round(equity * 100) / 100 });
  }

  return out;
}

function buildMockHoldings(): Holding[] {
  return [
    { symbol: "AAPL", sector: "Tech", qty: 18, avg: 171.25, last: 198.1 },
    { symbol: "MSFT", sector: "Tech", qty: 9, avg: 332.4, last: 417.2 },
    { symbol: "JPM", sector: "Financials", qty: 14, avg: 152.1, last: 168.55 },
    { symbol: "XOM", sector: "Energy", qty: 20, avg: 99.35, last: 108.7 },
    { symbol: "O", sector: "Real Estate", qty: 40, avg: 55.1, last: 53.8 },
    { symbol: "UNH", sector: "Healthcare", qty: 3, avg: 476.2, last: 522.35 },
  ];
}

function buildMockAllocation(holdings: Holding[]): AllocationSlice[] {
  const bySector = new Map<string, number>();
  for (const h of holdings) {
    const mv = h.qty * h.last;
    bySector.set(h.sector, (bySector.get(h.sector) ?? 0) + mv);
  }
  const total = Array.from(bySector.values()).reduce((a, b) => a + b, 0) || 1;

  const slices = Array.from(bySector.entries())
    .map(([name, mv]) => ({ name, value: (mv / total) * 100 }))
    .sort((a, b) => b.value - a.value);

  // Ensure all appear for checkbox list
  const existing = new Set(slices.map((s) => s.name));
  for (const s of SECTORS) {
    if (!existing.has(s)) slices.push({ name: s, value: 0 });
  }
  return slices;
}

function DeltaPill({ value }: { value: number }) {
  const up = value >= 0;
  return (
    <span
      className={[
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium border",
        up
          ? "bg-emerald-950/40 text-emerald-300 border-emerald-800/50"
          : "bg-rose-950/40 text-rose-300 border-rose-800/50",
      ].join(" ")}
    >
      {up ? "▲" : "▼"} {fmtMoney(value)}
    </span>
  );
}

function Card({
  title,
  children,
  right,
}: {
  title: string;
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-slate-800/70 bg-slate-950/40 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="flex items-center justify-between px-4 pt-4">
        <div className="text-sm font-semibold text-slate-200">{title}</div>
        {right ? <div className="text-xs text-slate-400">{right}</div> : null}
      </div>
      <div className="px-4 pb-4 pt-3">{children}</div>
    </div>
  );
}

export default function ProfilePage() {
  const [range, setRange] = useState<RangeKey>("1M");
  const [selectedSectors, setSelectedSectors] = useState<string[]>([...SECTORS]);

  const holdings = useMemo(() => buildMockHoldings(), []);
  const allocation = useMemo(() => buildMockAllocation(holdings), [holdings]);
  const equitySeries = useMemo(() => buildMockEquity(range), [range]);

  const summary = useMemo(() => {
    const cash = 27028.0;
    const invested = holdings.reduce((acc, h) => acc + h.qty * h.last, 0);
    const total = cash + invested;

    const costBasis = holdings.reduce((acc, h) => acc + h.qty * h.avg, 0);
    const totalProfit = invested - costBasis;

    const eq0 = equitySeries[0]?.equity ?? total;
    const eqN = equitySeries[equitySeries.length - 1]?.equity ?? total;
    const rangeProfit = eqN - eq0;

    return {
      cash,
      invested,
      total,
      totalProfit,
      rangeProfit,
      allocated: 0, // placeholder until bots reserve cash
      holdingsCount: holdings.length,
    };
  }, [holdings, equitySeries]);

  const filteredHoldings = useMemo(() => {
    return holdings
      .filter((h) => selectedSectors.includes(h.sector))
      .map((h) => {
        const mv = h.qty * h.last;
        const cb = h.qty * h.avg;
        const pnl = mv - cb;
        const pnlPct = cb > 0 ? pnl / cb : 0;
        return { ...h, mv, pnl, pnlPct };
      })
      .sort((a, b) => b.mv - a.mv);
  }, [holdings, selectedSectors]);

  const toggleSector = (sector: string) => {
    setSelectedSectors((prev) =>
      prev.includes(sector) ? prev.filter((s) => s !== sector) : [...prev, sector]
    );
  };

  const selectAllSectors = () => setSelectedSectors([...SECTORS]);
  const clearSectors = () => setSelectedSectors([]);

  const chartMinMax = useMemo(() => {
    const vals = equitySeries.map((p) => p.equity);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    return { min: min * 0.995, max: max * 1.005 };
  }, [equitySeries]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-950 to-black text-white">
      <div className="mx-auto w-full max-w-[1400px] px-6 py-8">
        {/* Header */}
        <div className="flex items-end justify-between gap-4">
          <div>
            <div className="text-3xl font-bold tracking-tight">Profile</div>
            <div className="mt-1 text-sm text-slate-400">
              Portfolio overview, allocations, and current holdings.
            </div>
          </div>

          {/* Range tabs */}
          <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-slate-800/70 bg-slate-950/40 p-2">
            {RANGES.map((r) => {
              const active = r.key === range;
              return (
                <button
                  key={r.key}
                  onClick={() => setRange(r.key)}
                  className={[
                    "rounded-xl px-3 py-1.5 text-sm font-medium transition",
                    active
                      ? "bg-sky-600/90 text-white shadow"
                      : "bg-slate-900/40 text-slate-300 hover:bg-slate-900/70",
                  ].join(" ")}
                >
                  {r.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Top grid: chart + stats */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Equity chart */}
          <div className="lg:col-span-8">
            <Card
              title="Equity Curve"
              right={
                <span className="inline-flex items-center gap-2">
                  <span className="text-slate-500">Range:</span>
                  <span className="text-slate-300">{range}</span>
                  <span className="text-slate-500">•</span>
                  <DeltaPill value={summary.rangeProfit} />
                </span>
              }
            >
              <div className="h-[360px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={equitySeries} margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                    <XAxis
                      dataKey="t"
                      tick={{ fill: "#94a3b8", fontSize: 12 }}
                      axisLine={{ stroke: "rgba(148,163,184,0.25)" }}
                      tickLine={{ stroke: "rgba(148,163,184,0.15)" }}
                      minTickGap={12}
                    />
                    <YAxis
                      domain={[chartMinMax.min, chartMinMax.max]}
                      tick={{ fill: "#94a3b8", fontSize: 12 }}
                      axisLine={{ stroke: "rgba(148,163,184,0.25)" }}
                      tickLine={{ stroke: "rgba(148,163,184,0.15)" }}
                      tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
                      width={80}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(2,6,23,0.92)",
                        border: "1px solid rgba(148,163,184,0.2)",
                        borderRadius: 12,
                        color: "white",
                      }}
                      labelStyle={{ color: "#cbd5e1" }}
                      formatter={(value: any) => [fmtMoney(Number(value)), "Equity"]}
                    />
                    <Line type="monotone" dataKey="equity" strokeWidth={2.5} dot={false} stroke="rgba(56,189,248,0.95)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Stats */}
          <div className="lg:col-span-4">
            <Card title="Totals">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs text-slate-400">Total Cash + Investments</div>
                  <div className="mt-1 text-2xl font-semibold">{fmtMoney(summary.total)}</div>
                </div>

                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs text-slate-400">Total Profit</div>
                  <div className="mt-1 text-2xl font-semibold">{fmtMoney(summary.totalProfit)}</div>
                </div>

                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs text-slate-400">Profit ({range})</div>
                  <div className="mt-1 text-xl font-semibold">{fmtMoney(summary.rangeProfit)}</div>
                </div>

                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs text-slate-400">Holdings</div>
                  <div className="mt-1 text-xl font-semibold">{summary.holdingsCount}</div>
                </div>

                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs text-slate-400">Cash</div>
                  <div className="mt-1 text-xl font-semibold">{fmtMoney(summary.cash)}</div>
                </div>

                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs text-slate-400">Invested</div>
                  <div className="mt-1 text-xl font-semibold">{fmtMoney(summary.invested)}</div>
                </div>

                <div className="col-span-2 rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs text-slate-400">Allocated (Bots)</div>
                      <div className="mt-1 text-lg font-semibold">{fmtMoney(summary.allocated)}</div>
                    </div>
                    <div className="text-xs text-slate-500">Placeholder until bots are wired.</div>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Bottom grid: allocation + holdings */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Allocation */}
          <div className="lg:col-span-4">
            <Card
              title="Allocation"
              right={
                <div className="flex items-center gap-2">
                  <button
                    onClick={selectAllSectors}
                    className="rounded-lg border border-slate-800/70 bg-slate-950/40 px-2 py-1 text-xs text-slate-300 hover:bg-slate-900/60"
                  >
                    All
                  </button>
                  <button
                    onClick={clearSectors}
                    className="rounded-lg border border-slate-800/70 bg-slate-950/40 px-2 py-1 text-xs text-slate-300 hover:bg-slate-900/60"
                  >
                    None
                  </button>
                </div>
              }
            >
              <div className="grid grid-cols-1 gap-4">
                <div className="h-[220px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={allocation}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={70}
                        outerRadius={95}
                        paddingAngle={2}
                        stroke="rgba(15,23,42,0.6)"
                      >
                        {allocation.map((_, idx) => (
                          <Cell
                            key={`cell-${idx}`}
                            fill={[
                              "rgba(56,189,248,0.85)",
                              "rgba(34,197,94,0.75)",
                              "rgba(245,158,11,0.75)",
                              "rgba(168,85,247,0.75)",
                              "rgba(244,63,94,0.70)",
                            ][idx % 5]}
                          />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          background: "rgba(2,6,23,0.92)",
                          border: "1px solid rgba(148,163,184,0.2)",
                          borderRadius: 12,
                          color: "white",
                        }}
                        formatter={(value: any, name: any) => [`${Number(value).toFixed(1)}%`, String(name)]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                <div className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-3">
                  <div className="text-xs font-semibold text-slate-200">Sectors</div>
                  <div className="mt-2 space-y-2">
                    {SECTORS.map((s) => {
                      const checked = selectedSectors.includes(s);
                      return (
                        <label key={s} className="flex items-center justify-between gap-3 text-sm text-slate-300">
                          <span className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggleSector(s)}
                              className="h-4 w-4 accent-sky-500"
                            />
                            {s}
                          </span>
                          <span className="text-xs text-slate-500">
                            {allocation.find((a) => a.name === s)?.value.toFixed(1)}%
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Holdings */}
          <div className="lg:col-span-8">
            <Card title="Current Holdings">
              <div className="overflow-x-auto">
                <table className="w-full border-separate border-spacing-0">
                  <thead>
                    <tr className="text-left text-xs text-slate-400">
                      <th className="border-b border-slate-800/70 pb-2 pr-3">Ticker</th>
                      <th className="border-b border-slate-800/70 pb-2 pr-3">Sector</th>
                      <th className="border-b border-slate-800/70 pb-2 pr-3">Amount</th>
                      <th className="border-b border-slate-800/70 pb-2 pr-3">Avg Price</th>
                      <th className="border-b border-slate-800/70 pb-2 pr-3">Last</th>
                      <th className="border-b border-slate-800/70 pb-2 pr-3">Market Value</th>
                      <th className="border-b border-slate-800/70 pb-2">Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredHoldings.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-6 text-center text-sm text-slate-500">
                          No holdings match the selected sectors.
                        </td>
                      </tr>
                    ) : (
                      filteredHoldings.map((h) => {
                        const up = h.pnl >= 0;
                        return (
                          <tr key={h.symbol} className="text-sm">
                            <td className="border-b border-slate-900/60 py-3 pr-3 font-semibold text-slate-100">
                              {h.symbol}
                            </td>
                            <td className="border-b border-slate-900/60 py-3 pr-3 text-slate-300">{h.sector}</td>
                            <td className="border-b border-slate-900/60 py-3 pr-3 text-slate-200">{h.qty}</td>
                            <td className="border-b border-slate-900/60 py-3 pr-3 text-slate-200">{fmtMoney(h.avg)}</td>
                            <td className="border-b border-slate-900/60 py-3 pr-3 text-slate-200">{fmtMoney(h.last)}</td>
                            <td className="border-b border-slate-900/60 py-3 pr-3 text-slate-200">{fmtMoney(h.mv)}</td>
                            <td className="border-b border-slate-900/60 py-3">
                              <div className="flex items-center justify-between gap-3">
                                <span className={up ? "text-emerald-300" : "text-rose-300"}>{fmtMoney(h.pnl)}</span>
                                <span className="text-xs text-slate-500">{fmtPct(h.pnlPct)}</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>

              <div className="mt-3 text-xs text-slate-500">
                Tip: sector checkboxes filter the holdings table (and later can filter the chart too).
              </div>
            </Card>
          </div>
        </div>

        <div className="mt-6 text-center text-xs text-slate-600">
          Mock data mode — swap in real backend calls once your server migration completes.
        </div>
      </div>
    </div>
  );
}