"use client";

import * as React from "react";
import { useEffect, useState, useCallback } from "react";
import { RefreshCw } from "lucide-react";

import { Select, type SelectOption } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";

type BotMode = "paper" | "live";

interface RawBotPosition {
  symbol: string;
  qty: number;
  entry: number;
  stop: number;
  target: number;
  last_price?: number | null;
  unrealized_pnl?: number | null;
}

interface RawBotState {
  cash: number;
  equity: number;
  num_positions: number;
  positions: RawBotPosition[];
}

interface StatusApiResponse {
  price_status: string;
  bots: Record<string, RawBotState>;
}

/** UI-facing summary for each bot */
interface BotSummary {
  id: string;
  name: string;
  mode: BotMode;
  status: "idle" | "running" | "error";
  pnlToday: number;
  pnlTotal: number;
  positions: number;
  lastRunAt?: string | null;
}

/** A flattened position row for the table */
interface PositionRow {
  id: string;
  botId: string;
  symbol: string;
  qty: number;
  entry: number;
  lastPrice?: number | null;
  unrealizedPnl?: number | null;
  stop?: number;
  target?: number;
}

/** Options for mode filter */
const MODE_OPTIONS: SelectOption[] = [
  { label: "All modes", value: "all" },
  { label: "Paper only", value: "paper" },
  { label: "Live only", value: "live" },
];

function formatUsd(v: number): string {
  if (!Number.isFinite(v)) return "-";
  const sign = v < 0 ? "-" : "";
  const abs = Math.abs(v);
  return `${sign}$${abs.toFixed(2)}`;
}

function formatDateTime(ts?: string | null): string {
  if (!ts) return "—";
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleString();
  } catch {
    return "—";
  }
}

export default function EodBotsPanel() {
  const [modeFilter, setModeFilter] = useState<string>("all");
  const [bots, setBots] = useState<BotSummary[]>([]);
  const [positions, setPositions] = useState<PositionRow[]>([]);
  const [priceStatus, setPriceStatus] = useState<string>("unknown");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadBots = useCallback(async () => {
    try {
      setError(null);
      setRefreshing(true);

      // Use proxy route instead of direct backend call
      const res = await fetch("/api/backend/eod/status");

      if (!res.ok) {
        throw new Error(`failed (status ${res.status})`);
      }

      const data = (await res.json()) as StatusApiResponse;

      setPriceStatus(data.price_status ?? "unknown");

      const rawBots = data.bots ?? {};

      // ---- Build BotSummary[] for the top cards + table ----
      const uiBots: BotSummary[] = Object.entries(rawBots).map(
        ([key, b]): BotSummary => {
          const name =
            key
              .replace(/^eod_/, "")
              .replace(/_/g, " ")
              .toUpperCase() || key.toUpperCase();

          // Rough lifetime P&L proxy: equity - cash (unrealized-ish)
          const pnlTotal = (b.equity ?? 0) - (b.cash ?? 0);

          return {
            id: key,
            name,
            mode: "paper", // default until you add live routing
            status: "idle", // can later wire to bot status logs
            pnlToday: 0, // wire from logs later
            pnlTotal,
            positions: b.num_positions ?? (b.positions?.length ?? 0),
            lastRunAt: null, // wire from logs later
          };
        }
      );

      // ---- Flatten positions across bots ----
      const uiPositions: PositionRow[] = [];
      Object.entries(rawBots).forEach(([botKey, b]) => {
        (b.positions || []).forEach((p, idx) => {
          uiPositions.push({
            id: `${botKey}-${p.symbol}-${idx}`,
            botId: botKey,
            symbol: p.symbol,
            qty: p.qty,
            entry: p.entry,
            lastPrice: p.last_price,
            unrealizedPnl: p.unrealized_pnl,
            stop: p.stop,
            target: p.target,
          });
        });
      });

      // Sort positions by |unrealized PnL| desc (most “interesting” first)
      uiPositions.sort((a, b) => {
        const av = Math.abs(a.unrealizedPnl ?? 0);
        const bv = Math.abs(b.unrealizedPnl ?? 0);
        return bv - av;
      });

      setBots(uiBots);
      setPositions(uiPositions);
    } catch (err: any) {
      console.error("Error: failed at loadBots", err);
      setError("Failed to load EOD bot status from backend.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  // Load once on mount
  useEffect(() => {
    loadBots();
  }, [loadBots]);

  const filteredBots =
    modeFilter === "all"
      ? bots
      : bots.filter((b) => b.mode === (modeFilter as BotMode));

  const topPositions = positions.slice(0, 50);

  const totalToday = bots.reduce((acc, b) => acc + (b.pnlToday ?? 0), 0);
  const totalLifetime = bots.reduce(
    (acc, b) => acc + (b.pnlTotal ?? 0),
    0
  );

  return (
    <div className="space-y-4">
      {/* Header + controls */}
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">
            End-of-Day / Swing Bots
          </h2>
          <p className="text-xs text-slate-400">
            Bots that execute around the market close using AION’s EOD
            predictions.
          </p>
          <p className="mt-1 text-[11px] text-slate-500">
            Price feed status:{" "}
            <span
              className={
                priceStatus === "ok"
                  ? "text-emerald-400"
                  : "text-amber-300"
              }
            >
              {priceStatus}
            </span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Select
            value={modeFilter}
            onChange={setModeFilter}
            options={MODE_OPTIONS}
            className="w-40"
            placeholder="Mode filter"
          />

          <button
            type="button"
            onClick={loadBots}
            disabled={refreshing}
            className="inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-100 hover:border-slate-500 disabled:opacity-60"
          >
            <RefreshCw
              size={14}
              className={refreshing ? "animate-spin" : "opacity-70"}
            />
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {/* Summary row */}
      {loading ? (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="text-xs text-slate-400">Active bots</div>
            <div className="mt-1 text-2xl font-semibold text-slate-50">
              {filteredBots.length}
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {bots.length} total configured
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="text-xs text-slate-400">Today P&L (all bots)</div>
            <div className="mt-1 text-2xl font-semibold text-emerald-400">
              {formatUsd(totalToday)}
            </div>
            <div className="mt-1 text-xs text-slate-500">
              (will populate once log wiring is added)
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <div className="text-xs text-slate-400">
              Lifetime P&L (proxy: equity - cash)
            </div>
            <div className="mt-1 text-2xl font-semibold text-slate-50">
              {formatUsd(totalLifetime)}
            </div>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="rounded-md border border-red-500/50 bg-red-950/40 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}

      {/* Bots table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60">
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Bot status
          </h3>
          <span className="text-[11px] text-slate-500">
            Mode filter: {modeFilter}
          </span>
        </div>

        {loading ? (
          <div className="space-y-2 p-4">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ) : filteredBots.length === 0 ? (
          <div className="p-4 text-xs text-slate-400">
            No bots match the current filter.
          </div>
        ) : (
          <div className="max-h-80 overflow-auto">
            <table className="min-w-full border-t border-slate-800 text-xs">
              <thead className="bg-slate-950/50 text-slate-400">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Bot</th>
                  <th className="px-3 py-2 text-left font-medium">Mode</th>
                  <th className="px-3 py-2 text-left font-medium">Status</th>
                  <th className="px-3 py-2 text-right font-medium">
                    Today P&L
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    Lifetime P&L
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    Positions
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    Last run
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {filteredBots.map((b) => (
                  <tr key={b.id}>
                    <td className="px-3 py-2 text-slate-100">{b.name}</td>
                    <td className="px-3 py-2 capitalize text-slate-300">
                      {b.mode}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-medium ${
                          b.status === "running"
                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/40"
                            : b.status === "error"
                            ? "bg-red-500/10 text-red-400 border border-red-500/40"
                            : "bg-slate-700/40 text-slate-200 border border-slate-500/50"
                        }`}
                      >
                        {b.status}
                      </span>
                    </td>
                    <td
                      className={`px-3 py-2 text-right ${
                        b.pnlToday >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {formatUsd(b.pnlToday)}
                    </td>
                    <td
                      className={`px-3 py-2 text-right ${
                        b.pnlTotal >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {formatUsd(b.pnlTotal)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-200">
                      {b.positions}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-400">
                      {formatDateTime(b.lastRunAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Active positions (all bots) */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60">
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Active positions (all EOD bots)
          </h3>
          <span className="text-[11px] text-slate-500">
            Showing top {topPositions.length} by P&L magnitude
          </span>
        </div>

        {loading ? (
          <div className="space-y-2 p-4">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ) : topPositions.length === 0 ? (
          <div className="p-4 text-xs text-slate-400">
            No open positions found for any EOD bots.
          </div>
        ) : (
          <div className="max-h-80 overflow-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-950/50 text-slate-400">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Bot</th>
                  <th className="px-3 py-2 text-left font-medium">Symbol</th>
                  <th className="px-3 py-2 text-right font-medium">Qty</th>
                  <th className="px-3 py-2 text-right font-medium">Entry</th>
                  <th className="px-3 py-2 text-right font-medium">Last</th>
                  <th className="px-3 py-2 text-right font-medium">
                    Unrealized P&L
                  </th>
                  <th className="px-3 py-2 text-right font-medium">Stop</th>
                  <th className="px-3 py-2 text-right font-medium">Target</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {topPositions.map((p) => (
                  <tr key={p.id}>
                    <td className="px-3 py-2 text-slate-200">{p.botId}</td>
                    <td className="px-3 py-2 font-mono text-slate-100">
                      {p.symbol}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-200">
                      {p.qty}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-200">
                      {p.entry.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-200">
                      {p.lastPrice != null ? p.lastPrice.toFixed(2) : "—"}
                    </td>
                    <td
                      className={`px-3 py-2 text-right ${
                        (p.unrealizedPnl ?? 0) >= 0
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {p.unrealizedPnl != null
                        ? formatUsd(p.unrealizedPnl)
                        : "—"}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-400">
                      {p.stop != null ? p.stop.toFixed(2) : "—"}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-400">
                      {p.target != null ? p.target.toFixed(2) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
