"use client";

import * as React from "react";
import { useEffect, useMemo, useRef, useState, memo } from "react";
import Link from "next/link";

import { Activity, Clock, DollarSign, Shield, SlidersHorizontal, RefreshCw, Settings } from "lucide-react";

import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

// Import centralized bots API client and types
import {
  getBotsPageBundle,
  detectApiPrefix,
  updateEodBotConfig,
  updateIntradayBotConfig,
} from "@/lib/botsApi";

import type {
  BotsPageBundle,
  EodStatusResponse,
  EodConfigResponse,
  IntradayPnlResponse,
  IntradayFill,
  IntradaySignal,
  BotDraft,
} from "@/lib/botsTypes";

// -----------------------------
// Constants
// -----------------------------



function fmtMoney(n: any): string {
  const x = Number(n);
  if (!isFinite(x)) return "$0";
  try {
    return x.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  } catch {
    return `$${Math.round(x)}`;
  }
}

function fmtMaybeMoney(n: any): string {
  if (n === null || n === undefined) return "—";
  const x = Number(n);
  if (!isFinite(x)) return "—";
  return fmtMoney(x);
}

function fmtPct(n: any): string {
  const x = Number(n);
  if (!isFinite(x)) return "0%";
  return `${Math.round(x * 100)}%`;
}

function clampNum(n: any, fallback: number): number {
  const x = Number(n);
  return isFinite(x) ? x : fallback;
}

function loadLocal<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function saveLocal<T>(key: string, value: T) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore
  }
}

// -----------------------------
// UI bits
// -----------------------------

const HORIZONS = ["1d", "1w", "1m", "3m", "6m", "1y"] as const;
type Horizon = (typeof HORIZONS)[number];

function OnOffDot({
  enabled,
  onToggle,
  disabled,
}: {
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={!!disabled}
      aria-label={enabled ? "Bot enabled" : "Bot disabled"}
      title={enabled ? "Enabled (click to turn off)" : "Disabled (click to turn on)"}
      className={[
        "h-10 w-10 rounded-full border transition",
        "shadow-sm",
        enabled ? "bg-green-500/80 border-green-400" : "bg-red-500/70 border-red-400",
        disabled ? "opacity-50 cursor-not-allowed" : "hover:scale-[1.03]",
      ].join(" ")}
    />
  );
}

function MiniPerfChart({ data }: { data: Array<{ t: string; value: number }> }) {
  const chartConfig = {
    value: {
      label: "Equity",
      color: "hsl(210, 100%, 60%)",
    },
  } satisfies ChartConfig;

  // Increased height from 120px to 140px to better accommodate chart with grid lines
  return (
    <div className="h-[140px] w-full">
      <ChartContainer config={chartConfig} className="h-full w-full">
        <LineChart
          data={data}
          margin={{ left: 0, right: 0, top: 5, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
          <XAxis dataKey="t" hide />
          <YAxis hide domain={["auto", "auto"]} />
          <ChartTooltip
            content={<ChartTooltipContent hideLabel />}
            cursor={{ stroke: "rgba(255,255,255,0.2)" }}
          />
          <Line
            dataKey="value"
            type="monotone"
            stroke="var(--color-value)"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}

function horizonButtons(h: Horizon, setH: (x: Horizon) => void) {
  return (
    <div className="flex flex-wrap gap-1">
      {HORIZONS.map((x) => (
        <Button
          key={x}
          type="button"
          variant={x === h ? "default" : "outline"}
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={() => setH(x)}
        >
          {x.toUpperCase()}
        </Button>
      ))}
    </div>
  );
}

function buildFallbackSeries(value?: number): Array<{ t: string; value: number }> {
  const v = clampNum(value, 0);
  const now = Date.now();
  return Array.from({ length: 12 }, (_, i) => {
    const t = new Date(now - (11 - i) * 60_000).toISOString();
    const jiggle = (i % 3 === 0 ? -1 : 1) * (i % 4) * 0.002;
    return { t, value: v * (1 + jiggle) };
  });
}

function coerceCurve(curve: any, fallbackValue?: number): Array<{ t: string; value: number }> {
  if (Array.isArray(curve) && curve.length) {
    const out: Array<{ t: string; value: number }> = [];
    for (let i = 0; i < curve.length; i++) {
      const row = curve[i];
      if (typeof row === "number") {
        out.push({ t: String(i), value: clampNum(row, 0) });
      } else if (row && typeof row === "object") {
        const t = String((row as any).t ?? (row as any).ts ?? (row as any).time ?? i);
        const v = clampNum((row as any).value ?? (row as any).v ?? (row as any).equity ?? (row as any).pnl, 0);
        out.push({ t, value: v });
      }
    }
    if (out.length) return out;
  }
  return buildFallbackSeries(fallbackValue);
}

// -----------------------------
// Draft hook
// -----------------------------

function useBotDraft(storageKey: string, base: BotDraft) {
  const normalizedBase = useMemo<BotDraft>(() => {
    return {
      max_alloc: clampNum(base.max_alloc, 10_000),
      max_positions: Math.max(0, Math.round(clampNum(base.max_positions, 10))),
      stop_loss: Math.max(0, clampNum(base.stop_loss, 0.05)),
      take_profit: Math.max(0, clampNum(base.take_profit, 0.10)),
      aggression: Math.min(1, Math.max(0, clampNum(base.aggression, 0.5))),
      enabled: base.enabled ?? true,
      penny_only: !!base.penny_only,
      allow_etfs: base.allow_etfs ?? true,
      max_daily_trades: base.max_daily_trades ?? 6,
    };
  }, [base]);

  const [draft, setDraft] = useState<BotDraft>(normalizedBase);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const cached = loadLocal<BotDraft>(storageKey, normalizedBase);
    setDraft({ ...normalizedBase, ...cached });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey]);

  function setField<K extends keyof BotDraft>(k: K, v: BotDraft[K]) {
    setDraft((p) => {
      const next = { ...p, [k]: v };
      saveLocal(storageKey, next);
      return next;
    });
    setDirty(true);
  }

  function reset() {
    setDraft(normalizedBase);
    saveLocal(storageKey, normalizedBase);
    setDirty(false);
  }

  return { draft, dirty, saving, setSaving, setField, reset, setDirty };
}

// -----------------------------
// Rules panel
// -----------------------------

function BotRulesPanel({
  botKey,
  botType,
  draft,
  setField,
  dirty,
  saving,
  onSave,
  onReset,
}: {
  botKey: string;
  botType: "swing" | "dt";
  draft: BotDraft;
  setField: <K extends keyof BotDraft>(k: K, v: BotDraft[K]) => void;
  dirty: boolean;
  saving: boolean;
  onSave: () => void;
  onReset: () => void;
}) {
  return (
    <div className="h-full rounded-xl border bg-card/40 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
          <div>
            <div className="text-sm font-semibold">Bot Rules</div>
            <div className="text-xs text-muted-foreground">
              {botType === "swing" ? "Swing / horizon bot" : "Day-trading bot"} • {botKey}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {dirty ? <Badge variant="muted">Unsaved</Badge> : <Badge variant="outline">Saved</Badge>}
        </div>
      </div>

      <Separator className="my-3" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-3">
          <div className="text-xs font-semibold text-muted-foreground">Allocation & Limits</div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">Max Spend</Label>
              <Input
                type="number"
                value={draft.max_alloc}
                onChange={(e) => setField("max_alloc", clampNum(e.target.value, draft.max_alloc))}
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Max Holdings</Label>
              <Input
                type="number"
                value={draft.max_positions}
                onChange={(e) =>
                  setField("max_positions", Math.max(0, Math.round(clampNum(e.target.value, draft.max_positions))))
                }
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center justify-between gap-3 rounded-lg border bg-background/40 p-3">
              <div className="text-xs">
                <div className="font-medium">Penny-only</div>
                <div className="text-muted-foreground">Filter to low-priced</div>
              </div>
              <Switch checked={!!draft.penny_only} onCheckedChange={(v) => setField("penny_only", v)} />
            </div>
            <div className="flex items-center justify-between gap-3 rounded-lg border bg-background/40 p-3">
              <div className="text-xs">
                <div className="font-medium">Allow ETFs</div>
                <div className="text-muted-foreground">ETFs permitted</div>
              </div>
              <Switch checked={!!draft.allow_etfs} onCheckedChange={(v) => setField("allow_etfs", v)} />
            </div>
          </div>

          {botType === "dt" ? (
            <div className="space-y-1">
              <Label className="text-xs">Max Daily Trades</Label>
              <Input
                type="number"
                value={draft.max_daily_trades ?? 6}
                onChange={(e) => setField("max_daily_trades", Math.max(0, Math.round(clampNum(e.target.value, 6))))}
              />
            </div>
          ) : null}
        </div>

        <div className="space-y-3">
          <div className="text-xs font-semibold text-muted-foreground">Risk & Exits</div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">Stop Loss</Label>
              <Input
                type="number"
                step="0.01"
                value={draft.stop_loss}
                onChange={(e) => setField("stop_loss", Math.max(0, clampNum(e.target.value, draft.stop_loss)))}
              />
              <div className="text-[11px] text-muted-foreground">Shown as {fmtPct(draft.stop_loss)}</div>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Take Profit</Label>
              <Input
                type="number"
                step="0.01"
                value={draft.take_profit}
                onChange={(e) => setField("take_profit", Math.max(0, clampNum(e.target.value, draft.take_profit)))}
              />
              <div className="text-[11px] text-muted-foreground">Shown as {fmtPct(draft.take_profit)}</div>
            </div>
          </div>

          <div className="space-y-1">
            <Label className="text-xs">Aggressiveness</Label>
            <Input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={draft.aggression}
              onChange={(e) => setField("aggression", clampNum(e.target.value, draft.aggression))}
            />
            <div className="text-[11px] text-muted-foreground">
              {draft.aggression < 0.34 ? "Conservative" : draft.aggression < 0.67 ? "Balanced" : "Aggressive"} •{" "}
              {fmtPct(draft.aggression)}
            </div>
          </div>

          <div className="mt-2 flex items-center justify-end gap-2">
            <Button type="button" variant="outline" onClick={onReset} disabled={saving}>
              Reset defaults
            </Button>
            <Button type="button" onClick={onSave} disabled={saving || !dirty}>
              {saving ? "Saving…" : "Save"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// -----------------------------
// Live tape
// -----------------------------

function LiveIntradayTape({
  fills,
  signals,
  updatedAt,
}: {
  fills: IntradayFill[];
  signals: IntradaySignal[];
  updatedAt?: string | null;
}) {
  const topFills = fills.slice(0, 10);
  const topSignals = signals.slice(0, 10);

  return (
    <Card className="border-white/10 bg-white/5">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="text-lg">Intraday Live Tape</CardTitle>
            <div className="mt-1 text-xs text-white/60">
              {updatedAt ? `Updated: ${updatedAt}` : "Updated: —"} • fills={fills.length} • signals={signals.length}
            </div>
          </div>
          <Badge variant="outline">FROM /bots/page</Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-xl border bg-card/30 p-4">
            <div className="mb-2 text-sm font-semibold">Recent fills</div>
            {topFills.length ? (
              <div className="space-y-2">
                {topFills.map((f, i) => {
                  const side = ((f.side ?? f.action ?? "—") as string).toUpperCase();
                  return (
                    <div key={i} className="flex items-center justify-between rounded-lg border bg-background/40 px-3 py-2 text-xs">
                      <div className="flex items-center gap-2">
                        <Badge variant={side === "BUY" ? "muted" : "outline"}>{side}</Badge>
                        <span className="font-semibold">{(f.symbol ?? "—").toString()}</span>
                        <span className="text-white/60">{(f.ts ?? f.time ?? "—").toString()}</span>
                      </div>
                      <div className="text-white/70">
                        {f.qty != null ? `${f.qty} @ ` : ""}
                        {f.price != null ? fmtMoney(f.price) : "—"}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-xs text-white/60">No fills parsed yet (backend will populate as artifacts appear).</div>
            )}
          </div>

          <div className="rounded-xl border bg-card/30 p-4">
            <div className="mb-2 text-sm font-semibold">Latest signals</div>
            {topSignals.length ? (
              <div className="space-y-2">
                {topSignals.map((s, i) => {
                  const act = ((s.action ?? s.side ?? "—") as string).toUpperCase();
                  return (
                    <div key={i} className="flex items-center justify-between rounded-lg border bg-background/40 px-3 py-2 text-xs">
                      <div className="flex items-center gap-2">
                        <Badge variant={act === "BUY" ? "muted" : "outline"}>{act}</Badge>
                        <span className="font-semibold">{(s.symbol ?? "—").toString()}</span>
                        <span className="text-white/60">{(s.ts ?? s.time ?? "—").toString()}</span>
                      </div>
                      <div className="text-white/70">
                        {s.confidence != null ? `conf ${Math.round(Number(s.confidence) * 100)}%` : "conf —"}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-xs text-white/60">No signals parsed yet (ensure dt_signals artifacts exist).</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// -----------------------------
// Bot row card
// -----------------------------

function BotRow({
  botKey,
  botType,
  statusNode,
  baseConfig,
  displayName,
  dtMeta,
  apiPrefix,
}: {
  botKey: string;
  botType: "swing" | "dt";
  statusNode?: any;
  baseConfig: BotDraft;
  displayName: string;
  dtMeta?: { realized?: number; unrealized?: number; total?: number; fillsCount?: number };
  apiPrefix: string; // "" or "/api/backend" (we’ll use it for saves)
}) {
  const storageKey = `aion.bot_rules.${botKey}`;
  const { draft, dirty, saving, setSaving, setField, reset, setDirty } = useBotDraft(storageKey, baseConfig);

  const [h, setH] = useState<Horizon>("1d");

  const lastUpdate = statusNode?.last_update ?? undefined;

  const cash = statusNode?.cash;
  const invested = statusNode?.invested;
  const allocated = statusNode?.allocated ?? draft.max_alloc ?? 0;
  const holdings = statusNode?.holdings_count;

  const equity = clampNum(statusNode?.equity, clampNum(cash, 0) + clampNum(invested, 0));

  const series = useMemo(() => {
    const maybeCurve = statusNode?.equity_curve ?? statusNode?.pnl_curve;
    return coerceCurve(maybeCurve, equity);
  }, [statusNode, equity]);

  async function save() {
    setSaving(true);
    try {
      // Use centralized API client
      if (botType === "swing") {
        await updateEodBotConfig(botKey, draft, apiPrefix);
      } else {
        await updateIntradayBotConfig(botKey, draft, apiPrefix);
      }
      setDirty(false);
    } catch {
      // keep local rules even if backend doesn't support it
      setDirty(false);
    } finally {
      setSaving(false);
    }
  }

  function toggleEnabled() {
    setField("enabled", !draft.enabled);
  }

  const riskLabel = draft.stop_loss <= 0.03 ? "Tight" : draft.stop_loss <= 0.07 ? "Normal" : "Loose";

  return (
    <Card className="border-white/10 bg-white/5">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-[240px]">
            <CardTitle className="text-xl">{displayName}</CardTitle>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge variant="outline">{botType === "swing" ? "SWING" : "DAY"}</Badge>
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3.5 w-3.5" />
                {lastUpdate ? lastUpdate : "No update yet"}
              </span>
              <span className="inline-flex items-center gap-1">
                <Activity className="h-3.5 w-3.5" />
                Equity {fmtMoney(equity)}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">{horizonButtons(h, setH)}</div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="grid gap-4 lg:grid-cols-[360px_240px_minmax(0,1fr)_56px] items-stretch">
          <div className="rounded-xl border bg-card/30 p-4 min-w-0">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm font-semibold">Performance</div>
              <Badge variant="muted" className="text-xs">
                {h.toUpperCase()}
              </Badge>
            </div>
            <MiniPerfChart data={series} />
            <div className="mt-2 text-[11px] text-muted-foreground">
              Chart is per-horizon. Day bots reset daily; swing bots drift with horizon.
            </div>
          </div>

          <div className="rounded-xl border bg-card/30 p-4">
            <div className="mb-3 flex items-center gap-2">
              <DollarSign className="h-4 w-4 text-muted-foreground" />
              <div className="text-sm font-semibold">Quick Read</div>
            </div>

            {botType === "dt" ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Realized</div>
                    <div className="text-sm font-semibold">{fmtMaybeMoney(dtMeta?.realized)}</div>
                  </div>
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Unrealized</div>
                    <div className="text-sm font-semibold">{fmtMaybeMoney(dtMeta?.unrealized)}</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Total PnL</div>
                    <div className="text-sm font-semibold">{fmtMaybeMoney(dtMeta?.total)}</div>
                  </div>
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Fills</div>
                    <div className="text-sm font-semibold">{dtMeta?.fillsCount ?? "—"}</div>
                  </div>
                </div>

                <div className="rounded-lg border bg-background/40 p-3">
                  <div className="mb-2 flex items-center gap-2">
                    <Shield className="h-4 w-4 text-muted-foreground" />
                    <div className="text-xs font-semibold">Risk quick read</div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <Badge variant="outline">SL {fmtPct(draft.stop_loss)}</Badge>
                    <Badge variant="outline">TP {fmtPct(draft.take_profit)}</Badge>
                    <Badge variant="muted">{riskLabel}</Badge>
                    <Badge variant="muted">Agg {fmtPct(draft.aggression)}</Badge>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Available</div>
                    <div className="text-sm font-semibold">{fmtMaybeMoney(cash)}</div>
                  </div>
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Invested</div>
                    <div className="text-sm font-semibold">{fmtMaybeMoney(invested)}</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Allocated</div>
                    <div className="text-sm font-semibold">{fmtMoney(allocated)}</div>
                  </div>
                  <div className="rounded-lg border bg-background/40 p-3">
                    <div className="text-[11px] text-muted-foreground">Holdings</div>
                    <div className="text-sm font-semibold">{holdings ?? "—"}</div>
                  </div>
                </div>

                <div className="rounded-lg border bg-background/40 p-3">
                  <div className="mb-2 flex items-center gap-2">
                    <Shield className="h-4 w-4 text-muted-foreground" />
                    <div className="text-xs font-semibold">Risk quick read</div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <Badge variant="outline">SL {fmtPct(draft.stop_loss)}</Badge>
                    <Badge variant="outline">TP {fmtPct(draft.take_profit)}</Badge>
                    <Badge variant="muted">{riskLabel}</Badge>
                    <Badge variant="muted">Agg {fmtPct(draft.aggression)}</Badge>
                  </div>
                </div>
              </div>
            )}
          </div>

          <BotRulesPanel
            botKey={botKey}
            botType={botType}
            draft={draft}
            setField={setField}
            dirty={dirty}
            saving={saving}
            onSave={save}
            onReset={() => {
              reset();
              setDirty(true);
            }}
          />

          <div className="flex items-start justify-center pt-2">
            <OnOffDot enabled={!!draft.enabled} onToggle={toggleEnabled} disabled={saving} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// -----------------------------
// Page
// -----------------------------

export default function BotsPage() {
  const [bundle, setBundle] = useState<BotsPageBundle | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [live, setLive] = useState(true);
  const [pollMs, setPollMs] = useState(5000);

  const inFlightRef = useRef(false);
  const apiPrefixRef = useRef<string>("/api/backend"); // we'll auto-detect

  async function refresh() {
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    setErr(null);
    try {
      const data = await getBotsPageBundle();
      setBundle(data);
      
      // Detect which API prefix is working for save operations
      const prefix = await detectApiPrefix();
      apiPrefixRef.current = prefix;
    } catch (e: any) {
      setErr(e?.message ?? "Failed to load");
    } finally {
      setLoading(false);
      inFlightRef.current = false;
    }
  }

  // Initial load
  useEffect(() => {
    refresh();

    const onVis = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-refresh polling when live mode enabled
  useEffect(() => {
    if (!live) return;
    const t = setInterval(refresh, pollMs);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, pollMs]);

  const eodStatus: EodStatusResponse | null = (bundle?.swing?.status && !bundle?.swing?.status?.error)
    ? (bundle?.swing?.status as any)
    : null;

  const eodConfigs: EodConfigResponse | null = (bundle?.swing?.configs && !bundle?.swing?.configs?.error)
    ? (bundle?.swing?.configs as any)
    : null;

  const intradayPnl: IntradayPnlResponse | null = (bundle?.intraday?.pnl_last_day && !bundle?.intraday?.pnl_last_day?.error)
    ? (bundle?.intraday?.pnl_last_day as any)
    : null;

  const fillsArr = useMemo(() => {
    const fillsData = bundle?.intraday?.fills_recent;
    if (fillsData && !fillsData?.error && Array.isArray(fillsData?.fills)) {
      return fillsData.fills as IntradayFill[];
    }
    return [];
  }, [bundle]);

  const signalsArr = useMemo(() => {
    const signalsData = bundle?.intraday?.signals_latest;
    if (signalsData && !signalsData?.error && Array.isArray(signalsData?.signals)) {
      return signalsData.signals as IntradaySignal[];
    }
    return [];
  }, [bundle]);

  const dtUpdatedAt = bundle?.intraday?.fills_recent?.updated_at ?? bundle?.intraday?.signals_latest?.updated_at ?? null;

  const swingBots = useMemo(() => {
    const bots = eodStatus?.bots ?? {};
    const configs = eodConfigs?.configs ?? {};
    const keys = Array.from(new Set([...Object.keys(bots), ...Object.keys(configs)])).sort();
    return keys.map((k) => {
      const statusNode = bots[k];
      const cfg = configs[k];
      const base: BotDraft = {
        max_alloc: cfg?.max_alloc ?? 10_000,
        max_positions: cfg?.max_positions ?? 10,
        stop_loss: cfg?.stop_loss ?? 0.05,
        take_profit: cfg?.take_profit ?? 0.1,
        aggression: cfg?.aggression ?? 0.5,
        enabled: cfg?.enabled ?? true,
      };
      return { botKey: k, statusNode, base, displayName: humanName(k, "swing") };
    });
  }, [eodStatus, eodConfigs]);

  const dtBots = useMemo(() => {
    const bots = intradayPnl?.bots ?? {};
    const keys = Object.keys(bots).sort();
    const finalKeys = keys.length ? keys : ["intraday_engine"];

    return finalKeys.map((k) => {
      const per = bots[k] ?? {};
      const total = clampNum(per?.total ?? (intradayPnl as any)?.total?.total ?? (intradayPnl as any)?.total, 0);

      const base: BotDraft = {
        max_alloc: 5_000,
        max_positions: 6,
        stop_loss: 0.02,
        take_profit: 0.04,
        aggression: 0.65,
        enabled: true,
        max_daily_trades: 10,
        penny_only: false,
        allow_etfs: true,
      };

      const statusNode: any = {
        allocated: base.max_alloc,
        equity: total,
        last_update: dtUpdatedAt ?? undefined,
      };

      const dtMeta = {
        realized: per?.realized ?? (intradayPnl as any)?.total?.realized,
        unrealized: per?.unrealized ?? (intradayPnl as any)?.total?.unrealized,
        total: per?.total ?? (intradayPnl as any)?.total?.total ?? (intradayPnl as any)?.total,
        fillsCount: fillsArr.length,
      };

      return { botKey: k, statusNode, base, displayName: humanName(k, "dt"), dtMeta };
    });
  }, [intradayPnl, dtUpdatedAt, fillsArr.length]);

  const bg = "min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white";

  return (
    <main className={bg}>
      <div className="mx-auto max-w-7xl px-4 py-8">
        <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold">Bots</h1>
            <p className="text-sm text-white/60">Toggle bots on/off, tune rules, and watch per-horizon performance.</p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Link
              href="/bots/config"
              className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm hover:bg-white/10 transition-colors"
            >
              <Settings className="h-4 w-4" />
              <span>Configure Knobs</span>
            </Link>
            
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2">
              <Switch checked={live} onCheckedChange={setLive} />
              <div className="text-xs text-white/70">Live</div>
              <Badge variant="outline" className="text-xs">Polling</Badge>
              <Separator orientation="vertical" className="mx-2 h-5 bg-white/10" />
              <Button size="sm" variant={pollMs === 2000 ? "default" : "outline"} onClick={() => setPollMs(2000)}>
                2s
              </Button>
              <Button size="sm" variant={pollMs === 5000 ? "default" : "outline"} onClick={() => setPollMs(5000)}>
                5s
              </Button>
              <Button size="sm" variant={pollMs === 15000 ? "default" : "outline"} onClick={() => setPollMs(15000)}>
                15s
              </Button>
            </div>

            <Button variant="outline" onClick={refresh} disabled={loading} className="gap-2">
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
              Refresh
            </Button>
          </div>
        </div>

        {err ? (
          <div className="mb-6 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-200">{err}</div>
        ) : null}

        {loading && !bundle ? (
          <div className="mb-6 rounded-xl border border-white/10 bg-white/5 p-8 text-center">
            <RefreshCw className="mx-auto h-8 w-8 animate-spin text-white/40 mb-3" />
            <div className="text-sm text-white/60">Loading bots data...</div>
          </div>
        ) : null}

        {/* SWING BOTS */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Swing bots</h2>
            <div className="text-xs text-white/60">
              {eodStatus?.running ? "Engine: running" : "Engine: idle"} • {swingBots.length} bots
            </div>
          </div>

          <div className="space-y-4">
            {swingBots.map((b) => (
              <BotRow
                key={b.botKey}
                botKey={b.botKey}
                botType="swing"
                statusNode={b.statusNode}
                baseConfig={b.base}
                displayName={b.displayName}
                apiPrefix={apiPrefixRef.current}
              />
            ))}
            {!swingBots.length ? (
              <Card className="border-white/10 bg-white/5">
                <CardContent className="p-6 text-sm text-white/60">
                  No swing bots found yet. If you expect them, confirm the backend mounted <code>/bots/page</code>.
                </CardContent>
              </Card>
            ) : null}
          </div>
        </section>

        <Separator className="my-8 bg-white/10" />

        {/* DAY TRADING BOTS */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Day-trading bots</h2>
            <div className="text-xs text-white/60">{dtBots.length} bots • intraday snapshot</div>
          </div>

          <div className="space-y-4">
            {dtBots.map((b) => (
              <BotRow
                key={b.botKey}
                botKey={b.botKey}
                botType="dt"
                statusNode={b.statusNode}
                baseConfig={b.base}
                displayName={b.displayName}
                dtMeta={b.dtMeta}
                apiPrefix={apiPrefixRef.current}
              />
            ))}
          </div>

          <LiveIntradayTape fills={fillsArr} signals={signalsArr} updatedAt={dtUpdatedAt} />
        </section>

        <div className="mt-10 text-xs text-white/40">
          Note: Save is best-effort. If your backend doesn’t expose config-update endpoints yet, the page still persists
          rules locally.
        </div>
      </div>
    </main>
  );
}

function humanName(botKey: string, t: "swing" | "dt") {
  const k = botKey.replace(/[_-]+/g, " ").trim();
  if (!k) return t === "swing" ? "Swing bot" : "Day bot";
  const suffix = k.match(/\d+/)?.[0];
  if (suffix) return `${t === "swing" ? "Swing" : "Day Trading"} Bot ${suffix}`;
  return `${t === "swing" ? "Swing" : "Day Trading"} ${k}`;
}
