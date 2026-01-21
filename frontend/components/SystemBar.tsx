"use client";
import { useEffect, useState } from "react";

function prettyStatus(s?: string) {
  const v = String(s || "").toLowerCase();
  if (v === "ok") return "✅ Stable";
  if (v === "warning") return "🟡 Warning";
  if (v === "critical" || v === "error") return "🔴 Critical";
  if (v === "running") return "🟠 Running";
  return "⚪ —";
}

export default function SystemBar() {
  const [status, setStatus] = useState({
    drift: "⚪ Checking...",
    retraining: "⚪ Checking...",
    lastUpdate: "—",
    retrainCycles: "—",
    newsCount: "—",
    tickersTracked: "—",
    version: "SAP v1.4.2",
    debug: "", // 👈 show errors / URL
  });

  /**
   * Fetch system status from backend via Next.js proxy route.
   * Uses /api/backend/system/status which proxies to backend /api/system/status.
   * This avoids hardcoded URLs and works with remote backends.
   */
  async function fetchStatus() {
    const url = "/api/backend/system/status";

    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${res.statusText}${text ? ` — ${text.slice(0, 120)}` : ""}`);
      }

      const data = await res.json();

      const sup = data?.supervisor;
      const driftStatus = sup?.components?.drift?.status;
      const modelsStatus = sup?.components?.models?.status;
      const newsStatus = sup?.components?.intel?.news_intel?.status;
      const tickers = data?.coverage?.symbols;

      setStatus({
        drift: prettyStatus(driftStatus),
        retraining: prettyStatus(modelsStatus),
        lastUpdate: sup?.generated_at || data?.server_time || "—",
        retrainCycles: "—",
        newsCount: prettyStatus(newsStatus),
        tickersTracked: typeof tickers === "number" ? String(tickers) : "—",
        version: "AION v1.1.2",
        debug: "",
      });
    } catch (e: any) {
      setStatus((prev) => ({
        ...prev,
        drift: "⚠️ Offline",
        retraining: "—",
        debug: `FAIL ${url} — ${e?.message || String(e)}`,
      }));
      // Also log so you can see it in console immediately
      console.error("[SystemBar]", url, e);
    }
  }

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 15000);
    return () => clearInterval(id);
  }, []);

  const items = [
    `Drift: ${status.drift}`,
    `Retraining: ${status.retraining}`,
    `Last Update: ${status.lastUpdate}`,
    `Retrain Cycles: ${status.retrainCycles}`,
    `News Articles: ${status.newsCount}`,
    `Tickers Tracked: ${status.tickersTracked}`,
    `${status.version}`,
  ];

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-slate-950/80 backdrop-blur-xs border-t border-slate-800">
      <div className="mx-auto max-w-7xl px-4 py-2 text-xs text-slate-400 flex flex-wrap items-center justify-center gap-4">
        {items.map((t, i) => (
          <span key={i} className="opacity-80">
            {t}
          </span>
        ))}
        {/* Debug line so you can see the actual URL + error while nightly runs */}
        <span className="opacity-50">{status.debug}</span>
      </div>
    </div>
  );
}
