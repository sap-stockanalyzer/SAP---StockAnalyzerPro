"use client";
import { useEffect, useState } from "react";
import { TrendingUp } from "lucide-react";

type Row = { ticker: string; gain_pct?: number };

export default function TopPredictions({
  title,
  horizon,
}: {
  title: string;
  horizon: "1w" | "4w";
}) {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    async function fetchTop() {
      try {
        // Map 4w -> 1m to match backend horizon
        const backendHorizon = horizon === "4w" ? "1m" : horizon;

        // Use portfolio holdings endpoint instead of predictions
        const res = await fetch(`/api/backend/portfolio/holdings/top/${backendHorizon}`, {
          cache: "no-store",
        });

        const data = await res.json();

        // Handle new portfolio holdings response format
        if (data && Array.isArray(data.holdings)) {
          const topRows = data.holdings.slice(0, 3).map((h: any) => ({
            ticker: h.ticker,
            gain_pct: h.pnl_percent,  // Use pnl_percent from holdings
          }));
          setRows(topRows);
        }
      } catch (err) {
        console.error("Failed to fetch top performers:", err);
      }
    }
    fetchTop();
  }, [horizon]);

  return (
    <div className="card card-hover p-6 flex-1 min-h-[180px]">
      <div className="flex items-center gap-2 text-slate-300 mb-3">
        <TrendingUp size={18} /> <span className="text-sm">{title}</span>
      </div>
      <ol className="space-y-2">
        {rows.length > 0 ? (
          rows.map((r, i) => (
            <li
              key={r.ticker}
              className="flex items-center justify-between text-sm"
            >
              <span className="text-slate-200">{`${i + 1}️⃣ ${r.ticker}`}</span>
              <span
                className={`font-semibold ${
                  (r.gain_pct ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                }`}
              >
                {r.gain_pct != null
                  ? `${r.gain_pct >= 0 ? "+" : ""}${r.gain_pct.toFixed(2)}%`
                  : "—"}
              </span>
            </li>
          ))
        ) : (
          <p className="text-slate-500 text-sm">No data available</p>
        )}
      </ol>
    </div>
  );
}
