"use client";
import { useEffect, useState } from "react";
import { Activity } from "lucide-react";

export default function AccuracyCard() {
  const [acc, setAcc] = useState<number | null>(null);
  const [wow, setWow] = useState<number | null>(null);
  const [summary, setSummary] = useState<string>("");

  useEffect(() => {
    async function fetchAccuracy() {
      try {
        // Use Next.js proxy route (server decides real backend via env)
        const res = await fetch("/api/backend/dashboard/metrics", {
          cache: "no-store",
        });

        const data = await res.json();
        if (data && typeof data.accuracy_30d === "number") {
          setAcc(data.accuracy_30d * 100);  // Use combined accuracy
          
          // Show breakdown in summary if execution accuracy is available
          if (typeof data.model_accuracy === "number" && typeof data.execution_accuracy === "number") {
            const modelAcc = (data.model_accuracy * 100).toFixed(1);
            const execAcc = (data.execution_accuracy * 100).toFixed(1);
            setSummary(`Model: ${modelAcc}% • Trades: ${execAcc}%`);
          } else if (typeof data.model_accuracy === "number") {
            // If only model accuracy, show that
            const modelAcc = (data.model_accuracy * 100).toFixed(1);
            setSummary(`Model: ${modelAcc}%`);
          } else {
            setSummary(data.summary || "");
          }
          
          // You could later calculate WoW here if you start tracking weekly snapshots
          setWow(null);
        }
      } catch (err) {
        console.error("Failed to fetch dashboard metrics:", err);
      }
    }
    fetchAccuracy();
  }, []);

  const pct = acc != null ? acc.toFixed(1) : null;

  return (
    <div className="card card-hover p-6 flex-1 min-h-[180px] flex flex-col justify-between">
      <div className="flex items-center gap-2 text-slate-300">
        <Activity size={18} /> <span className="text-sm">Accuracy</span>
      </div>
      <div className="text-center">
        <div className="text-5xl font-extrabold text-brand-400">
          {pct != null ? `${pct}%` : "—"}
        </div>
        {wow != null ? (
          <div className="mt-2 text-sm text-slate-400">
            {`${wow >= 0 ? "+" : ""}${wow.toFixed(1)}% WoW`}
          </div>
        ) : (
          <div className="mt-2 text-sm text-slate-400">{summary}</div>
        )}
      </div>
    </div>
  );
}
