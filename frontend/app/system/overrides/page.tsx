"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button"; // ✅ FIX: Button import added
import clsx from "clsx";

export default function OverridesPage() {
  const [loading, setLoading] = useState<string | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

  const runScript = async (task: string) => {
    try {
      setLoading(task);
      const res = await fetch(`${API_BASE}/api/system/run/${task}`, {
        method: "POST",
      });

      if (!res.ok) {
        const txt = await res.text();
        alert(`Failed: ${res.status} — ${txt}`);
      } else {
        alert(`Task '${task}' started successfully.`);
      }
    } catch (err) {
      alert("Request failed: " + err);
    } finally {
      setLoading(null);
    }
  };

  const btn = (task: string, label: string) => (
    <Button
      onClick={() => runScript(task)}
      disabled={loading === task}
      className={clsx("w-full text-sm", loading === task && "opacity-70")}
    >
      {label}
    </Button>
  );

  return (
    <div className="max-w-4xl mx-auto py-8 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">System Tools — Manual Overrides</CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          <p className="text-xs text-slate-400">
            Manually trigger backend maintenance actions, data rebuilds, and ML workflows.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {btn("nightly", "🛠 Run Nightly Job")}
            {btn("dashboard", "📊 Recompute Dashboard")}
            {btn("insights", "💡 Build Insights")}
            {btn("train", "🧠 Train Models")}
            {btn("metrics", "📈 Refresh Metrics")}
            {btn("fundamentals", "🏦 Fetch Fundamentals")}
            {btn("news", "📰 Update News")}
            {btn("verify", "🔍 Verify Cache")}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
