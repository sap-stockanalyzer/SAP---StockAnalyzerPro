"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import clsx from "clsx";

type TaskStatus = "idle" | "queued" | "running" | "done" | "error";

type TaskUI = {
  status: TaskStatus;
  error?: string | null;
  // visual progress only (since your API doesn’t stream real progress)
  progress: number; // 0..100
  startedAt?: number | null;
  finishedAt?: number | null;
};

const TASKS: Array<{ task: string; label: string }> = [
  { task: "nightly", label: "🛠 Run Nightly Job" },
  { task: "dashboard", label: "📊 Recompute Dashboard" },
  { task: "insights", label: "💡 Build Insights" },
  { task: "train", label: "🧠 Train Models" },
  { task: "metrics", label: "📈 Refresh Metrics" },
  { task: "fundamentals", label: "🏦 Fetch Fundamentals" },
  { task: "news", label: "📰 Update News" },
  { task: "verify", label: "🔍 Verify Cache" },
];

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function ms() {
  return Date.now();
}

function ProgressButton(props: {
  label: string;
  status: TaskStatus;
  progress: number;
  onClick: () => void;
  disabled?: boolean;
}) {
  const { label, status, progress, onClick, disabled } = props;

  const isBusy = status === "queued" || status === "running";
  const isDone = status === "done";
  const isErr = status === "error";

  return (
    <Button
      onClick={onClick}
      disabled={disabled || isBusy}
      className={clsx(
        "w-full text-sm relative overflow-hidden justify-start",
        isBusy && "opacity-90",
        isDone && "opacity-95",
        isErr && "opacity-95"
      )}
      variant={isErr ? "destructive" : "default"}
    >
      {/* background progress bar */}
      {isBusy && (
        <span
          aria-hidden
          className="absolute inset-y-0 left-0 bg-white/15"
          style={{ width: `${clamp(progress, 0, 100)}%` }}
        />
      )}

      <span className="relative z-10 flex items-center gap-2 w-full">
        <span className="truncate">
          {status === "queued"
            ? `Queued… ${label}`
            : status === "running"
            ? `Running… ${label}`
            : status === "done"
            ? `Done ✓ ${label}`
            : status === "error"
            ? `Failed ✕ ${label}`
            : label}
        </span>

        {/* right-side percentage when busy */}
        {isBusy && (
          <span className="ml-auto tabular-nums text-xs opacity-90">
            {clamp(Math.round(progress), 0, 100)}%
          </span>
        )}
      </span>
    </Button>
  );
}

export default function OverridesPage() {
  const API_BASE =
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    "http://209.126.82.160:8000";

  // Per-task UI state
  const [tasks, setTasks] = useState<Record<string, TaskUI>>(() => {
    const init: Record<string, TaskUI> = {};
    for (const t of TASKS) {
      init[t.task] = { status: "idle", progress: 0, error: null };
    }
    return init;
  });

  // FIFO queue (tasks waiting to run)
  const [queue, setQueue] = useState<string[]>([]);
  const runningRef = useRef<string | null>(null);
  const tickerRef = useRef<number | null>(null);

  // --- visual progress ticker for the currently running task (fake progress)
  const startProgressTicker = (task: string) => {
    stopProgressTicker();

    tickerRef.current = window.setInterval(() => {
      setTasks((prev) => {
        const cur = prev[task];
        if (!cur || cur.status !== "running") return prev;

        // fake progress behavior:
        // - ramps up quickly early
        // - slows near 92% until the request finishes
        const p = cur.progress ?? 0;
        const bump =
          p < 50 ? 6 :
          p < 75 ? 3 :
          p < 90 ? 1.5 :
          p < 92 ? 0.4 : 0.0;

        const next = clamp(p + bump, 0, 92);

        return {
          ...prev,
          [task]: { ...cur, progress: next },
        };
      });
    }, 450);
  };

  const stopProgressTicker = () => {
    if (tickerRef.current) {
      window.clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
  };

  // --- core runner: runs ONE task at a time
  const runTask = async (task: string) => {
    runningRef.current = task;

    setTasks((prev) => ({
      ...prev,
      [task]: {
        status: "running",
        progress: Math.max(prev[task]?.progress ?? 0, 6),
        error: null,
        startedAt: ms(),
        finishedAt: null,
      },
    }));

    startProgressTicker(task);

    try {
      const res = await fetch(`${API_BASE}/api/system/run/${task}`, {
        method: "POST",
      });

      const txt = await res.text();

      if (!res.ok) {
        throw new Error(`HTTP ${res.status} — ${txt}`);
      }

      // If your endpoint returns JSON, we don't strictly need it for UI,
      // but parsing here can help catch invalid responses.
      // (Ignore parse errors; the task still started.)
      try {
        JSON.parse(txt);
      } catch {
        // no-op
      }

      setTasks((prev) => ({
        ...prev,
        [task]: {
          ...prev[task],
          status: "done",
          progress: 100,
          finishedAt: ms(),
          error: null,
        },
      }));
    } catch (err: any) {
      setTasks((prev) => ({
        ...prev,
        [task]: {
          ...prev[task],
          status: "error",
          progress: 100,
          finishedAt: ms(),
          error: String(err?.message ?? err),
        },
      }));
    } finally {
      stopProgressTicker();
      runningRef.current = null;

      // move on to next queued task (if any)
      setQueue((q) => q.slice(1));
    }
  };

  // --- queue consumer: whenever queue changes and nothing is running, run next
  useEffect(() => {
    if (runningRef.current) return;
    if (!queue.length) return;

    const next = queue[0];
    runTask(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queue]);

  // cleanup
  useEffect(() => {
    return () => stopProgressTicker();
  }, []);

  const isAnythingRunning = runningRef.current != null;

  const enqueue = (task: string) => {
    setTasks((prev) => {
      const cur = prev[task];
      // If already queued/running, ignore clicks (no duplicates)
      if (cur?.status === "queued" || cur?.status === "running") return prev;

      return {
        ...prev,
        [task]: {
          ...cur,
          status: "queued",
          progress: 0,
          error: null,
          startedAt: null,
          finishedAt: null,
        },
      };
    });

    setQueue((q) => {
      // avoid duplicate enqueue (double click, etc.)
      if (q.includes(task)) return q;
      return [...q, task];
    });
  };

  const clearAll = () => {
    // Only clear queue; do not interrupt a running task
    setQueue((q) => q.slice(0, runningRef.current ? 1 : 0));
    setTasks((prev) => {
      const next: Record<string, TaskUI> = { ...prev };
      for (const k of Object.keys(next)) {
        if (next[k].status === "queued") {
          next[k] = { status: "idle", progress: 0, error: null };
        }
      }
      return next;
    });
  };

  const queuedCount = queue.length - (runningRef.current ? 1 : 0);
  const runningTask = runningRef.current;

  return (
    <div className="max-w-4xl mx-auto py-8 space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-3 flex-wrap">
          <div>
            <CardTitle className="text-lg font-semibold">
              System Tools — Manual Overrides
            </CardTitle>
            <p className="text-xs text-slate-400 mt-1">
              Runs are serialized: one at a time. Extra clicks get queued.
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              API_BASE: <span className="font-mono">{API_BASE}</span>
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={clearAll}
              disabled={queuedCount <= 0}
              className="text-sm"
            >
              Clear Queue
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge
              variant="secondary"
              className={clsx(isAnythingRunning && "opacity-100")}
            >
              Running: {runningTask ?? "none"}
            </Badge>
            <Badge variant="secondary">Queued: {Math.max(queuedCount, 0)}</Badge>
          </div>

          <Separator />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {TASKS.map(({ task, label }) => {
              const st = tasks[task] ?? {
                status: "idle",
                progress: 0,
                error: null,
              };

              return (
                <div key={task} className="space-y-2">
                  <ProgressButton
                    label={label}
                    status={st.status}
                    progress={st.progress}
                    onClick={() => enqueue(task)}
                    // block multi-click while queued or running (handled internally too)
                    disabled={st.status === "queued" || st.status === "running"}
                  />

                  {st.status === "error" && st.error ? (
                    <div className="text-[11px] text-red-500 break-words">
                      {st.error}
                    </div>
                  ) : null}

                  {st.status === "done" && st.startedAt && st.finishedAt ? (
                    <div className="text-[11px] text-muted-foreground">
                      Finished in{" "}
                      <span className="font-medium">
                        {((st.finishedAt - st.startedAt) / 1000).toFixed(2)}s
                      </span>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
