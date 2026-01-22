"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  FormEvent,
} from "react";
import {
  Play,
  StopCircle,
  Activity,
  BarChart3,
  Calendar,
  RefreshCw,
  Zap,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

// -----------------------------
// Types
// -----------------------------

type JobStatus =
  | "pending"
  | "running"
  | "done"
  | "error"
  | "cancelled"
  | "cancel_requested"
  | string;

interface ReplayJob {
  job_id: string;
  status: JobStatus;
  progress?: number; // 0..1
  start?: string | null;
  end?: string | null;
  error?: string | null;
  result?: any;
  created_at?: string | null;
}

interface ReplayDaySummary {
  date: string;
  num_symbols: number;
  pnl?: {
    total_gross_pnl?: number;
    gross_pnl?: number;
    [k: string]: any;
  } | null;
}

// -----------------------------
// Small helpers
// -----------------------------

function formatPercent(p?: number): string {
  if (p == null || Number.isNaN(p)) return "0%";
  const clamped = Math.max(0, Math.min(1, p));
  return `${(clamped * 100).toFixed(1)}%`;
}

function formatDateLabel(d?: string | null): string {
  if (!d) return "—";
  return d;
}

function formatPnl(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const sign = v < 0 ? "-" : "";
  const abs = Math.abs(v);
  return `${sign}$${abs.toFixed(2)}`;
}

function statusBadgeClass(status: JobStatus): string {
  if (status === "running")
    return "bg-emerald-500/10 text-emerald-400 border border-emerald-500/40";
  if (status === "pending" || status === "cancel_requested")
    return "bg-amber-500/10 text-amber-300 border border-amber-500/40";
  if (status === "error")
    return "bg-red-500/10 text-red-400 border border-red-500/40";
  if (status === "cancelled")
    return "bg-slate-700/40 text-slate-300 border border-slate-500/40";
  if (status === "done")
    return "bg-sky-500/10 text-sky-300 border border-sky-500/40";
  return "bg-slate-700/40 text-slate-200 border border-slate-500/40";
}

function isJobActive(status: JobStatus): boolean {
  return status === "pending" || status === "running" || status === "cancel_requested";
}

// -----------------------------
// Page
// -----------------------------

export default function ReplayPage() {
  // Form state
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [mode, setMode] = useState<string>("full");
  const [computeMode, setComputeMode] = useState<string>("auto");

  // Jobs + polling
  const [jobs, setJobs] = useState<ReplayJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [polling, setPolling] = useState<boolean>(false);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Timeline
  const [days, setDays] = useState<ReplayDaySummary[]>([]);
  const [daysLoading, setDaysLoading] = useState<boolean>(false);
  const [daysError, setDaysError] = useState<string | null>(null);
  const [expandedDay, setExpandedDay] = useState<string | null>(null);

  const activeJob: ReplayJob | undefined = useMemo(
    () => jobs.find((j) => j.job_id === activeJobId) ?? jobs.find((j) => isJobActive(j.status)),
    [jobs, activeJobId],
  );

  const overallProgress = activeJob?.progress ?? 0;

  // -----------------------------
  // API calls
  // -----------------------------

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/replay/jobs`, { cache: "no-store" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = (await res.json()) as { jobs: ReplayJob[] } | ReplayJob[];
      const list = Array.isArray(data) ? data : data.jobs ?? [];
      setJobs(list);

      // If there is any active job, keep polling
      const anyActive = list.some((j) => isJobActive(j.status));
      setPolling(anyActive);

      // If current active job is finished, clear active id
      if (activeJobId) {
        const current = list.find((j) => j.job_id === activeJobId);
        if (current && !isJobActive(current.status)) {
          setActiveJobId(null);
        }
      }
    } catch (err) {
      console.error("Failed to fetch replay jobs", err);
    }
  }, [activeJobId]);

  const fetchDays = useCallback(async () => {
    try {
      setDaysError(null);
      setDaysLoading(true);
      const res = await fetch(`${API_BASE}/api/replay/days`, { cache: "no-store" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = (await res.json()) as { days: ReplayDaySummary[] };
      setDays(data.days ?? []);
    } catch (err) {
      console.error("Failed to fetch replay days", err);
      setDaysError("Failed to load replay timeline.");
    } finally {
      setDaysLoading(false);
    }
  }, []);

  // Polling loop
  useEffect(() => {
    // initial load
    fetchJobs();
    fetchDays();
  }, [fetchJobs, fetchDays]);

  useEffect(() => {
    if (!polling) return;
    const id = setInterval(() => {
      fetchJobs();
      // Optionally refresh days as jobs progress
      fetchDays();
    }, 3000);
    return () => clearInterval(id);
  }, [polling, fetchJobs, fetchDays]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setErrorMsg(null);

    try {
      const payload: any = {
        mode,
        compute_mode: computeMode,
      };
      if (startDate) payload.start_date = startDate;
      if (endDate) payload.end_date = endDate;

      const res = await fetch(`${API_BASE}/api/replay/run-now`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`status ${res.status}: ${txt}`);
      }

      const data = (await res.json()) as { job_id: string };
      const jobId = data.job_id;
      setActiveJobId(jobId);
      setPolling(true);
      // Immediately refresh jobs
      fetchJobs();
    } catch (err: any) {
      console.error("Failed to start replay job", err);
      setErrorMsg(err?.message ?? "Failed to start replay job.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await fetch(`${API_BASE}/api/replay/jobs/${jobId}/cancel`, {
        method: "POST",
      });
      fetchJobs();
    } catch (err) {
      console.error("Failed to cancel job", err);
    }
  };

  // -----------------------------
  // Render
  // -----------------------------

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-50">
            Historical Replay
          </h1>
          <p className="text-sm text-slate-400">
            Re-run intraday sessions through the full AION pipeline and analyze performance.
          </p>
        </div>
        <div className="hidden md:flex items-center gap-2 text-xs text-slate-400">
          <Activity className="h-4 w-4" />
          <span>Intraday backtest engine</span>
        </div>
      </header>

      {/* Top row: run controls + active job */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Run controls */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Play className="h-4 w-4 text-sky-400" />
              <h2 className="text-sm font-semibold text-slate-100">
                Run new replay
              </h2>
            </div>
            <button
              type="button"
              onClick={() => {
                fetchJobs();
                fetchDays();
              }}
              className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-950/60 px-2 py-1 text-[11px] font-medium text-slate-300 hover:border-slate-500"
            >
              <RefreshCw className="h-3 w-3" />
              Refresh
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  Start date
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="h-8 rounded-md border border-slate-700 bg-slate-950/80 px-2 text-xs text-slate-100 outline-none focus:border-sky-500"
                />
                <p className="text-[10px] text-slate-500">
                  Leave empty to use earliest available day.
                </p>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  End date
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="h-8 rounded-md border border-slate-700 bg-slate-950/80 px-2 text-xs text-slate-100 outline-none focus:border-sky-500"
                />
                <p className="text-[10px] text-slate-500">
                  Leave empty to use latest available day.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  Replay mode
                </label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="h-8 rounded-md border border-slate-700 bg-slate-950/80 px-2 text-xs text-slate-100 outline-none focus:border-sky-500"
                >
                  <option value="full">Full (policy + execution)</option>
                  <option value="fast">Fast (reduced detail)</option>
                </select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  Compute
                </label>
                <select
                  value={computeMode}
                  onChange={(e) => setComputeMode(e.target.value)}
                  className="h-8 rounded-md border border-slate-700 bg-slate-950/80 px-2 text-xs text-slate-100 outline-none focus:border-sky-500"
                >
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU-only</option>
                  <option value="gpu">Prefer GPU</option>
                </select>
              </div>
            </div>

            {errorMsg && (
              <div className="rounded-md border border-red-500/50 bg-red-950/40 px-2 py-1 text-[11px] text-red-200">
                {errorMsg}
              </div>
            )}

            <div className="mt-2 flex items-center justify-between gap-2">
              <button
                type="submit"
                disabled={submitting}
                className="inline-flex items-center gap-2 rounded-md bg-sky-500 px-3 py-1.5 text-xs font-semibold text-slate-950 shadow-sm hover:bg-sky-400 disabled:opacity-60"
              >
                <Play className="h-3 w-3" />
                {submitting ? "Starting replay…" : "Run replay"}
              </button>
              <p className="text-[11px] text-slate-500 flex items-center gap-1">
                <Zap className="h-3 w-3 text-amber-300" />
                Runs in the background. You can keep using the app.
              </p>
            </div>
          </form>
        </section>

        {/* Active job + big progress bar */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-emerald-400" />
              <h2 className="text-sm font-semibold text-slate-100">
                Current replay run
              </h2>
            </div>
          </div>

          {activeJob ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs">
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] text-slate-400">
                      Job {activeJob.job_id.slice(0, 8)}
                    </span>
                    <span
                      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold capitalize ${statusBadgeClass(
                        activeJob.status,
                      )}`}
                    >
                      {activeJob.status}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-400">
                    <span>
                      Range:{" "}
                      <span className="text-slate-200">
                        {formatDateLabel(activeJob.start)} →{" "}
                        {formatDateLabel(activeJob.end)}
                      </span>
                    </span>
                    {typeof activeJob.result?.days === "number" && (
                      <span>
                        Days replayed:{" "}
                        <span className="text-slate-200">
                          {activeJob.result.days}
                        </span>
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="text-right">
                    <div className="text-[11px] text-slate-400">Progress</div>
                    <div className="text-xs font-semibold text-slate-100">
                      {formatPercent(overallProgress)}
                    </div>
                  </div>
                  {isJobActive(activeJob.status) && (
                    <button
                      type="button"
                      onClick={() => handleCancel(activeJob.job_id)}
                      className="inline-flex items-center gap-1 rounded-md border border-red-500/40 bg-red-900/30 px-2 py-1 text-[11px] font-medium text-red-100 hover:border-red-400"
                    >
                      <StopCircle className="h-3 w-3" />
                      Cancel
                    </button>
                  )}
                </div>
              </div>

              {/* big progress bar */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                  style={{ width: `${Math.max(2, overallProgress * 100)}%` }}
                />
              </div>

              {activeJob.error && (
                <div className="rounded-md border border-red-500/50 bg-red-950/40 px-2 py-1 text-[11px] text-red-200">
                  {activeJob.error}
                </div>
              )}
            </div>
          ) : (
            <div className="flex h-24 flex-col items-center justify-center gap-2 text-xs text-slate-400">
              <p>No active replay job.</p>
              <p className="text-[11px] text-slate-500">
                Start a new run on the left to see live progress.
              </p>
            </div>
          )}
        </section>
      </div>

      {/* Jobs table + Timeline */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Jobs table */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/70 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-sky-400" />
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Replay jobs
              </h2>
            </div>
            <span className="text-[11px] text-slate-500">
              {jobs.length} job{jobs.length === 1 ? "" : "s"}
            </span>
          </div>

          {jobs.length === 0 ? (
            <div className="p-4 text-xs text-slate-400">
              No replay jobs have been launched yet.
            </div>
          ) : (
            <div className="max-h-72 overflow-auto">
              <table className="min-w-full text-xs">
                <thead className="bg-slate-950/60 text-slate-400">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Job</th>
                    <th className="px-3 py-2 text-left font-medium">Range</th>
                    <th className="px-3 py-2 text-left font-medium">Status</th>
                    <th className="px-3 py-2 text-left font-medium">Progress</th>
                    <th className="px-3 py-2 text-right font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {jobs
                    .slice()
                    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))
                    .map((job) => {
                      const p = job.progress ?? 0;
                      const active = isJobActive(job.status);
                      return (
                        <tr key={job.job_id}>
                          <td className="px-3 py-2 font-mono text-[11px] text-slate-300">
                            {job.job_id}
                          </td>
                          <td className="px-3 py-2 text-[11px] text-slate-300">
                            {formatDateLabel(job.start)} → {formatDateLabel(job.end)}
                          </td>
                          <td className="px-3 py-2">
                            <span
                              className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold capitalize ${statusBadgeClass(
                                job.status,
                              )}`}
                            >
                              {job.status}
                            </span>
                          </td>
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800">
                                <div
                                  className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                                  style={{ width: `${Math.max(2, p * 100)}%` }}
                                />
                              </div>
                              <span className="text-[10px] text-slate-400">
                                {formatPercent(p)}
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-2 text-right">
                            {active ? (
                              <button
                                type="button"
                                onClick={() => handleCancel(job.job_id)}
                                className="inline-flex items-center gap-1 rounded-md border border-red-500/40 bg-red-900/30 px-2 py-1 text-[10px] font-medium text-red-100 hover:border-red-400"
                              >
                                <StopCircle className="h-3 w-3" />
                                Cancel
                              </button>
                            ) : (
                              <span className="text-[10px] text-slate-500">—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Timeline / per-day summary */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/70 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-emerald-400" />
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Replay timeline
              </h2>
            </div>
            <span className="text-[11px] text-slate-500">
              {days.length} day{days.length === 1 ? "" : "s"}
            </span>
          </div>

          {daysError && (
            <div className="m-3 rounded-md border border-red-500/50 bg-red-950/40 px-2 py-1 text-[11px] text-red-200">
              {daysError}
            </div>
          )}

          {daysLoading && days.length === 0 ? (
            <div className="p-4 text-xs text-slate-400">
              Loading replay days…
            </div>
          ) : days.length === 0 ? (
            <div className="p-4 text-xs text-slate-400">
              No replay days recorded yet.
            </div>
          ) : (
            <div className="max-h-72 overflow-auto px-3 py-3 space-y-2">
              {days
                .slice()
                .sort((a, b) => b.date.localeCompare(a.date))
                .map((d) => {
                  const pnlRaw =
                    (d.pnl?.total_gross_pnl ??
                      d.pnl?.gross_pnl ??
                      0) as number;
                  const pnlClass =
                    pnlRaw > 0
                      ? "text-emerald-400"
                      : pnlRaw < 0
                      ? "text-red-400"
                      : "text-slate-300";

                  const pnlMagnitude = Math.min(
                    1,
                    Math.abs(pnlRaw) / (Math.abs(pnlRaw) + 100 || 100),
                  );

                  return (
                    <div
                      key={d.date}
                      className="rounded-lg border border-slate-800/80 bg-slate-950/60 p-2"
                    >
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedDay((prev) =>
                            prev === d.date ? null : d.date,
                          )
                        }
                        className="flex w-full items-center justify-between gap-3 text-left"
                      >
                        <div className="flex items-center gap-2">
                          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-900/80 text-[11px] font-medium text-slate-100">
                            {new Date(d.date).getDate() || d.date}
                          </div>
                          <div>
                            <div className="text-xs font-semibold text-slate-100">
                              {d.date}
                            </div>
                            <div className="text-[11px] text-slate-400">
                              {d.num_symbols} symbols replayed
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-col items-end gap-1">
                          <div className={`text-xs font-semibold ${pnlClass}`}>
                            {formatPnl(pnlRaw)}
                          </div>
                          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800">
                            <div
                              className={`h-full rounded-full ${
                                pnlRaw >= 0 ? "bg-emerald-500" : "bg-red-500"
                              }`}
                              style={{
                                width: `${Math.max(5, pnlMagnitude * 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      </button>

                      {/* Simple popout / expanded stats */}
                      {expandedDay === d.date && (
                        <div className="mt-2 rounded-md bg-slate-900/80 p-2 text-[11px] text-slate-300">
                          <div className="flex items-center justify-between">
                            <span className="text-slate-400">
                              Gross P&L (approx)
                            </span>
                            <span className={pnlClass}>
                              {formatPnl(pnlRaw)}
                            </span>
                          </div>
                          {d.pnl && (
                            <div className="mt-1 grid grid-cols-2 gap-1 text-slate-400">
                              {Object.entries(d.pnl)
                                .slice(0, 6)
                                .map(([k, v]) => (
                                  <div
                                    key={k}
                                    className="flex items-center justify-between gap-2"
                                  >
                                    <span className="truncate">{k}</span>
                                    <span className="text-slate-200">
                                      {typeof v === "number"
                                        ? v.toFixed(4)
                                        : String(v)}
                                    </span>
                                  </div>
                                ))}
                            </div>
                          )}
                          <p className="mt-2 text-[10px] text-slate-500">
                            For full per-day trade breakdowns, we can later add
                            a dedicated detail view powered by the replay
                            results JSON.
                          </p>
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
