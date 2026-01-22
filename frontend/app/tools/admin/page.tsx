"use client";

import { useEffect, useRef, useState } from "react";

/* -------------------------------------------------- */
/* Backend base (always via Next.js proxy)            */
/* -------------------------------------------------- */

function getBackendBaseUrl() {
  return "/api/backend";
}

// Your dt_backend proxy base. This MUST match your Next.js route folder.
// If your route is: app/api/dt/[...path]/route.ts  -> keep "/api/dt"
// If your route is: app/api/dt-backend/[...path]/route.ts -> change to "/api/dt-backend"
function getDtBackendBaseUrl() {
  return "/api/dt";
}

/* -------------------------------------------------- */
/* Types                                              */
/* -------------------------------------------------- */

type ReplayStatus = {
  status?: string;
  percent_complete?: number;
  current_day?: string | null;
  eta_secs?: number | null;
};

type DtReplayStatus = {
  status?: string;
  progress?: number; // 0..1
  current_day?: string | null;
  eta_secs?: number | null;
};

function clampPct(x: number) {
  return Math.min(100, Math.max(0, x));
}

/* -------------------------------------------------- */
/* Component                                          */
/* -------------------------------------------------- */

export default function AdminPage() {
  const [token, setToken] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");

  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Visual feedback for admin actions (otherwise clicks can feel like "nothing happened")
  const [lastActionResult, setLastActionResult] = useState<any>(null);

  const [replay, setReplay] = useState<ReplayStatus | null>(null);
  const [dtReplay, setDtReplay] = useState<DtReplayStatus | null>(null);

  // Use browser-safe timer type (avoids NodeJS typings in client bundles)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dtPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* ---------------- Live Log ---------------- */

  const [liveLog, setLiveLog] = useState<string>("");
  const logRef = useRef<HTMLPreElement | null>(null);

  /* ---------------- Smooth Progress ---------------- */

  const [swingDisplayPct, setSwingDisplayPct] = useState(0);
  const [dtDisplayPct, setDtDisplayPct] = useState(0);

  /* ================================================== */
  /* Auth                                               */
  /* ================================================== */

  async function login() {
    setError(null);
    setStatus("Logging in...");

    const res = await fetch(`${getBackendBaseUrl()}/admin/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
      cache: "no-store",
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.token) {
      setError(data?.detail || "Invalid password");
      setStatus(null);
      return;
    }

    setToken(data.token);
    setStatus("Login success ✅");
  }

  function logout() {
    setToken(null);
    setReplay(null);
    setDtReplay(null);
    setPassword("");
    setNewPassword("");
    setStatus(null);
    setError(null);
    stopPolling();
    stopDtPolling();
    setLiveLog("");
  }

  /* ================================================== */
  /* Swing Replay                                       */
  /* ================================================== */

  async function fetchReplayStatus() {
    if (!token) return;

    const res = await fetch(`${getBackendBaseUrl()}/admin/replay/status`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });

    const data = await res.json().catch(() => null);
    if (data) setReplay(data);

    const st = String(data?.status || "");
    if (["complete", "stopped", "never_ran"].includes(st)) {
      stopPolling();
    }
  }

  function startPolling() {
    stopPolling();
    fetchReplayStatus();
    pollRef.current = setInterval(fetchReplayStatus, 3000);
  }

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  async function startReplay() {
    if (!token) return;

    setStatus("Starting swing replay...");
    const res = await fetch(`${getBackendBaseUrl()}/admin/replay/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ weeks: 4 }),
      cache: "no-store",
    });

    if (res.ok) {
      setReplay((r) => ({ ...(r || {}), status: "running" }));
      startPolling();
    } else {
      const msg = await res.text().catch(() => "");
      setStatus(`Failed to start swing replay ❌ ${msg}`);
    }
  }

  /* ================================================== */
  /* DT Replay                                          */
  /* ================================================== */

  async function fetchDtReplayStatus() {
    // DT endpoints do not require swing admin token in your current backend;
    // they are protected (or not) by dt_backend itself.
    const res = await fetch(`${getDtBackendBaseUrl()}/api/replay/status`, {
      cache: "no-store",
    });

    const data = await res.json().catch(() => null);
    if (data) setDtReplay(data);

    const st = String(data?.status || "");
    if (["idle", "done", "complete", "stopped"].includes(st)) {
      stopDtPolling();
    }
  }

  function startDtPolling() {
    stopDtPolling();
    fetchDtReplayStatus();
    dtPollRef.current = setInterval(fetchDtReplayStatus, 3000);
  }

  function stopDtPolling() {
    if (dtPollRef.current) {
      clearInterval(dtPollRef.current);
      dtPollRef.current = null;
    }
  }

  async function startDtReplay() {
    setStatus("Starting DT replay...");
    const res = await fetch(`${getDtBackendBaseUrl()}/api/replay/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weeks: 4 }),
      cache: "no-store",
    });

    if (res.ok) {
      setDtReplay((r) => ({ ...(r || {}), status: "running", progress: 0 }));
      startDtPolling();
    } else {
      const msg = await res.text().catch(() => "");
      setStatus(`Failed to start DT replay ❌ ${msg}`);
    }
  }

  /* ================================================== */
  /* Live Log Feed                                      */
  /* ================================================== */
  // IMPORTANT: Your backend provides logs at /admin/tools/logs (returns JSON {lines: [...]})
  // NOT /admin/live-log (text). So we match your router.

  async function fetchLiveLog() {
    if (!token) return;

    const res = await fetch(`${getBackendBaseUrl()}/admin/tools/logs`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });

    if (!res.ok) return;

    const data = await res.json().catch(() => null);
    const lines: unknown = data?.lines;

    if (Array.isArray(lines)) {
      setLiveLog(lines.join(""));
    } else if (typeof lines === "string") {
      setLiveLog(lines);
    }
  }

  useEffect(() => {
    if (!token) return;
    fetchLiveLog();
    const t = setInterval(fetchLiveLog, 2000);
    return () => clearInterval(t);
  }, [token]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [liveLog]);

  /* ================================================== */
  /* Admin Actions                                      */
  /* ================================================== */

  async function clearLocks() {
    if (!token) return;
    if (!confirm("Remove ALL locks and reset swing replay state?")) return;

    setError(null);
    setLastActionResult(null);

    const res = await fetch(`${getBackendBaseUrl()}/admin/tools/clear-locks`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setError(`Failed to clear locks: ${data?.detail ?? "unknown_error"}`);
      return;
    }

    setLastActionResult(data);
    setStatus("Locks cleared + replay reset ✅");

    // refresh UI state
    fetchReplayStatus();
    fetchDtReplayStatus();
  }

  async function pullFromGit() {
    if (!token) return;
    if (!confirm("Pull latest code from GitHub?")) return;

    setError(null);
    setLastActionResult(null);
    setStatus("Pulling from GitHub...");
    const res = await fetch(`${getBackendBaseUrl()}/admin/tools/git-pull`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setError(`Git pull failed ❌ ${data?.detail ?? "unknown_error"}`);
      return;
    }

    setLastActionResult(data);

    setStatus("Git pull complete ✅");
  }

  async function restartServices() {
    if (!token) return;
    if (!confirm("Restart services now? This will briefly interrupt the backend.")) return;

    setError(null);
    setLastActionResult(null);
    setStatus("Restarting services...");
    const res = await fetch(`${getBackendBaseUrl()}/admin/system/restart`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setError(`Restart failed ❌ ${data?.detail ?? "unknown_error"}`);
      return;
    }

    setLastActionResult(data);

    setStatus("Restart triggered ✅ (backend will drop for a moment)");

    // Optional: after a short delay, try to refetch logs/status
    setTimeout(() => {
      fetchLiveLog();
      fetchReplayStatus();
      fetchDtReplayStatus();
    }, 2500);
  }

  async function refreshUniverses() {
    if (!token) return;
    if (!confirm("Refresh universes from StockAnalysis (NASDAQ + NYSE) now?")) return;

    setError(null);
    setLastActionResult(null);
    setStatus("Refreshing universes from StockAnalysis...");

    const res = await fetch(`${getBackendBaseUrl()}/admin/tools/refresh-universes`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      // Safe even if the backend ignores it; use when backend supports SA sourcing.
      body: JSON.stringify({ source: "stockanalysis", exchanges: ["nasdaq", "nyse"] }),
      cache: "no-store",
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data?.detail || (await res.text().catch(() => "")) || "unknown_error";
      setError(`Universe refresh failed ❌ ${detail}`);
      return;
    }

    setLastActionResult(data);

    const baseN = data?.base_symbols ?? data?.base ?? data?.base_count;
    const dtN = data?.dt_symbols ?? data?.dt ?? data?.dt_count;
    if (typeof baseN === "number" || typeof dtN === "number") {
      setStatus(`Universe refresh complete ✅ (master=${baseN ?? "?"}, dt=${dtN ?? "?"})`);
    } else {
      setStatus("Universe refresh complete ✅");
    }
  }

  /* ================================================== */
  /* Progress easing                                    */
  /* ================================================== */

  const swingTargetPct = clampPct(replay?.percent_complete ?? 0);
  const dtTargetPct = clampPct((dtReplay?.progress ?? 0) * 100);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setSwingDisplayPct((p) => {
        const d = swingTargetPct - p;
        return Math.abs(d) < 0.15 ? swingTargetPct : clampPct(p + d * 0.12);
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [swingTargetPct]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setDtDisplayPct((p) => {
        const d = dtTargetPct - p;
        return Math.abs(d) < 0.15 ? dtTargetPct : clampPct(p + d * 0.12);
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [dtTargetPct]);

  /* ================================================== */
  /* Login screen                                       */
  /* ================================================== */

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-950">
        <div className="w-full max-w-sm bg-gray-900 p-6 rounded-xl border border-gray-800">
          <h2 className="text-xl mb-4 text-center">Admin Login</h2>

          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full mb-3 p-2 bg-gray-800 border border-gray-700 rounded"
          />

          {status && <div className="text-sm mb-2">{status}</div>}
          {error && <div className="text-sm text-red-400 mb-2">{error}</div>}

          <button onClick={login} className="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded">
            Login
          </button>
        </div>
      </div>
    );
  }

  /* ================================================== */
  /* Admin Panel                                        */
  /* ================================================== */

  const actionBtn =
    "mb-3 rounded bg-indigo-600 py-2 font-medium hover:bg-indigo-700 transition";

  return (
    <div className="min-h-screen bg-gray-950 p-8 text-white">
      <h1 className="text-2xl mb-6 text-center">Admin Tools</h1>

      {(status || error) && (
        <div className="max-w-7xl mx-auto mb-4">
          <div
            className={
              "rounded-xl border p-3 text-sm " +
              (error
                ? "border-red-800 bg-red-950/40 text-red-200"
                : "border-gray-800 bg-gray-900/60 text-gray-200")
            }
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <div className="font-medium">{error ? "Error" : "Status"}</div>
                <div className="mt-1 whitespace-pre-wrap">{error || status}</div>

                {lastActionResult && (
                  <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-black/40 p-2 text-xs text-gray-200">
                    {JSON.stringify(lastActionResult, null, 2)}
                  </pre>
                )}
              </div>

              <button
                onClick={() => {
                  setStatus(null);
                  setError(null);
                  setLastActionResult(null);
                }}
                className="shrink-0 rounded-lg border border-gray-700 px-3 py-1 text-xs hover:bg-gray-800"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-7xl mx-auto">
        {/* Live Log */}
        <div className="md:col-span-3 bg-black rounded-xl border border-gray-800 p-4">
          <h3 className="mb-2 text-sm text-gray-300">Live Backend Log</h3>
          <pre
            ref={logRef}
            className="h-[260px] overflow-y-auto text-xs text-green-400 bg-black"
          >
            {liveLog || "Waiting for logs..."}
          </pre>
        </div>

        {/* Admin Actions */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 flex flex-col">
          <h3 className="mb-4 text-lg font-medium">Admin Actions</h3>

          <button onClick={clearLocks} className={actionBtn}>
            Remove All Locks & Reset Replay
          </button>

          <button onClick={restartServices} className={actionBtn}>
            Restart Services
          </button>

          <button onClick={pullFromGit} className={actionBtn}>
            Pull From GitHub
          </button>

          <button onClick={refreshUniverses} className={actionBtn}>
            Refresh Universes (StockAnalysis)
          </button>

          <button
            onClick={logout}
            className="mt-auto rounded bg-gray-700 py-2 font-medium hover:bg-gray-600 transition"
          >
            Logout
          </button>
        </div>

        {/* Swing Replay */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
          <h3 className="mb-2">Swing Replay</h3>
          <div className="text-sm mb-1">Status: {replay?.status ?? "idle"}</div>
          <div className="h-3 bg-gray-800 rounded overflow-hidden mb-1">
            <div
              className="h-full bg-green-600 transition-all"
              style={{ width: `${swingDisplayPct}%` }}
            />
          </div>
          <div className="text-xs mb-2">{swingTargetPct.toFixed(1)}%</div>
          {replay?.current_day && <div className="text-xs mb-3">Day: {replay.current_day}</div>}
          <button onClick={startReplay} className="w-full bg-green-600 hover:bg-green-700 py-2 rounded">
            Start Replay
          </button>
        </div>

        {/* DT Replay */}
        <div className="bg-gray-900 p-6 rounded-xl border border-gray-800">
          <h3 className="mb-2">DT Replay</h3>
          <div className="text-sm mb-1">Status: {dtReplay?.status ?? "idle"}</div>
          <div className="h-3 bg-gray-800 rounded overflow-hidden mb-1">
            <div
              className="h-full bg-sky-600 transition-all"
              style={{ width: `${dtDisplayPct}%` }}
            />
          </div>
          <div className="text-xs mb-2">{dtTargetPct.toFixed(1)}%</div>
          {dtReplay?.current_day && <div className="text-xs mb-3">Day: {dtReplay.current_day}</div>}
          <button onClick={startDtReplay} className="w-full bg-sky-600 hover:bg-sky-700 py-2 rounded">
            Start DT Replay
          </button>
        </div>
      </div>

      {/* tiny status line */}
      {status && <div className="max-w-7xl mx-auto mt-6 text-sm text-gray-300">{status}</div>}
    </div>
  );
}
