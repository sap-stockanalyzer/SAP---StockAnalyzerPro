"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

const API_BASE = "/api/backend";

type KeysResponse = { keys: Record<string, string> };
type StatusResponse = {
  status: Record<string, { present: boolean; value_length: number }>;
};
type TestResponse = {
  source: string;
  ok: boolean;
  missing_broker_keys: string[];
};

const GROUPS: Array<{
  title: string;
  description: string;
  keys: Array<{
    name: string;
    label: string;
    helper?: string;
    secret?: boolean;
    placeholder?: string;
  }>;
}> = [
  {
    title: "Trading",
    description: "Broker credentials and endpoints used for trading + paper execution.",
    keys: [
      {
        name: "ALPACA_API_KEY_ID",
        label: "Alpaca API Key ID",
        secret: true,
        placeholder: "ALPACA_API_KEY_ID",
      },
      {
        name: "ALPACA_API_SECRET_KEY",
        label: "Alpaca API Secret Key",
        secret: true,
        placeholder: "ALPACA_API_SECRET_KEY",
      },
      {
        name: "ALPACA_PAPER_BASE_URL",
        label: "Alpaca Paper Base URL",
        secret: false,
        helper: 'Usually "https://paper-api.alpaca.markets"',
        placeholder: "https://paper-api.alpaca.markets",
      },
    ],
  },
  {
    title: "Market / News",
    description: "News + market data providers used by intel, dashboards, and enrichment.",
    keys: [
      { name: "PERIGON_KEY", label: "Perigon API Key", secret: true, placeholder: "PERIGON_KEY" },
      { name: "FINNHUB_API_KEY", label: "Finnhub API Key", secret: true, placeholder: "FINNHUB_API_KEY" },
      { name: "NEWSAPI_KEY", label: "NewsAPI Key", secret: true, placeholder: "NEWSAPI_KEY" },
      { name: "RSS2JSON_KEY", label: "RSS2JSON Key", secret: true, placeholder: "RSS2JSON_KEY" },
      { name: "MARKETAUX_API_KEY", label: "MarketAux API Key", secret: true, placeholder: "MARKETAUX_API_KEY" },
    ],
  },
  {
    title: "Social",
    description: "Social sources used by sentiment + narrative intelligence.",
    keys: [
      { name: "REDDIT_CLIENT_ID", label: "Reddit Client ID", secret: true, placeholder: "REDDIT_CLIENT_ID" },
      {
        name: "REDDIT_CLIENT_SECRET",
        label: "Reddit Client Secret",
        secret: true,
        placeholder: "REDDIT_CLIENT_SECRET",
      },
      {
        name: "REDDIT_USER_AGENT",
        label: "Reddit User Agent",
        secret: false,
        helper: "Example: AionAnalytics/1.0 (by u/yourname)",
        placeholder: "AionAnalytics/1.0 (by u/yourname)",
      },
      { name: "TWITTER_BEARER", label: "X / Twitter Bearer Token", secret: true, placeholder: "TWITTER_BEARER" },
    ],
  },
];

function maskPreview(v: string) {
  if (!v) return "";
  const trimmed = v.replace(/^"+|"+$/g, "");
  if (trimmed.length <= 8) return "••••••••";
  return `${trimmed.slice(0, 2)}••••••${trimmed.slice(-4)}`;
}

export default function ApiKeysSettingsPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [status, setStatus] = useState<StatusResponse["status"]>({});
  const [loading, setLoading] = useState(true);

  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  const [reveal, setReveal] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TestResponse | null>(null);

  const allKeyNames = useMemo(() => {
    const out: string[] = [];
    for (const g of GROUPS) for (const k of g.keys) out.push(k.name);
    return out;
  }, []);

  const fetchAll = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);

      const [keysRes, statusRes] = await Promise.all([
        fetch(`${API_BASE}/api/settings/keys`, { cache: "no-store" }),
        fetch(`${API_BASE}/api/settings/status`, { cache: "no-store" }),
      ]);

      if (!keysRes.ok) throw new Error(`Failed to load keys: HTTP ${keysRes.status}`);
      if (!statusRes.ok) throw new Error(`Failed to load status: HTTP ${statusRes.status}`);

      const keysJson = (await keysRes.json()) as KeysResponse;
      const statusJson = (await statusRes.json()) as StatusResponse;

      // Ensure all expected keys exist in state (even if backend returns "")
      const merged: Record<string, string> = { ...(keysJson?.keys ?? {}) };
      for (const k of allKeyNames) merged[k] = merged[k] ?? "";

      setValues(merged);
      setStatus(statusJson?.status ?? {});
    } catch (e: any) {
      console.error("[api-keys] load failed", e);
      setError(e?.message ?? "Failed to load API keys");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setKeyValue = (key: string, val: string) => {
    setValues((prev) => ({ ...prev, [key]: val }));
  };

  const toggleReveal = (key: string) => {
    setReveal((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSaveAll = async () => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(null);
      setTestResult(null);

      // Send exactly what we have; backend will preserve unknown keys anyway.
      const res = await fetch(`${API_BASE}/api/settings/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates: values }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Failed to save: HTTP ${res.status} — ${txt}`);
      }

      setSuccess("Saved API keys to .env");
      await fetchAll();
    } catch (e: any) {
      console.error("[api-keys] save failed", e);
      setError(e?.message ?? "Failed to save API keys");
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    try {
      setTesting(true);
      setError(null);
      setSuccess(null);

      const res = await fetch(`${API_BASE}/api/settings/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates: values }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Test failed: HTTP ${res.status} — ${txt}`);
      }

      const out = (await res.json()) as TestResponse;
      setTestResult(out);

      if (out.ok) setSuccess("Settings test passed (basic validation).");
      else setError(`Missing broker keys: ${out.missing_broker_keys.join(", ")}`);
    } catch (e: any) {
      console.error("[api-keys] test failed", e);
      setError(e?.message ?? "Failed to test API keys");
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">API Keys — Settings</h1>
          <p className="text-sm text-muted-foreground">
            Manage provider credentials stored in <span className="font-mono">.env</span>. Changes may require a backend restart.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" disabled={loading || saving} onClick={fetchAll}>
            {loading ? "Loading…" : "Refresh"}
          </Button>
          <Button variant="secondary" size="sm" disabled={loading || testing} onClick={handleTest}>
            {testing ? "Testing…" : "Test"}
          </Button>
          <Button size="sm" disabled={loading || saving} onClick={handleSaveAll}>
            {saving ? "Saving…" : "Save All"}
          </Button>
        </div>
      </div>

      {error && (
        <div className="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2">
          {error}
        </div>
      )}
      {success && (
        <div className="text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-900 rounded-md px-3 py-2">
          {success}
        </div>
      )}

      {testResult && (
        <div className="text-xs text-muted-foreground bg-slate-950/60 border border-slate-800 rounded-md px-3 py-2">
          <span className="font-medium text-slate-200">Test result:</span>{" "}
          {testResult.ok ? (
            <span className="text-emerald-400">OK</span>
          ) : (
            <span className="text-red-400">Missing broker keys</span>
          )}{" "}
          <span className="text-[10px]">({testResult.source})</span>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        {GROUPS.map((group) => (
          <Card key={group.title} className="border-slate-800 bg-slate-950/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">{group.title}</CardTitle>
              <p className="text-xs text-muted-foreground">{group.description}</p>
            </CardHeader>

            <CardContent className="space-y-3 text-xs">
              {group.keys.map((k) => {
                const val = values[k.name] ?? "";
                const st = status?.[k.name];
                const isPresent = st?.present ?? (val?.trim?.() ? true : false);

                const isSecret = !!k.secret;
                const show = !!reveal[k.name];

                return (
                  <div key={k.name} className="space-y-1">
                    <div className="flex items-center justify-between gap-2">
                      <Label className="text-[11px]">{k.label}</Label>
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-[10px] ${
                            isPresent ? "text-emerald-400" : "text-amber-400"
                          }`}
                          title={st ? `length=${st.value_length}` : undefined}
                        >
                          {isPresent ? "Set" : "Missing"}
                        </span>
                        {isSecret && (
                          <button
                            type="button"
                            className="text-[10px] text-muted-foreground hover:text-slate-200"
                            onClick={() => toggleReveal(k.name)}
                          >
                            {show ? "Hide" : "Show"}
                          </button>
                        )}
                      </div>
                    </div>

                    <Input
                      type={isSecret && !show ? "password" : "text"}
                      value={val}
                      placeholder={k.placeholder ?? k.name}
                      onChange={(e) => setKeyValue(k.name, e.target.value)}
                    />

                    {isSecret && !show && val && (
                      <p className="text-[10px] text-muted-foreground">
                        Stored: <span className="font-mono">{maskPreview(val)}</span>
                      </p>
                    )}

                    {k.helper && (
                      <p className="text-[10px] text-muted-foreground">{k.helper}</p>
                    )}
                  </div>
                );
              })}
            </CardContent>

            <CardFooter className="pt-2 flex justify-end">
              <Button size="sm" disabled={loading || saving} onClick={handleSaveAll}>
                {saving ? "Saving…" : "Save"}
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      <p className="text-[11px] text-muted-foreground">
        Note: updating <span className="font-mono">.env</span> usually requires a backend restart unless you’ve added an env reload hook.
      </p>
    </div>
  );
}
