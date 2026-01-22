"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { FileText, RefreshCw, Save, Settings } from "lucide-react";

type FileType = "knobs" | "dt-knobs";

type ConfigFile = {
  name: string;
  displayName: string;
  endpoint: string;
  content: string;
  loaded: boolean;
};

export default function ConfigEditorPage() {
  const [activeFile, setActiveFile] = useState<FileType>("knobs");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [statusType, setStatusType] = useState<"success" | "error" | null>(null);

  const [files, setFiles] = useState<Record<FileType, ConfigFile>>({
    knobs: {
      name: "knobs.env",
      displayName: "EOD/Nightly Configuration",
      endpoint: "/api/backend/settings/knobs",
      content: "",
      loaded: false,
    },
    "dt-knobs": {
      name: "dt_knobs.env",
      displayName: "Intraday/DT Configuration",
      endpoint: "/api/backend/settings/dt-knobs",
      content: "",
      loaded: false,
    },
  });

  // Load a specific file
  async function loadFile(fileType: FileType) {
    const file = files[fileType];
    try {
      const response = await fetch(file.endpoint, {
        method: "GET",
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Failed to load ${file.name}: ${response.status}`);
      }

      const data = await response.json();
      setFiles((prev) => ({
        ...prev,
        [fileType]: {
          ...prev[fileType],
          content: data.content || "",
          loaded: true,
        },
      }));
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      console.error(`Failed to load ${file.name}:`, error);
      showStatus(`Failed to load ${file.name}: ${errorMessage}`, "error");
    }
  }

  // Load both files on mount
  useEffect(() => {
    async function loadBothFiles() {
      setLoading(true);
      await Promise.all([loadFile("knobs"), loadFile("dt-knobs")]);
      setLoading(false);
    }
    loadBothFiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save current file
  async function saveFile() {
    const file = files[activeFile];
    setSaving(true);
    setStatusMessage(null);

    try {
      const response = await fetch(file.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content: file.content }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save ${file.name}: ${response.status}`);
      }

      const data = await response.json();
      showStatus(data.message || `${file.name} saved successfully`, "success");
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      console.error(`Failed to save ${file.name}:`, error);
      showStatus(`Failed to save ${file.name}: ${errorMessage}`, "error");
    } finally {
      setSaving(false);
    }
  }

  // Reload current file
  async function reloadFile() {
    setStatusMessage(null);
    await loadFile(activeFile);
    showStatus(`${files[activeFile].name} reloaded`, "success");
  }

  // Update file content
  function updateContent(content: string) {
    setFiles((prev) => ({
      ...prev,
      [activeFile]: {
        ...prev[activeFile],
        content,
      },
    }));
  }

  // Show status message
  function showStatus(message: string, type: "success" | "error") {
    setStatusMessage(message);
    setStatusType(type);
    setTimeout(() => {
      setStatusMessage(null);
      setStatusType(null);
    }, 5000);
  }

  const currentFile = files[activeFile];

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white">
      <div className="mx-auto max-w-7xl px-4 py-8">
        {/* Header */}
        <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <Settings className="h-8 w-8" />
              Configuration Editor
            </h1>
            <p className="text-sm text-white/60 mt-1">
              Edit knobs.env and dt_knobs.env configuration files
            </p>
          </div>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div
            className={`mb-6 rounded-xl border p-4 ${
              statusType === "error"
                ? "border-red-500/40 bg-red-500/10 text-red-200"
                : "border-green-500/40 bg-green-500/10 text-green-200"
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">{statusMessage}</div>
              <button
                onClick={() => setStatusMessage(null)}
                className="shrink-0 text-xs opacity-60 hover:opacity-100"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Main Card */}
        <Card className="border-white/10 bg-white/5">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <CardTitle className="text-xl">
                {currentFile.displayName}
              </CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="gap-1">
                  <FileText className="h-3 w-3" />
                  {currentFile.name}
                </Badge>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* File Tabs */}
            <div className="flex gap-2 border-b border-white/10 pb-2">
              <button
                onClick={() => setActiveFile("knobs")}
                className={`px-4 py-2 rounded-t-lg text-sm font-medium transition ${
                  activeFile === "knobs"
                    ? "bg-white/10 text-white border-b-2 border-blue-500"
                    : "text-white/60 hover:text-white hover:bg-white/5"
                }`}
              >
                knobs.env
              </button>
              <button
                onClick={() => setActiveFile("dt-knobs")}
                className={`px-4 py-2 rounded-t-lg text-sm font-medium transition ${
                  activeFile === "dt-knobs"
                    ? "bg-white/10 text-white border-b-2 border-blue-500"
                    : "text-white/60 hover:text-white hover:bg-white/5"
                }`}
              >
                dt_knobs.env
              </button>
            </div>

            {/* Help Text */}
            <div className="text-xs text-white/60 bg-white/5 rounded-lg p-3">
              {activeFile === "knobs" ? (
                <>
                  <strong>knobs.env</strong> controls EOD/Nightly (swing)
                  trading bot configuration including allocation, stop-loss,
                  take-profit, and other parameters.
                </>
              ) : (
                <>
                  <strong>dt_knobs.env</strong> controls Intraday/Day-trading
                  bot configuration including position limits, risk parameters,
                  and real-time trading settings.
                </>
              )}
            </div>

            {/* Editor */}
            {loading ? (
              <div className="h-[600px] flex items-center justify-center bg-black rounded-lg border border-white/10">
                <div className="text-white/60">Loading configuration files...</div>
              </div>
            ) : (
              <div className="space-y-3">
                <textarea
                  value={currentFile.content}
                  onChange={(e) => updateContent(e.target.value)}
                  className="w-full h-[600px] bg-black text-green-400 font-mono text-sm rounded-lg border border-white/10 p-4 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  spellCheck={false}
                  placeholder={`Edit ${currentFile.name} content here...`}
                />

                {/* Action Buttons */}
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs text-white/60">
                    {currentFile.content.split("\n").length} lines •{" "}
                    {currentFile.content.length} characters
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={reloadFile}
                      disabled={saving || loading}
                      className="gap-2"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Reload
                    </Button>
                    <Button
                      onClick={saveFile}
                      disabled={saving || loading}
                      className="gap-2"
                    >
                      <Save className="h-4 w-4" />
                      {saving ? "Saving..." : "Save Changes"}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Footer Help */}
        <div className="mt-6 text-xs text-white/40 text-center">
          Changes will be applied immediately after saving. Make sure to review
          your changes carefully before saving.
        </div>
      </div>
    </main>
  );
}
