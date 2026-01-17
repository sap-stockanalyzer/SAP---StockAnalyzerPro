"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface TestResult {
  test_name: string;
  status: "pass" | "fail";
  http_status: number;
  response_time_ms: number;
  timestamp: string;
  message: string;
  error?: string;
}

interface TestButtonProps {
  testName: string;
  endpoint: string;
  category: string;
}

type TestState = "idle" | "loading" | "pass" | "fail";

export default function TestButton({ testName, endpoint, category }: TestButtonProps) {
  const [state, setState] = useState<TestState>("idle");
  const [result, setResult] = useState<TestResult | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const runTest = async () => {
    setState("loading");
    setResult(null);

    try {
      // Determine if this is a testing endpoint (POST) or regular endpoint (GET)
      const isTestingEndpoint = endpoint.includes("/testing/");
      const method = isTestingEndpoint ? "POST" : "GET";
      
      const startTime = performance.now();
      const response = await fetch(endpoint, {
        method,
        cache: "no-store",
      });
      const endTime = performance.now();
      const responseTimeMs = endTime - startTime;

      // For testing endpoints, the response is already a TestResult
      if (isTestingEndpoint) {
        const data = await response.json();
        setResult(data);
        setState(data.status === "pass" ? "pass" : "fail");
      } else {
        // For regular endpoints, create a TestResult manually
        const isSuccess = response.ok;
        const data = await response.json().catch(() => null);
        
        const testResult: TestResult = {
          test_name: testName,
          status: isSuccess ? "pass" : "fail",
          http_status: response.status,
          response_time_ms: responseTimeMs,
          timestamp: new Date().toISOString(),
          message: isSuccess 
            ? `${testName} endpoint responded successfully`
            : `${testName} endpoint failed with status ${response.status}`,
          error: !isSuccess ? `HTTP ${response.status} ${response.statusText}` : undefined,
        };
        
        setResult(testResult);
        setState(isSuccess ? "pass" : "fail");
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      setResult({
        test_name: testName,
        status: "fail",
        http_status: 0,
        response_time_ms: 0,
        timestamp: new Date().toISOString(),
        message: "Failed to reach backend",
        error: errorMessage,
      });
      setState("fail");
    }
  };

  const copyToClipboard = () => {
    if (!result) return;

    const markdown = `## Test Result: ${result.test_name}
- **Status**: ${result.status === "pass" ? "✅ PASS" : "❌ FAIL"}
- **HTTP Status**: ${result.http_status}
- **Response Time**: ${result.response_time_ms.toFixed(0)}ms
- **Timestamp**: ${result.timestamp}
- **Message**: ${result.message}
${result.error ? `- **Error**: ${result.error}` : ""}`;

    navigator.clipboard.writeText(markdown);
  };

  const getStateIcon = () => {
    switch (state) {
      case "loading":
        return "⏳";
      case "pass":
        return "✅";
      case "fail":
        return "❌";
      default:
        return "";
    }
  };

  const getStateBadge = () => {
    switch (state) {
      case "loading":
        return <Badge variant="outline" className="ml-2 bg-blue-500/10 text-blue-400 border-blue-500/30">Loading...</Badge>;
      case "pass":
        return <Badge variant="outline" className="ml-2 bg-green-500/10 text-green-400 border-green-500/30">PASS</Badge>;
      case "fail":
        return <Badge variant="outline" className="ml-2 bg-red-500/10 text-red-400 border-red-500/30">FAIL</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-1">
          <Button
            onClick={runTest}
            disabled={state === "loading"}
            variant="outline"
            className="min-w-[200px] justify-start"
          >
            <span className="mr-2">{getStateIcon()}</span>
            {testName}
          </Button>
          {getStateBadge()}
          {result && (
            <>
              <span className="text-xs text-slate-400">
                {result.http_status}
              </span>
              <span className="text-xs text-slate-400">
                {result.response_time_ms.toFixed(0)}ms
              </span>
            </>
          )}
        </div>
        {result && (
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs"
            >
              {showDetails ? "Hide" : "Details"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={copyToClipboard}
              className="text-xs"
            >
              Copy
            </Button>
          </div>
        )}
      </div>

      {showDetails && result && (
        <Card className="border-slate-700 bg-slate-800/50">
          <CardContent className="p-4 space-y-2">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-slate-400">Status:</span>{" "}
                <span className={result.status === "pass" ? "text-green-400" : "text-red-400"}>
                  {result.status.toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-slate-400">HTTP:</span>{" "}
                <span className="text-white">{result.http_status}</span>
              </div>
              <div>
                <span className="text-slate-400">Response Time:</span>{" "}
                <span className="text-white">{result.response_time_ms.toFixed(0)}ms</span>
              </div>
              <div>
                <span className="text-slate-400">Timestamp:</span>{" "}
                <span className="text-white text-xs">
                  {new Date(result.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
            <div className="pt-2 border-t border-slate-700">
              <div className="text-slate-400 text-xs mb-1">Message:</div>
              <div className="text-white text-sm">{result.message}</div>
            </div>
            {result.error && (
              <div className="pt-2 border-t border-slate-700">
                <div className="text-red-400 text-xs mb-1">Error:</div>
                <div className="text-red-300 text-sm font-mono bg-red-500/10 p-2 rounded">
                  {result.error}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
