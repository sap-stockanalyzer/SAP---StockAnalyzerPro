import { NextRequest, NextResponse } from "next/server";

/**
 * Next.js (App Router) proxy to the dt_backend.
 * ✔ Safe for Next.js 15+ (awaited params)
 * ✔ Node 18+ streaming-safe (duplex fix)
 * ✔ Server-side env resolution
 * ✔ No silent localhost fallbacks
 */

function getDtBackendBaseUrl(): string {
  const url =
    process.env.DT_BACKEND_URL ||
    process.env.NEXT_PUBLIC_DT_BACKEND_URL ||
    'http://localhost:8010'; // NOTE: localhost default is for development only. Production should set DT_BACKEND_URL or NEXT_PUBLIC_DT_BACKEND_URL explicitly.

  return url.replace(/\/+$/, "");
}

function buildTargetUrl(req: NextRequest, pathParts: string[]) {
  const base = getDtBackendBaseUrl();
  const incoming = new URL(req.url);
  
  // pathParts are the URL segments after /api/dt/ (e.g., ['replay', 'status'])
  // We need to build: http://localhost:8010/api/replay/status
  const path = pathParts.map(encodeURIComponent).join("/");
  const target = new URL(`${base}/api/${path}`);

  // Preserve query string
  target.search = incoming.search;
  return target;
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  // Next.js 15+: ctx.params may be a Promise
  const { path = [] } = await ctx.params;

  const target = buildTargetUrl(req, Array.isArray(path) ? path : []);
  const headers = new Headers(req.headers);

  // Remove hop-by-hop headers
  headers.delete("host");
  headers.delete("content-length");

  const hasBody = req.method !== "GET" && req.method !== "HEAD";

  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers,
    body: hasBody ? req.body : undefined,
    redirect: "manual",

    // 🔥 REQUIRED for Node 18+ when forwarding streamed request bodies
    duplex: hasBody ? "half" : undefined,
  };

  const upstream = await fetch(target, init);

  // Mirror response headers safely
  const resHeaders = new Headers(upstream.headers);
  resHeaders.delete("content-encoding");

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: resHeaders,
  });
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function PUT(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function PATCH(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function DELETE(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

export async function OPTIONS(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}
