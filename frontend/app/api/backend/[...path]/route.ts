import { NextRequest, NextResponse } from "next/server";

/**
 * Backend proxy (swing backend)
 * ✔ Node 18+ compatible
 * ✔ Fixes "duplex option is required"
 * ✔ Works with streaming bodies
 */

function getBackendBaseUrl() {
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "/api/backend";
}

function buildTargetUrl(req: NextRequest, pathParts: string[]) {
  const base = getBackendBaseUrl().replace(/\/+$/, "");
  const incoming = new URL(req.url);
  const path = pathParts.map(encodeURIComponent).join("/");
  const target = new URL(`${base}/${path}`);
  target.search = incoming.search;
  return target;
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path = [] } = await ctx.params;
  const target = buildTargetUrl(req, Array.isArray(path) ? path : []);

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("content-length");

  const hasBody = req.method !== "GET" && req.method !== "HEAD";

  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers,
    body: hasBody ? req.body : undefined,
    redirect: "manual",

    // 🔥 REQUIRED FOR NODE FETCH WITH STREAMING BODY
    duplex: hasBody ? "half" : undefined,
  };

  const upstream = await fetch(target, init);

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

export async function OPTIONS() {
  return NextResponse.json({}, { status: 204 });
}
