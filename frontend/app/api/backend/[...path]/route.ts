/**
 * Next.js API Route: /api/backend/[...path]
 * 
 * This proxy forwards frontend requests to the main backend server.
 * 
 * Routing Logic:
 * - Dashboard routes (/dashboard/*) → forward as-is (no /api prefix)
 * - Admin routes (/admin/*) → forward as-is (no /api prefix)
 * - All other routes → prepend '/api' to path
 * 
 * Example transformations:
 * - /api/backend/bots/page → http://backend:8000/api/bots/page
 * - /api/backend/dashboard/metrics → http://backend:8000/dashboard/metrics
 * - /api/backend/admin/login → http://backend:8000/admin/login
 * 
 * Note: Port consolidation - backend now uses unified port 8000
 * (previously split between 8000 and 8001).
 */

import { NextRequest, NextResponse } from "next/server";

// Supported HTTP methods
const ALLOWED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"];

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleRequest("GET", request, context);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleRequest("POST", request, context);
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleRequest("PUT", request, context);
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleRequest("PATCH", request, context);
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleRequest("DELETE", request, context);
}

export async function OPTIONS(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return handleRequest("OPTIONS", request, context);
}

async function handleRequest(
  method: string,
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
): Promise<NextResponse> {
  try {
    // Handle Next.js 15 async params
    const params = await Promise.resolve(context.params);
    const pathParts = params.path || [];

    // Get backend URL from environment
    const backendUrl =
      process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL;

    if (!backendUrl) {
      console.error("[Backend Proxy] BACKEND_URL not configured");
      return NextResponse.json(
        { error: "Backend URL not configured" },
        { status: 502 }
      );
    }

    // Construct target path with appropriate prefix
    let targetPath: string;
    if (pathParts.length > 0) {
      const firstSegment = pathParts[0];
      
      // Dashboard and admin routes don't need /api prefix
      if (firstSegment === "dashboard" || firstSegment === "admin") {
        targetPath = `/${pathParts.join("/")}`;
      } else {
        // All other routes get /api prefix
        targetPath = `/api/${pathParts.join("/")}`;
      }
    } else {
      targetPath = "/api";
    }

    // Preserve query string
    const url = new URL(request.url);
    const queryString = url.search;
    const targetUrl = `${backendUrl}${targetPath}${queryString}`;

    console.log(
      `[Backend Proxy] ${method} ${url.pathname} → ${targetUrl}`
    );

    // Prepare request headers
    const headers = new Headers();
    request.headers.forEach((value, key) => {
      // Skip host header as it should be set by fetch
      if (key.toLowerCase() !== "host") {
        headers.set(key, value);
      }
    });

    // Prepare fetch options
    const fetchOptions: RequestInit & { duplex?: string } = {
      method,
      headers,
      // Handle Node 18+ duplex streaming for request bodies
      duplex: "half",
    };

    // Add body for methods that support it
    if (["POST", "PUT", "PATCH"].includes(method)) {
      try {
        const body = await request.arrayBuffer();
        if (body.byteLength > 0) {
          fetchOptions.body = body;
        }
      } catch (e) {
        console.warn("[Backend Proxy] Failed to read request body:", e);
      }
    }

    // Forward request to backend
    const response = await fetch(targetUrl, fetchOptions);

    // Get response body
    const responseBody = await response.arrayBuffer();

    // Create response with same status and headers
    const proxyResponse = new NextResponse(responseBody, {
      status: response.status,
      statusText: response.statusText,
    });

    // Copy response headers
    response.headers.forEach((value, key) => {
      proxyResponse.headers.set(key, value);
    });

    // Add cache control headers for client-side caching
    // Backend returns fresh data always - client handles caching with TTL
    proxyResponse.headers.set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    proxyResponse.headers.set("Pragma", "no-cache");
    proxyResponse.headers.set("Expires", "0");

    return proxyResponse;
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error("[Backend Proxy] Request failed:", error);
    return NextResponse.json(
      {
        error: "Backend request failed",
        message: errorMessage,
      },
      { status: 502 }
    );
  }
}
