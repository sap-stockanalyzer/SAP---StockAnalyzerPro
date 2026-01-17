# SAP Frontend (StockAnalyzerPro) — v1.4.2

Next.js 15 + TypeScript + Tailwind CSS
Dark, sleek dashboard with system dropdown.

## Quickstart
```bash
npm install
npm run dev
```

## Configuration

Set your backend URLs:
```bash
cp .env.example .env.local
# Edit the following variables:
# - BACKEND_URL (main backend, port 8000)
# - DT_BACKEND_URL (DT backend, port 8010)
# - NEXT_PUBLIC_BACKEND_URL (client-side fallback)
# - NEXT_PUBLIC_DT_BACKEND_URL (client-side fallback)
```

## API Proxy Routes

The frontend includes Next.js API routes that proxy requests to backend services:

### `/api/backend/[...path]`
Forwards requests to the main backend (EOD/Nightly) running on port 8000.

**Path Transformation:**
- `/api/backend/bots/page` → `http://backend:8000/api/bots/page`
- `/api/backend/dashboard/metrics` → `http://backend:8000/dashboard/metrics` (no /api prefix)
- `/api/backend/admin/login` → `http://backend:8000/admin/login` (no /api prefix)

### `/api/dt/[...path]`
Forwards requests to the DT backend (Intraday) running on port 8010.

**Path Transformation:**
- `/api/dt/health` → `http://dt-backend:8010/health`
- `/api/dt/jobs/status` → `http://dt-backend:8010/jobs/status`
- `/api/dt/data/positions` → `http://dt-backend:8010/data/positions`

## Configuration Editor

Access the configuration editor at `/bots/config` to edit:
- `knobs.env` - EOD/Nightly configuration
- `dt_knobs.env` - Intraday/DT configuration
