// /frontend/lib/dtTypes.ts
// TypeScript type definitions for DT (day-trading/intraday) backend API responses

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  timestamp?: string;
  uptime_seconds?: number;
  version?: string;
}

/**
 * Readiness probe response
 */
export interface ReadyResponse {
  ready: boolean;
  message?: string;
}

/**
 * Liveness probe response
 */
export interface LiveResponse {
  alive: boolean;
  message?: string;
}

/**
 * Knob value with metadata
 */
export interface KnobValue {
  current: number;
  default: number;
  range: [number, number];
  diff: number;
  diff_pct: number;
}

/**
 * Learning metrics response
 */
export interface LearningMetrics {
  performance_7d: {
    win_rate: number;
    accuracy: number;
    total_trades: number;
    profit_factor: number;
    sharpe_ratio: number;
    avg_win: number;
    avg_loss: number;
    consecutive_wins: number;
    consecutive_losses: number;
  };
  performance_30d: {
    win_rate: number;
    accuracy: number;
    total_trades: number;
  };
  baseline: {
    win_rate: number;
    accuracy: number;
    profit_factor: number;
  };
  model_health: {
    days_since_retrain: number;
    confidence_calibration: number;
    next_retrain_estimate: string;
  };
  missed_opportunities: {
    total_evaluated: number;
    profitable_missed_pct: number;
    missed_pnl_usd: number;
    suggestions: string[];
  };
  dt_brain_knobs: Record<string, KnobValue>;
  trade_quality: {
    total_trades_7d: number;
    win_rate_7d: number;
    profit_factor_7d: number;
    sharpe_ratio_7d: number;
  };
}

/**
 * Replay operation response
 */
export interface ReplayResponse {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  progress?: number;
  message?: string;
}

/**
 * Replay status response
 */
export interface ReplayStatusResponse {
  status: string;
  progress?: number;
  current_day?: string | null;
  eta_secs?: number | null;
  message?: string;
}

/**
 * Job operation response
 */
export interface JobResponse {
  job_id: string;
  status: string;
  created_at: string;
  message?: string;
}

/**
 * Job status response
 */
export interface JobStatusResponse {
  job_id: string;
  status: string;
  progress?: number;
  message?: string;
}

/**
 * Rolling data response (flexible structure)
 */
export interface RollingDataResponse {
  [key: string]: any;
}

/**
 * Position details
 */
export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price?: number;
  unrealized_pnl?: number;
  side?: "long" | "short";
  entry_time?: string;
}

/**
 * Positions response
 */
export interface PositionsResponse {
  positions: Position[];
  total_count?: number;
  total_value?: number;
}

/**
 * Metrics response (flexible structure)
 */
export interface MetricsResponse {
  metrics: any;
  timestamp?: string;
}

/**
 * Generic DT API error response
 */
export interface DtApiErrorResponse {
  detail: string;
  message?: string;
  status_code?: number;
}
