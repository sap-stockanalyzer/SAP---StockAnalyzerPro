// /frontend/lib/types.ts
// TypeScript type definitions for API responses

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  timestamp: string;
  uptime_seconds?: number;
  version?: string;
  services?: {
    broker?: { status: string; message?: string };
    data?: { status: string; message?: string };
    ml_models?: { status: string; message?: string };
  };
}

/**
 * Metrics overview response
 */
export interface MetricsOverviewResponse {
  accuracy?: number;
  total_predictions?: number;
  active_models?: number;
  last_updated?: string;
}

/**
 * Portfolio holdings response
 */
export interface PortfolioHoldingsResponse {
  holdings: PortfolioHolding[];
  total_value: number;
  last_updated: string;
}

export interface PortfolioHolding {
  symbol: string;
  shares: number;
  avg_price: number;
  current_price: number;
  market_value: number;
  pnl: number;
  pnl_percent: number;
}

/**
 * Intraday snapshot response
 */
export interface IntradaySnapshotResponse {
  snapshots: IntradaySnapshot[];
  timestamp: string;
  count: number;
}

export interface IntradaySnapshot {
  symbol: string;
  timestamp: string;
  price: number;
  volume: number;
  change_percent: number;
}

/**
 * Bots page response
 */
export interface BotsPageResponse {
  swing?: BotFamilyStatus;
  dt?: BotFamilyStatus;
  timestamp: string;
}

export interface BotFamilyStatus {
  status: {
    bots: Record<string, BotStatus>;
    total_equity: number;
    active_count: number;
  };
}

export interface BotStatus {
  equity: number;
  pnl: number;
  pnl_percent: number;
  positions_count: number;
  status: string;
  equity_curve?: EquityPoint[];
}

export interface EquityPoint {
  t: string;
  value: number;
}

/**
 * Settings responses
 */
export interface KnobsResponse {
  content: string;
}

export interface SaveKnobsResponse {
  ok: boolean;
  message: string;
}

export interface SettingsKeysResponse {
  keys: Record<string, string>;
}

export interface UpdateSettingsResponse {
  status: string;
  message?: string;
}

/**
 * Model training response
 */
export interface ModelTrainResponse {
  status: string;
  message: string;
  duration_seconds?: number;
}

/**
 * Drift monitoring response
 */
export interface DriftResponse {
  report_path: string;
  drift_detected: boolean;
  metrics?: Record<string, any>;
}

/**
 * Generic API error response
 */
export interface ApiErrorResponse {
  detail: string;
  message?: string;
  status_code?: number;
}

/**
 * Metric log entry
 */
export type MetricLog = Record<
  string,
  { R2?: number; MAE?: number; RMSE?: number; ACC?: number; F1?: number; AUC?: number }
>;
