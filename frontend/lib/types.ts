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

// ========== Optimizer Types ==========

export interface OptimizerParams {
  target_horizon: "1w" | "1m" | "3m" | "6m" | "1y";
  risk_tolerance: number; // 0.01 to 0.50
  max_positions: number;
  min_allocation_pct?: number;
  rebalance_frequency?: "daily" | "weekly" | "monthly";
}

export interface OptimizerResults {
  efficient_frontier: Point[];
  recommended_portfolio: Allocation[];
  expected_return: number;
  expected_risk: number;
  sharpe_ratio: number;
  status: "success" | "partial" | "failed";
  warnings?: string[];
  computation_time_ms: number;
}

export interface Point {
  risk: number;
  return: number;
}

export interface Allocation {
  symbol: string;
  allocation_pct: number;
  shares?: number;
  dollar_amount?: number;
}

export interface EfficientFrontierData {
  points: Point[];
  current_portfolio?: Point;
  riskfree_rate: number;
  capital_allocation_line: Point[];
}

// ========== Report Types ==========

export interface Report {
  id: string;
  type: "backtest" | "model" | "performance" | "analysis";
  name: string;
  description?: string;
  created_at: string;
  updated_at?: string;
  size_bytes: number;
  formats_available: ("pdf" | "csv" | "json")[];
  status: "ready" | "generating" | "failed";
}

export interface ReportDetails extends Report {
  metrics: Record<string, any>;
  summary: ReportSummary;
  charts: ChartInfo[];
  data_points?: any[];
}

export interface ReportSummary {
  total_trades?: number;
  win_rate?: number;
  profit_factor?: number;
  max_drawdown?: number;
  sharpe_ratio?: number;
  return_pct?: number;
}

export interface ChartInfo {
  id: string;
  title: string;
  type: string;
  data: any;
}

export interface ReportGenerationParams {
  report_type: string;
  start_date?: string;
  end_date?: string;
  include_metrics?: boolean;
  include_charts?: boolean;
}

// ========== Model Registry Types ==========

export interface ModelEntry {
  id: string;
  name: string;
  description?: string;
  type: "swing" | "intraday" | "ensemble";
  version: string;
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  trained_at: string;
  training_data_size: number;
  features_count: number;
  status: "active" | "inactive" | "deprecated";
}

export interface ModelDetails extends ModelEntry {
  performance_metrics: PerformanceMetrics;
  training_history: TrainingStep[];
  hyperparameters: Record<string, any>;
  feature_importance: FeatureImportance[];
  training_config: TrainingConfig;
}

export interface PerformanceMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  confusion_matrix: number[][];
  per_class_metrics?: Record<string, any>;
}

export interface FeatureImportance {
  feature_name: string;
  importance_score: number;
  rank: number;
}

export interface TrainingStep {
  epoch: number;
  timestamp: string;
  training_loss: number;
  validation_loss: number;
  metrics: Record<string, number>;
}

export interface TrainingConfig {
  algorithm: string;
  hyperparameters: Record<string, any>;
  training_data_source: string;
  validation_split: number;
  test_split: number;
  random_seed: number;
}

export interface ModelUploadData {
  file: File;
  name: string;
  type: "swing" | "intraday" | "ensemble";
  description?: string;
  hyperparameters?: Record<string, any>;
}
