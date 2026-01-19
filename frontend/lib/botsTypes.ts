/**
 * Type definitions for bots API
 * 
 * Centralized types for swing (EOD) and day trading (intraday) bots.
 * These types are used across the bots page and related components.
 */

// ============================================
// EOD (Swing) Bot Types
// ============================================

export type EodBotConfig = {
  max_alloc: number;
  max_positions: number;
  stop_loss: number;
  take_profit: number;
  aggression: number;
  enabled?: boolean;
};

export type EodConfigResponse = { 
  configs: Record<string, EodBotConfig> 
};

export type EodStatusResponse = {
  running?: boolean;
  last_update?: string;
  bots?: Record<
    string,
    {
      cash?: number;
      invested?: number;
      allocated?: number;
      holdings_count?: number;
      equity?: number;
      last_update?: string;
      type?: string;
      equity_curve?: Array<{ t?: string; value?: number } | number>;
      pnl_curve?: Array<{ t?: string; value?: number } | number>;
      positions?: any[];
      enabled?: boolean;
    }
  >;
};

// ============================================
// Intraday (Day Trading) Bot Types
// ============================================

export type IntradayPnlResponse = {
  total?: { realized?: number; unrealized?: number; total?: number; updated_at?: string };
  bots?: Record<string, { realized?: number; unrealized?: number; total?: number }>;
  date?: string;
};

export type IntradayFill = {
  ts?: string;
  time?: string;
  symbol?: string;
  side?: string;
  action?: string;
  qty?: number;
  price?: number;
  pnl?: number;
};

export type IntradaySignal = {
  ts?: string;
  time?: string;
  symbol?: string;
  action?: string;
  side?: string;
  confidence?: number;
};

// ============================================
// Combined Bots Page Bundle
// ============================================

// Note: Some fields use `any` type for backward compatibility with dynamic
// backend responses. The backend may return different shapes for status,
// configs, and log data depending on availability and version.
// This preserves existing behavior while providing type safety for known fields.

export type BotsPageBundle = {
  as_of?: string;
  swing?: {
    status?: EodStatusResponse | any;  // Can be error object or status
    configs?: EodConfigResponse | any;  // Can be error object or configs
    log_days?: any;  // Dynamic log data structure
  };
  intraday?: {
    status?: any;  // Intraday status structure varies
    configs?: any;  // Intraday config structure varies
    log_days?: any;  // Dynamic log data structure
    pnl_last_day?: IntradayPnlResponse | any;  // Can be error object or PnL
    tape?: {
      updated_at?: string | null;
      fills?: IntradayFill[];
      signals?: IntradaySignal[];
    };
  };
};

// ============================================
// Draft Configuration (UI State)
// ============================================

export type BotDraft = EodBotConfig & {
  penny_only?: boolean;
  allow_etfs?: boolean;
  max_daily_trades?: number;
};

// ============================================
// API Request/Response Types
// ============================================

export interface UpdateBotConfigRequest {
  bot_key: string;
  config: EodBotConfig | BotDraft;
}

export interface UpdateBotConfigResponse {
  ok?: boolean;
  status?: string;
  message?: string;
}
