"use client";

import { useEffect, useState } from 'react';

interface PnLData {
  daily_pnl: number;
  weekly_pnl: number;
  monthly_pnl: number;
  ytd_pnl: number;
  win_rate: number;
  total_trades: number;
  avg_trade_size: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  pnl_by_strategy: Record<string, {
    pnl: number;
    trades: number;
    win_rate: number;
  }>;
  pnl_by_feature: Record<string, {
    pnl: number;
    contribution: number;
  }>;
  daily_history: Array<{
    date: string;
    pnl: number;
    trades: number;
  }>;
  timestamp: string;
}

export default function Dashboard() {
  const [pnlData, setPnlData] = useState<PnLData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchPnLData = async () => {
      try {
        const response = await fetch('/api/backend/pnl/dashboard');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setPnlData(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching P&L data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load P&L data');
        setLoading(false);
      }
    };
    
    fetchPnLData();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchPnLData, 30000);
    return () => clearInterval(interval);
  }, []);
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl text-gray-400">Loading P&L Dashboard...</div>
      </div>
    );
  }
  
  if (error || !pnlData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl text-red-400">
          {error || 'Failed to load P&L data'}
        </div>
      </div>
    );
  }
  
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">P&L Dashboard</h1>
        <div className="text-sm text-gray-400">
          Last updated: {new Date(pnlData.timestamp).toLocaleTimeString()}
        </div>
      </div>
      
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Daily P&L</div>
          <div className={`text-3xl font-bold ${pnlData.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${pnlData.daily_pnl.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Win Rate</div>
          <div className="text-3xl font-bold text-white">
            {(pnlData.win_rate * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Sharpe Ratio</div>
          <div className="text-3xl font-bold text-white">
            {pnlData.sharpe_ratio.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Max Drawdown</div>
          <div className="text-3xl font-bold text-red-400">
            {pnlData.max_drawdown_pct.toFixed(1)}%
          </div>
        </div>
      </div>
      
      {/* Period Performance */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Weekly P&L</div>
          <div className={`text-2xl font-bold ${pnlData.weekly_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${pnlData.weekly_pnl.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Monthly P&L</div>
          <div className={`text-2xl font-bold ${pnlData.monthly_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${pnlData.monthly_pnl.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="text-sm text-gray-400 mb-2">YTD P&L</div>
          <div className={`text-2xl font-bold ${pnlData.ytd_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${pnlData.ytd_pnl.toFixed(2)}
          </div>
        </div>
      </div>
      
      {/* P&L by Strategy */}
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">P&L by Strategy</h2>
        <div className="space-y-3">
          {Object.entries(pnlData.pnl_by_strategy).map(([strategy, data]) => (
            <div key={strategy} className="flex items-center justify-between p-3 bg-gray-900 rounded">
              <div className="flex items-center space-x-4">
                <span className="text-white font-semibold">{strategy}</span>
                <span className="text-gray-400 text-sm">{data.trades} trades</span>
                <span className="text-gray-400 text-sm">
                  {(data.win_rate * 100).toFixed(1)}% win rate
                </span>
              </div>
              <span className={`font-bold ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${data.pnl.toFixed(2)}
              </span>
            </div>
          ))}
          {Object.keys(pnlData.pnl_by_strategy).length === 0 && (
            <div className="text-gray-400 text-center py-4">No strategy data available</div>
          )}
        </div>
      </div>
      
      {/* P&L by Feature */}
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">P&L by Feature</h2>
        <div className="space-y-3">
          {Object.entries(pnlData.pnl_by_feature)
            .sort(([, a], [, b]) => b.pnl - a.pnl)
            .slice(0, 10)
            .map(([feature, data]) => (
              <div key={feature} className="flex items-center justify-between p-3 bg-gray-900 rounded">
                <div className="flex items-center space-x-4">
                  <span className="text-white font-semibold">{feature}</span>
                  <span className="text-gray-400 text-sm">
                    {(data.contribution * 100).toFixed(1)}% contribution
                  </span>
                </div>
                <span className={`font-bold ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${data.pnl.toFixed(2)}
                </span>
              </div>
            ))}
          {Object.keys(pnlData.pnl_by_feature).length === 0 && (
            <div className="text-gray-400 text-center py-4">No feature data available</div>
          )}
        </div>
      </div>
      
      {/* Trading Stats */}
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h2 className="text-xl font-bold text-white mb-4">Trading Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-400">Total Trades</div>
            <div className="text-xl font-bold text-white">{pnlData.total_trades}</div>
          </div>
          <div>
            <div className="text-sm text-gray-400">Avg Trade Size</div>
            <div className="text-xl font-bold text-white">
              ${pnlData.avg_trade_size.toFixed(2)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
