"""
rl_env.py — v1.2 (Minimal Import-Safe RL Environment)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Lightweight StockTradingEnv for reinforcement-learning experiments.
- Mimics Gym-like API (reset, step, render) without any heavy dependencies.
- Safe to import even without gym or stable-baselines installed.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List


class StockTradingEnv:
    """Dependency-light environment scaffold for basic trading simulations."""

    def __init__(self, price_series: Optional[List[float]] = None, starting_cash: float = 10_000.0):
        self.series = price_series or [100.0 for _ in range(500)]
        self.starting_cash = float(starting_cash)
        self.reset()

    # ------------------------------------------------------------------
    # Core Environment API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """Resets environment to initial state."""
        self.idx = 0
        self.cash = self.starting_cash
        self.position = 0  # number of shares held
        self._update_nav()
        return self._obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes one environment step.

        Args:
            action: 0=hold, 1=buy 1 share, 2=sell 1 share

        Returns:
            obs: dict observation
            reward: current NAV (not delta; modify as needed)
            done: True if episode finished
            info: empty dict for compatibility
        """
        price = self.series[self.idx]

        # Execute basic discrete actions
        if action == 1 and self.cash >= price:
            self.cash -= price
            self.position += 1
        elif action == 2 and self.position > 0:
            self.cash += price
            self.position -= 1

        # Advance timestep
        self.idx += 1
        done = self.idx >= len(self.series) - 1

        self._update_nav()
        reward = float(self.nav)

        return self._obs(), reward, done, {}

    def render(self) -> None:
        """Console visualization of environment state."""
        print(
            f"t={self.idx:03d} | price={self.series[self.idx]:.2f} | "
            f"pos={self.position:2d} | cash={self.cash:,.2f} | nav={self.nav:,.2f}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_nav(self) -> None:
        px = self.series[self.idx]
        self.nav = self.cash + self.position * px

    def _obs(self) -> Dict[str, Any]:
        px = self.series[self.idx]
        return {
            "t": int(self.idx),
            "price": float(px),
            "position": int(self.position),
            "cash": float(self.cash),
            "nav": float(self.nav),
        }
