#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple  # Python 3.8 compatible
import random

# --- Archival paths & helper (your enhancement) ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # src/simulators -> src -> repo
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")


def archive_input_csv(csv_path: str) -> str:
    """Archive input CSV to data/raw with timestamp (your enhancement)"""
    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(csv_path)
    safe_base = base.replace(" ", "_")
    archived = os.path.join(RAW_DIR, f"{ts}__{safe_base}")
    shutil.copy2(csv_path, archived)
    return archived


# ---------------- Core classes ----------------

@dataclass
class MatchContext:
    current_over: float = 0.0
    current_wickets: int = 0
    current_score: int = 0
    target_score: Optional[int] = None
    balls_remaining: int = 120
    wickets_remaining: int = 10

    @property
    def current_run_rate(self) -> float:
        balls_faced = 120 - self.balls_remaining
        return (self.current_score * 6) / max(balls_faced, 1)

    @property
    def required_run_rate(self) -> float:
        if self.target_score is None:
            return 6.0
        runs_needed = max(0, self.target_score - self.current_score)
        return (runs_needed * 6) / max(self.balls_remaining, 1)


class KellyPositionSizer:
    """Kelly Criterion position sizing for optimal bet sizing"""

    @staticmethod
    def calculate_kelly_fraction(win_prob: float, avg_win: float, avg_loss: float,
                                 max_fraction: float = 0.25) -> float:
        """Calculate Kelly fraction with safety cap"""
        if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
        b = avg_win / abs(avg_loss)
        p = win_prob
        q = 1 - p
        kelly_f = (b * p - q) / b
        return max(0, min(kelly_f, max_fraction))

    @staticmethod
    def estimate_win_probability(odds: float, market_edge: float = 0.02) -> float:
        """Convert decimal odds to win probability with market edge adjustment"""
        implied_prob = 1.0 / odds
        return implied_prob * (1 - market_edge)


class EnhancedRiskManager:
    """Advanced risk management with Kelly sizing and drawdown protection"""

    def __init__(self,
                 max_daily_loss: float = 50.0,
                 max_position_risk: float = 5.0,
                 var_confidence: float = 0.95,
                 bankroll: float = 1000.0):
        self.max_daily_loss = max_daily_loss
        self.max_position_risk = max_position_risk
        self.var_confidence = var_confidence
        self.bankroll = bankroll

        # Performance tracking
        self.daily_pnl = 0.0
        self.trade_history = []
        self.consecutive_losses = 0

    def update_performance(self, pnl: float, win: bool):
        """Update performance metrics after each trade"""
        self.daily_pnl += pnl
        self.trade_history.append(pnl)
        self.consecutive_losses = 0 if win else self.consecutive_losses + 1

    def calculate_var(self, lookback: int = 50) -> float:
        """Calculate Value at Risk from recent trade history"""
        if len(self.trade_history) < 10:
            return self.max_position_risk  # Conservative default
        recent_pnls = self.trade_history[-lookback:]
        return abs(np.percentile(recent_pnls, (1 - self.var_confidence) * 100))

    def get_position_size(self, entry_odds: float, tp_ticks: int, sl_ticks: int,
                          tick_value: float) -> float:
        """Calculate optimal position size using Kelly + risk constraints"""

        # Kelly calculation
        win_prob = KellyPositionSizer.estimate_win_probability(entry_odds)
        avg_win = tp_ticks * tick_value
        avg_loss = sl_ticks * tick_value

        kelly_fraction = KellyPositionSizer.calculate_kelly_fraction(
            win_prob, avg_win, avg_loss
        )

        # Apply various risk constraints
        kelly_stake = kelly_fraction * self.bankroll
        var_stake = self.calculate_var() * 2  # 2x VaR limit
        max_risk_stake = self.max_position_risk
        daily_limit_stake = max(1.0, (self.max_daily_loss - abs(self.daily_pnl)) / 3)

        # Reduce size after consecutive losses (drawdown protection)
        drawdown_multiplier = 0.5 ** min(self.consecutive_losses // 3, 3)

        final_stake = min(kelly_stake, var_stake, max_risk_stake, daily_limit_stake)
        final_stake *= drawdown_multiplier

        return max(0.5, final_stake)  # Minimum stake

    def should_trade(self) -> bool:
        """Check if we should continue trading based on daily limits"""
        return abs(self.daily_pnl) < self.max_daily_loss


class EnhancedCricketScalper:
    """Enhanced cricket scalper with realistic market modeling and risk management"""

    def __init__(self,
                 start_odds: float = 1.50,
                 tick: float = 0.01,
                 tp_ticks: int = 1,
                 sl_ticks: int = 2,
                 wicket_cooldown_balls: int = 6,
                 bankroll: float = 1000.0):

        self.start_odds = start_odds
        self.tick = tick
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.wicket_cooldown_balls = wicket_cooldown_balls

        # Enhanced components
        self.risk_manager = EnhancedRiskManager(bankroll=bankroll)
        self.match_context = MatchContext()

        self.reset()

    def reset(self):
        """Reset scalper state for new simulation"""
        self.odds = self.start_odds
        self.bid_ask_spread = 0.01  # Realistic spread
        self.position = None
        self.cooldown_balls_left = 0
        self.rows = []
        self.trades = []
        self.market_volatility = 1.0

    def _get_bid_ask(self) -> Tuple[float, float]:
        """Simulate realistic bid-ask spread that widens with volatility"""
        mid_price = self.odds
        half_spread = self.bid_ask_spread / 2

        # Spread widens with volatility
        spread_multiplier = 1 + (self.market_volatility - 1) * 0.5
        adjusted_spread = half_spread * spread_multiplier

        bid = mid_price - adjusted_spread
        ask = mid_price + adjusted_spread

        return round(bid, 2), round(ask, 2)

    def _move_odds_enhanced(self, runs: int, wicket: bool, boundary: bool):
        """Enhanced odds movement with context awareness and volatility modeling"""
        base_move = 0.0
        volatility_increase = 0.0

        if wicket:
            # Context-aware wicket impact
            base_move = 0.015
            if self.match_context.wickets_remaining <= 3:
                base_move = 0.025 + (0.01 * (4 - self.match_context.wickets_remaining))

            # Death overs multiplier
            if self.match_context.balls_remaining <= 30:  # Last 5 overs
                base_move *= 1.4

            volatility_increase = 0.3

        elif boundary:
            # Boundary impact based on match situation
            if self.match_context.current_run_rate > 0:
                pressure_factor = max(0.5, min(2.0,
                                               self.match_context.required_run_rate / self.match_context.current_run_rate))
            else:
                pressure_factor = 1.0

            if runs == 6:
                base_move = -0.015 * pressure_factor
            else:  # runs == 4
                base_move = -0.010 * pressure_factor

            volatility_increase = 0.1

        else:
            # Regular deliveries
            if runs == 0:  # Dot ball increases pressure
                pressure = self.match_context.required_run_rate - self.match_context.current_run_rate
                base_move = 0.002 * max(0.5, pressure / 2)
            elif runs in [1, 2, 3]:
                base_move = -0.001 * runs

        # Add market noise and mean reversion
        random_component = random.normalvariate(0, 0.002 * self.market_volatility)
        mean_reversion = (self.start_odds - self.odds) * 0.001

        total_move = base_move + random_component + mean_reversion

        # Update odds and volatility
        self.odds = float(np.clip(self.odds + total_move, 1.01, 10.0))
        self.market_volatility = max(0.5, min(3.0,
                                              self.market_volatility + volatility_increase - 0.05))  # Volatility decay

    def _calculate_slippage(self, is_entry: bool, stake: float) -> float:
        """Calculate realistic slippage based on stake size and market depth"""
        market_depth = 1000.0  # Assume £1000 market depth at best price
        impact = min(0.5, stake / market_depth)  # Max 0.5% impact

        # Entry slippage slightly higher than exit
        multiplier = 1.2 if is_entry else 1.0
        return impact * self.tick * multiplier

    def on_ball(self, r: pd.Series):
        """Process each ball event and update market state"""
        ts = r["timestamp"]
        runs = int(r["runs"])
        wicket = bool(r["wicket"])
        boundary = bool(r["boundary"])

        # Update match context
        self.match_context.current_over = r.get("over", 0)
        self.match_context.balls_remaining = max(0, self.match_context.balls_remaining - 1)
        if wicket:
            self.match_context.wickets_remaining -= 1
        self.match_context.current_score += runs

        # Market reaction to ball outcome
        self._move_odds_enhanced(runs, wicket, boundary)
        bid, ask = self._get_bid_ask()

        # Wicket cooldown management
        if wicket:
            self.cooldown_balls_left = max(self.cooldown_balls_left, self.wicket_cooldown_balls)

        # Position management (exit logic)
        if self.position is not None:
            entry_price = self.position["entry"]
            tp = self.position["tp"]
            sl = self.position["sl"]
            stake = self.position["stake"]

            # Check exit conditions using ask price (realistic execution)
            exit_reason = None
            exit_price = None

            if bid >= tp:  # LAY TP: price goes up, we profit
                exit_reason = "TP"
                exit_price = tp - self._calculate_slippage(False, stake)
            elif bid <= sl:  # LAY SL: price goes down, we lose
                exit_reason = "SL"
                exit_price = sl - self._calculate_slippage(False, stake)

            if exit_reason:
                # Calculate P&L with realistic tick values
                ticks_moved = round((exit_price - entry_price) / self.tick)  # Reverse for LAY
                tick_value = stake * (self.tick / (entry_price * max(entry_price - self.tick, 1e-6)))
                pnl = ticks_moved * tick_value

                # Update risk manager
                win = pnl > 0
                self.risk_manager.update_performance(pnl, win)

                # Record trade
                self.trades.append({
                    "entry_time": self.position["entry_time"],
                    "exit_time": ts,
                    "entry_back": entry_price,
                    "exit_lay": exit_price,
                    "stake": stake,
                    "ticks": ticks_moved,
                    "pnl": pnl,
                    "reason": exit_reason,
                    "ball": f"O{r.get('over', 0)}.{r.get('ball_idx', 0)}",
                    "raw": r.get("raw", "")
                })
                self.position = None

        # Entry logic: boundary trigger + risk checks
        if (self.position is None and
                self.cooldown_balls_left == 0 and
                boundary and
                self.risk_manager.should_trade()):

            # Calculate optimal stake using Kelly + risk management
            try:
                stake = self.risk_manager.get_position_size(
                    bid, self.tp_ticks, self.sl_ticks,
                    self.tick / (bid * max(bid - self.tick, 1e-6))
                )
            except:
                stake = 0.5  # Fallback minimum stake

            # Entry with realistic slippage (LAY position after boundary)
            entry_price = ask - self._calculate_slippage(True, stake)
            tp = round(entry_price + self.tp_ticks * self.tick, 2)  # LAY position - profit when price goes UP
            sl = round(entry_price - self.sl_ticks * self.tick, 2)  # LAY position - loss when price goes DOWN

            self.position = {
                "entry": entry_price,
                "tp": tp,
                "sl": sl,
                "stake": stake,
                "entry_time": ts
            }

        # Update cooldown counter
        if self.cooldown_balls_left > 0:
            self.cooldown_balls_left -= 1

        # Log market state for analysis
        self.rows.append({
            "timestamp": ts,
            "back": bid,
            "lay": ask,
            "mid": self.odds,
            "spread": ask - bid,
            "volatility": self.market_volatility,
            "runs": runs,
            "wicket": int(wicket),
            "boundary": int(boundary),
            "cooldown": self.cooldown_balls_left,
            "pos": 1 if self.position is not None else 0,
            "daily_pnl": self.risk_manager.daily_pnl
        })

    def run(self, tdf: pd.DataFrame):
        """Run simulation on timeline data"""
        for _, r in tdf.iterrows():
            self.on_ball(r)

        # Close any remaining position at end of day
        if self.position is not None and len(self.rows) > 0:
            last_row = self.rows[-1]
            entry_price = self.position["entry"]
            exit_price = last_row["lay"]
            stake = self.position["stake"]

            ticks_moved = round((entry_price - exit_price) / self.tick)
            tick_value = stake * (self.tick / (entry_price * max(entry_price - self.tick, 1e-6)))
            pnl = ticks_moved * tick_value

            self.risk_manager.update_performance(pnl, pnl > 0)

            self.trades.append({
                "entry_time": self.position["entry_time"],
                "exit_time": last_row["timestamp"],
                "entry_back": entry_price,
                "exit_lay": exit_price,
                "stake": stake,
                "ticks": ticks_moved,
                "pnl": pnl,
                "reason": "EOD",
                "ball": "EOD",
                "raw": ""
            })
            self.position = None

        return pd.DataFrame(self.rows), pd.DataFrame(self.trades)


# ---------------- Helpers (keeping your improvements) ----------------

def build_timeline_from_sheet(df: pd.DataFrame, ball_seconds: int = 4):
    """Build ball-by-ball timeline from CSV data (your version preserved)"""
    balls_cols = [c for c in df.columns if str(c).isdigit()]
    ts = datetime.now().replace(microsecond=0)
    timeline = []

    for _, row in df.iterrows():
        # Handle over number safely
        over = int(row.get("Over", 0)) if str(row.get("Over", "")).isdigit() else 0
        team = row.get("Team", "Team")

        for i, col in enumerate(balls_cols, start=1):
            val = str(row.get(col, "")).strip().lower()
            if not val:
                ts += pd.Timedelta(seconds=ball_seconds)
                continue

            # Parse ball outcome
            wicket = ("w" in val and not val.startswith("wd"))
            runs = 0
            try:
                runs = int(val[0]) if val[0].isdigit() else 0
            except:
                runs = 0
            boundary = runs in (4, 6)

            timeline.append({
                "timestamp": ts,
                "team": team,
                "over": over,
                "ball_idx": i,
                "raw": val,
                "runs": runs,
                "wicket": int(wicket),
                "boundary": int(boundary)
            })
            ts += pd.Timedelta(seconds=ball_seconds)

    return pd.DataFrame(timeline)


# ---------------- Main runner (preserving your logging) ----------------

def main():
    ap = argparse.ArgumentParser(description="Enhanced Cricket Scalper with realistic market modeling")
    ap.add_argument("--csv", required=True, help="Input CSV timeline")
    ap.add_argument("--out", default="out_live", help="Output folder")
    ap.add_argument("--tp", type=int, default=2, help="Take profit in ticks")
    ap.add_argument("--sl", type=int, default=3, help="Stop loss in ticks")
    ap.add_argument("--cooldown", type=int, default=10, help="Cooldown balls after wicket")
    ap.add_argument("--start-odds", type=float, default=1.50, help="Starting odds")
    ap.add_argument("--tick", type=float, default=0.01, help="Minimum price movement")
    ap.add_argument("--ball-secs", type=int, default=4, help="Seconds between balls")
    ap.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    args = ap.parse_args()

    # Your archival logging (preserved)
    archived = archive_input_csv(args.csv)
    print(f"🗂️ archived input → {archived}")

    # Load and parse CSV
    df = pd.read_csv(args.csv)
    if df.shape[1] <= 2:
        df = pd.read_csv(args.csv, sep=";")

    print(f"📄 CSV parsed: {df.shape[0]} rows × {df.shape[1]} cols")

    # Build timeline
    timeline = build_timeline_from_sheet(df, args.ball_secs)

    # Count key events for logging
    boundaries = timeline['boundary'].sum()
    wickets = timeline['wicket'].sum()
    print(f"🏏 boundaries: {boundaries} | wickets: {wickets}")

    # Run simulation
    scalper = EnhancedCricketScalper(
        start_odds=args.start_odds,
        tick=args.tick,
        tp_ticks=args.tp,
        sl_ticks=args.sl,
        wicket_cooldown_balls=args.cooldown,
        bankroll=args.bankroll
    )

    rows_df, trades_df = scalper.run(timeline)

    # Save results
    os.makedirs(args.out, exist_ok=True)
    rows_df.to_csv(os.path.join(args.out, "events_timeline.csv"), index=False)
    trades_df.to_csv(os.path.join(args.out, "trade_log.csv"), index=False)

    # Generate summary
    wins = (trades_df['pnl'] > 0).sum() if not trades_df.empty else 0
    losses = (trades_df['pnl'] <= 0).sum() if not trades_df.empty else 0
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"trades={len(trades_df)}\n")
        f.write(f"wins={wins}\n")
        f.write(f"losses={losses}\n")
        f.write(f"total_pnl={total_pnl:.4f}\n")

    print(f"✅ Done. Results written to {args.out}/")


if __name__ == "__main__":
    main()