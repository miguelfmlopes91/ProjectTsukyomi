#!/usr/bin/env python3
"""
Cricket Trading Results Analyzer
Comprehensive analysis tool for trading simulation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
import json


class CricketTradeAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.load_data()

    def load_data(self):
        """Load all result files"""
        try:
            self.trades = pd.read_csv(self.results_dir / "trade_log.csv")
            self.events = pd.read_csv(self.results_dir / "events_timeline.csv")

            # Convert timestamps
            if not self.trades.empty:
                self.trades['entry_time'] = pd.to_datetime(self.trades['entry_time'])
                self.trades['exit_time'] = pd.to_datetime(self.trades['exit_time'])

            if not self.events.empty:
                self.events['timestamp'] = pd.to_datetime(self.events['timestamp'])

            print(f"✅ Loaded {len(self.trades)} trades and {len(self.events)} events")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.trades = pd.DataFrame()
            self.events = pd.DataFrame()

    def basic_stats(self):
        """Print basic trading statistics"""
        if self.trades.empty:
            print("⚠️  No trades found")
            return

        print("\n" + "=" * 50)
        print("📊 BASIC TRADING STATISTICS")
        print("=" * 50)

        total_trades = len(self.trades)
        wins = (self.trades['pnl'] > 0).sum()
        losses = (self.trades['pnl'] <= 0).sum()
        total_pnl = self.trades['pnl'].sum()

        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = self.trades[self.trades['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins} ({win_rate:.1f}%)")
        print(f"Losses: {losses} ({100 - win_rate:.1f}%)")
        print(f"Total P&L: £{total_pnl:.4f}")
        print(f"Average Win: £{avg_win:.4f}")
        print(f"Average Loss: £{avg_loss:.4f}")

        if avg_loss != 0:
            profit_factor = abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 else float('inf')
            print(f"Profit Factor: {profit_factor:.2f}")

        # Risk metrics
        if total_trades > 1:
            returns = self.trades['pnl'].values
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_dd = self.calculate_max_drawdown()
            print(f"Sharpe Ratio (annualized): {sharpe:.2f}")
            print(f"Max Drawdown: £{max_dd:.4f}")

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if self.trades.empty:
            return 0

        cumulative_pnl = self.trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        return abs(drawdown.min())

    def trade_analysis(self):
        """Detailed trade-by-trade analysis"""
        if self.trades.empty:
            return

        print("\n" + "=" * 50)
        print("🔍 TRADE ANALYSIS")
        print("=" * 50)

        # Exit reason breakdown
        exit_reasons = self.trades['reason'].value_counts()
        print("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            percentage = count / len(self.trades) * 100
            avg_pnl = self.trades[self.trades['reason'] == reason]['pnl'].mean()
            print(f"  {reason}: {count} trades ({percentage:.1f}%) - Avg P&L: £{avg_pnl:.4f}")

        # Tick analysis
        if 'ticks' in self.trades.columns:
            print(f"\nTick Movement Analysis:")
            print(f"  Average ticks per trade: {self.trades['ticks'].mean():.2f}")
            print(f"  Tick range: {self.trades['ticks'].min()} to {self.trades['ticks'].max()}")

            # Ticks vs PnL correlation
            corr = self.trades['ticks'].corr(self.trades['pnl'])
            print(f"  Ticks-PnL correlation: {corr:.3f}")

        # Entry/Exit price analysis
        print(f"\nPrice Analysis:")
        print(f"  Entry prices range: {self.trades['entry_back'].min():.2f} - {self.trades['entry_back'].max():.2f}")
        print(f"  Average entry price: {self.trades['entry_back'].mean():.2f}")

        # Stake analysis
        if 'stake' in self.trades.columns:
            print(f"  Stake range: £{self.trades['stake'].min():.2f} - £{self.trades['stake'].max():.2f}")
            print(f"  Average stake: £{self.trades['stake'].mean():.2f}")

    def timing_analysis(self):
        """Analyze timing patterns"""
        if self.trades.empty:
            return

        print("\n" + "=" * 50)
        print("⏰ TIMING ANALYSIS")
        print("=" * 50)

        # Trade duration
        self.trades['duration_seconds'] = (self.trades['exit_time'] - self.trades['entry_time']).dt.total_seconds()
        avg_duration = self.trades['duration_seconds'].mean()
        print(f"Average trade duration: {avg_duration:.1f} seconds ({avg_duration / 60:.1f} minutes)")

        # Trade timing distribution
        self.trades['entry_minute'] = self.trades['entry_time'].dt.minute
        entry_dist = self.trades['entry_minute'].value_counts().sort_index()
        print(f"Most active minute: {entry_dist.idxmax()}min ({entry_dist.max()} trades)")

        # Ball information analysis
        if 'ball' in self.trades.columns:
            print(f"\nBall Analysis:")
            ball_stats = self.trades['ball'].value_counts()
            print(f"  Most common entry ball: {ball_stats.index[0]} ({ball_stats.iloc[0]} times)")

    def market_analysis(self):
        """Analyze market conditions during trades"""
        if self.events.empty:
            return

        print("\n" + "=" * 50)
        print("📈 MARKET ANALYSIS")
        print("=" * 50)

        # Volatility analysis
        if 'volatility' in self.events.columns:
            avg_volatility = self.events['volatility'].mean()
            print(f"Average market volatility: {avg_volatility:.3f}")

            # Volatility during trades
            if not self.trades.empty:
                trade_volatility = []
                for _, trade in self.trades.iterrows():
                    trade_events = self.events[
                        (self.events['timestamp'] >= trade['entry_time']) &
                        (self.events['timestamp'] <= trade['exit_time'])
                        ]
                    if not trade_events.empty:
                        trade_volatility.append(trade_events['volatility'].mean())

                if trade_volatility:
                    print(f"Average volatility during trades: {np.mean(trade_volatility):.3f}")

        # Spread analysis
        if 'spread' in self.events.columns:
            avg_spread = self.events['spread'].mean()
            print(f"Average bid-ask spread: {avg_spread:.4f}")

        # Price movement analysis
        if 'back' in self.events.columns:
            price_range = self.events['back'].max() - self.events['back'].min()
            price_volatility = self.events['back'].std()
            print(f"Price range: {price_range:.4f}")
            print(f"Price volatility (std): {price_volatility:.4f}")

    def failure_analysis(self):
        """Specific analysis for poor performance"""
        if self.trades.empty:
            return

        print("\n" + "=" * 50)
        print("🚨 FAILURE ANALYSIS")
        print("=" * 50)

        # Check if all trades are losses
        wins = (self.trades['pnl'] > 0).sum()
        if wins == 0:
            print("⚠️  CRITICAL: Zero winning trades detected!")

            # Analyze why trades are failing
            sl_trades = self.trades[self.trades['reason'] == 'SL']
            print(f"Stop Loss trades: {len(sl_trades)} ({len(sl_trades) / len(self.trades) * 100:.1f}%)")

            # Check if TP levels are too aggressive
            if 'ticks' in self.trades.columns:
                avg_ticks_against = sl_trades['ticks'].mean() if not sl_trades.empty else 0
                print(f"Average ticks moved against position: {abs(avg_ticks_against):.2f}")

            # Check entry conditions
            print("\nEntry Condition Analysis:")
            boundary_entries = self.trades[self.trades['raw'].str.contains('4|6', na=False)]
            print(f"Entries after boundaries: {len(boundary_entries)}/{len(self.trades)}")

            # Check if entries are too late/early
            if not self.events.empty:
                boundary_events = self.events[self.events['boundary'] == 1]
                print(f"Total boundary events in data: {len(boundary_events)}")
                print(
                    f"Conversion rate: {len(self.trades) / len(boundary_events) * 100:.1f}% of boundaries became trades")

    def generate_plots(self, save_dir: str = None):
        """Generate analysis plots"""
        if self.trades.empty:
            return

        save_dir = Path(save_dir) if save_dir else self.results_dir

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cricket Trading Analysis', fontsize=16)

        # 1. P&L distribution
        axes[0, 0].hist(self.trades['pnl'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('P&L Distribution')
        axes[0, 0].set_xlabel('P&L (£)')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[0, 0].legend()

        # 2. Cumulative P&L
        cumulative_pnl = self.trades['pnl'].cumsum()
        axes[0, 1].plot(cumulative_pnl, marker='o', markersize=3)
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L (£)')
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Trade duration vs P&L
        if 'duration_seconds' in self.trades.columns:
            scatter = axes[1, 0].scatter(self.trades['duration_seconds'], self.trades['pnl'],
                                         c=self.trades['pnl'], cmap='RdYlGn', alpha=0.7)
            axes[1, 0].set_title('Trade Duration vs P&L')
            axes[1, 0].set_xlabel('Duration (seconds)')
            axes[1, 0].set_ylabel('P&L (£)')
            plt.colorbar(scatter, ax=axes[1, 0])

        # 4. Exit reasons
        exit_counts = self.trades['reason'].value_counts()
        colors = ['red' if reason == 'SL' else 'green' if reason == 'TP' else 'gray'
                  for reason in exit_counts.index]
        axes[1, 1].pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
                       colors=colors)
        axes[1, 1].set_title('Exit Reasons')

        plt.tight_layout()

        plot_path = save_dir / "trading_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Analysis plot saved to: {plot_path}")

    def export_detailed_report(self, filename: str = None):
        """Export detailed analysis to text file"""
        filename = filename or str(self.results_dir / "detailed_analysis.txt")

        with open(filename, 'w') as f:
            f.write("CRICKET TRADING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Results directory: {self.results_dir}\n\n")

            # Redirect print output to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f

            self.basic_stats()
            self.trade_analysis()
            self.timing_analysis()
            self.market_analysis()
            self.failure_analysis()

            sys.stdout = old_stdout

        print(f"📋 Detailed report saved to: {filename}")

    def run_full_analysis(self, generate_plots: bool = True, export_report: bool = True):
        """Run complete analysis suite"""
        print("🔍 Running comprehensive trading analysis...")

        self.basic_stats()
        self.trade_analysis()
        self.timing_analysis()
        self.market_analysis()
        self.failure_analysis()

        if generate_plots and not self.trades.empty:
            self.generate_plots()

        if export_report:
            self.export_detailed_report()

        # Recommendations based on analysis
        self.generate_recommendations()

    def generate_recommendations(self):
        """Generate specific recommendations based on analysis"""
        print("\n" + "=" * 50)
        print("💡 RECOMMENDATIONS")
        print("=" * 50)

        if self.trades.empty:
            print("⚠️  No trades to analyze - check entry conditions")
            return

        wins = (self.trades['pnl'] > 0).sum()
        win_rate = wins / len(self.trades) * 100

        if win_rate == 0:
            print("🚨 CRITICAL ISSUES:")
            print("1. Zero win rate suggests fundamental problems:")
            print("   - Check if TP levels are achievable")
            print("   - Verify odds movement direction assumptions")
            print("   - Consider if market conditions match strategy")
            print("2. Immediate actions:")
            print("   - Test with TP=1, SL=3 (proven ratios)")
            print("   - Check if entries are happening at right time")
            print("   - Verify data quality (boundary detection)")

        elif win_rate < 30:
            print("⚠️  Low win rate - consider:")
            print("   - Reducing TP targets")
            print("   - Better entry timing")
            print("   - Adding market filters")

        # Check if this is a data problem
        if not self.events.empty:
            boundary_events = self.events[self.events['boundary'] == 1].shape[0]
            if len(self.trades) < boundary_events * 0.1:
                print("3. Possible issues:")
                print("   - Most boundaries not converting to trades")
                print("   - Check cooldown periods")
                print("   - Verify risk management not blocking trades")


def main():
    parser = argparse.ArgumentParser(description="Analyze cricket trading results")
    parser.add_argument("--dir", required=True, help="Results directory to analyze")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-report", action="store_true", help="Skip report export")

    args = parser.parse_args()

    analyzer = CricketTradeAnalyzer(args.dir)
    analyzer.run_full_analysis(
        generate_plots=not args.no_plots,
        export_report=not args.no_report
    )


if __name__ == "__main__":
    main()