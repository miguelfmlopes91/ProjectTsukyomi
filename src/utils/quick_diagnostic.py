#!/usr/bin/env python3
"""
Quick diagnostic for cricket trading data and results
"""

import pandas as pd
import numpy as np
import argparse


def diagnose_data(csv_path: str, results_dir: str = None):
    """Diagnose issues with cricket data and trading results"""

    print("🔍 CRICKET DATA DIAGNOSTIC")
    print("=" * 50)

    # Load raw CSV
    df = pd.read_csv(csv_path)
    if df.shape[1] <= 2:
        df = pd.read_csv(csv_path, sep=";")

    print(f"📊 Raw data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Teams: {df['Team'].unique() if 'Team' in df.columns else 'No Team column'}")

    # Check ball columns
    ball_cols = [c for c in df.columns if str(c).isdigit()]
    print(f"Ball columns: {len(ball_cols)} ({ball_cols})")

    # Analyze ball content
    all_balls = []
    boundaries = 0
    wickets = 0

    for col in ball_cols:
        for val in df[col].dropna():
            val_str = str(val).strip().lower()
            if val_str and val_str != 'nan':
                all_balls.append(val_str)

                # Count boundaries
                try:
                    runs = int(val_str[0]) if val_str[0].isdigit() else 0
                    if runs in [4, 6]:
                        boundaries += 1
                except:
                    pass

                # Count wickets
                if 'w' in val_str and not val_str.startswith('wd'):
                    wickets += 1

    print(f"📈 Event Analysis:")
    print(f"  Total balls: {len(all_balls)}")
    print(f"  Boundaries (4s & 6s): {boundaries}")
    print(f"  Wickets: {wickets}")
    print(f"  Boundary rate: {boundaries / len(all_balls) * 100:.1f}% of balls")

    # Show sample ball values
    unique_balls = list(set(all_balls))[:20]
    print(f"  Sample ball values: {unique_balls}")

    # Check if this looks like a real match
    if boundaries < 5:
        print("⚠️  WARNING: Very few boundaries detected!")
        print("   This might be a low-scoring or incomplete game")

    if len(all_balls) < 100:
        print("⚠️  WARNING: Very few balls in dataset!")
        print("   This might be an incomplete match")

    # Analyze trading results if provided
    if results_dir:
        try:
            trades = pd.read_csv(f"{results_dir}/trade_log.csv")
            events = pd.read_csv(f"{results_dir}/events_timeline.csv")

            print(f"\n🎯 TRADING RESULTS DIAGNOSTIC")
            print("=" * 30)
            print(f"Trades generated: {len(trades)}")

            if len(trades) > 0:
                print(f"Entry conditions:")

                # Check entry triggers
                boundary_entries = sum('4' in str(raw) or '6' in str(raw)
                                       for raw in trades['raw'] if pd.notna(raw))
                print(f"  Boundary-triggered entries: {boundary_entries}/{len(trades)}")

                # Check exit reasons
                exit_reasons = trades['reason'].value_counts()
                print(f"  Exit reasons: {dict(exit_reasons)}")

                # Check tick movements
                if 'ticks' in trades.columns:
                    avg_ticks = trades['ticks'].mean()
                    tick_range = [trades['ticks'].min(), trades['ticks'].max()]
                    print(f"  Average tick movement: {avg_ticks:.2f}")
                    print(f"  Tick range: {tick_range}")

                    # This is key - if all ticks are negative, we're betting wrong direction
                    if trades['ticks'].max() <= 0:
                        print("🚨 CRITICAL: All tick movements are negative!")
                        print("   This means price always moves AGAINST our position")
                        print("   → Strategy assumes wrong price direction")

                # Check price movements
                if not events.empty and 'back' in events.columns:
                    price_start = events['back'].iloc[0]
                    price_end = events['back'].iloc[-1]
                    price_change = (price_end - price_start) / price_start * 100
                    print(f"  Overall price movement: {price_change:+.2f}%")

                    # Check volatility during match
                    price_volatility = events['back'].std()
                    print(f"  Price volatility: {price_volatility:.4f}")

            else:
                print("⚠️  No trades generated!")
                print("  Possible issues:")
                print("  - No boundary events detected")
                print("  - Risk management blocking all trades")
                print("  - Cooldown too long")

                # Check boundary detection in events
                if not events.empty:
                    boundary_events = events[events['boundary'] == 1]
                    print(f"  Boundary events in timeline: {len(boundary_events)}")

        except Exception as e:
            print(f"⚠️  Could not analyze results: {e}")

    print(f"\n💡 RECOMMENDATIONS:")

    if boundaries < 10:
        print("1. Get more data - this match has too few trading opportunities")

    if results_dir:
        try:
            trades = pd.read_csv(f"{results_dir}/trade_log.csv")
            if len(trades) > 0 and 'ticks' in trades.columns:
                if trades['ticks'].mean() < -2:
                    print("2. 🚨 FLIP THE STRATEGY DIRECTION!")
                    print("   Current: Back after boundary (expecting price down)")
                    print("   Try: Lay after boundary (expecting price up)")
                    print("   OR: Reverse TP/SL logic")
        except:
            pass

    print("3. Try with a different game - this might be an unusual match")
    print("4. Check if Pakistan-Bangladesh had any special circumstances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV file to diagnose")
    parser.add_argument("--results", help="Results directory to analyze")

    args = parser.parse_args()

    diagnose_data(args.csv, args.results)


if __name__ == "__main__":
    main()