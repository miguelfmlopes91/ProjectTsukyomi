
#!/usr/bin/env python3
"""
Grid search tuner for cricket_scalper_sim.py parameters.
Run it against a single CSV and try several combos for TP/SL/cooldown.
Writes a small leaderboard CSV with PnL per combo.

Usage:
  python tune_params.py --csv "teste - Timeline.csv" --out outdir
"""

import argparse
import itertools
import os
import subprocess
import sys
import pandas as pd

def run_once(csv, out, tp, sl, cooldown):
    combo_out = os.path.join(out, f"tp{tp}_sl{sl}_cd{cooldown}")
    os.makedirs(combo_out, exist_ok=True)
    cmd = [
        sys.executable, "cricket_scalper_sim.py",
        "--csv", csv,
        "--out", combo_out,
        "--tp", str(tp),
        "--sl", str(sl),
        "--cooldown", str(cooldown)
    ]
    subprocess.run(cmd, check=True)
    # read summary
    summary_file = os.path.join(combo_out, "summary.txt")
    total_pnl = 0.0
    trades = wins = losses = eod = 0
    with open(summary_file, "r") as f:
        for line in f:
            k, v = line.strip().split("=", 1)
            if k == "total_pnl":
                total_pnl = float(v)
            elif k == "trades":
                trades = int(v)
            elif k == "wins":
                wins = int(v)
            elif k == "losses":
                losses = int(v)
            elif k == "eod":
                eod = int(v)
    return {"tp": tp, "sl": sl, "cooldown": cooldown, "trades": trades, "wins": wins, "losses": losses, "eod": eod, "total_pnl": total_pnl}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="tune_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # small grid
    tps = [1, 2]
    sls = [1, 2, 3]
    cds = [4, 6, 8, 10]

    rows = []
    for tp, sl, cd in itertools.product(tps, sls, cds):
        print(f"Trying tp={tp} sl={sl} cooldown={cd}...")
        res = run_once(args.csv, args.out, tp, sl, cd)
        rows.append(res)

    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    leaderboard = os.path.join(args.out, "leaderboard.csv")
    df.to_csv(leaderboard, index=False)
    print("✅ Tuning done. See", leaderboard)

if __name__ == "__main__":
    main()
