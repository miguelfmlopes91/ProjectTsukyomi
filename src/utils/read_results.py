"""
This utility script/notebook collects the results of multiple simulation runs
(e.g., from `out_live/` or parameter sweeps in `tune_out/`) and displays them
in a consolidated table. It automatically scans subfolders for `summary.txt`
files, parses metrics such as number of trades, wins, losses, and total PnL,
and outputs a leaderboard view. This makes it easy to quickly compare different
simulation configurations (e.g., TP/SL/cooldown values) and identify the most
profitable or stable setups.
"""

import pandas as pd
import glob, os

def load_summaries(base_dir="out_live"):
    summaries = []
    for path in glob.glob(os.path.join(base_dir, "**/summary.txt"), recursive=True):
        run_name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            vals = dict(line.strip().split("=") for line in f if "=" in line)
        summaries.append({
            "run": run_name,
            "trades": int(vals.get("trades",0)),
            "wins": int(vals.get("wins",0)),
            "losses": int(vals.get("losses",0)),
            "total_pnl": float(vals.get("total_pnl",0))
        })
    return pd.DataFrame(summaries)

df = load_summaries("tune_out")   # ou "out_live" se só tens um
print(df.sort_values("total_pnl", ascending=False))
