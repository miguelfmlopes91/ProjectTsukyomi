#!/usr/bin/env python3
"""
Cricket Scalper Simulator (event-driven, backtest on CSV)

Usage:
  python cricket_scalper_sim.py --csv "teste - Timeline.csv" --out outdir \
      --start-odds 1.50 --tick 0.01 --tp 1 --sl 2 --stake 2.0 --cooldown 6 --ball-secs 4 [--quiet]

Input CSV schema (columns may be ";" or "," separated):
  Team, Over, 1,2,3,4,5,6,7,8, Total, W, BOUN., Dif., Dif.2

The script:
  - builds a ball-by-ball timeline (detects wicket 'w' and boundaries 4/6);
  - simulates odds reaction and a simple scalping strategy (Back->Lay on boundary);
  - writes a trade log CSV + an odds plot + summary.txt in --out.

Flags:
  --quiet  -> suprime prints finais (✅ Done / Artifacts). Útil para uso via watcher.
"""

import argparse
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# I/O helpers
# ----------------------------
def load_sheet(path: str) -> pd.DataFrame:
    """Load CSV trying comma first, then semicolon."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] <= 2:
            raise ValueError("Few columns with comma, try semicolon")
        return df
    except Exception:
        df = pd.read_csv(path, sep=";")
        return df


def parse_ball_value(v: str):
    """Return (runs, wicket_flag, boundary_flag) from a cell like '4', '6', 'w', 'w1', '1wd', '1n', ''."""
    if pd.isna(v):
        return 0, False, False
    s = str(v).strip().lower()
    if s == "" or s == "nan":
        return 0, False, False

    wicket = False
    boundary = False
    runs = 0

    # wicket markers (ignore wide 'wd' prefix)
    if "w" in s and not s.startswith("wd"):
        wicket = True
        s = s.replace("w", "")

    # sum any digits present (handles '1n', '2wd', 'w1', etc.)
    m = re.findall(r"\d+", s)
    if m:
        runs = sum(int(x) for x in m)

    if runs in (4, 6):
        boundary = True

    return runs, wicket, boundary


def build_timeline_from_sheet(df: pd.DataFrame, start_time: datetime = None, ball_seconds: int = 4) -> pd.DataFrame:
    """Expand the sheet into a ball-by-ball timeline with timestamps spaced by ball_seconds."""
    balls_cols = [c for c in df.columns if str(c).isdigit()]
    start_time = start_time or datetime.now().replace(microsecond=0)
    timeline = []
    ts = start_time

    for _, row in df.iterrows():
        try:
            over = int(row.get("Over", 0))
        except Exception:
            over = 0
        team = row.get("Team", "Team")
        for i, col in enumerate(balls_cols, start=1):
            val = row.get(col, "")
            runs, wicket, boundary = parse_ball_value(val)
            # se célula vazia, avança no tempo mas não cria evento
            if (val is None) or (str(val).strip() == ""):
                ts += timedelta(seconds=ball_seconds)
                continue
            timeline.append({
                "timestamp": ts,
                "team": team,
                "over": over,
                "ball_idx": i,
                "raw": str(val),
                "runs": runs,
                "wicket": int(wicket),
                "boundary": int(boundary)
            })
            ts += timedelta(seconds=ball_seconds)

    tdf = pd.DataFrame(timeline)
    if not tdf.empty:
        tdf = tdf.sort_values("timestamp").reset_index(drop=True)
    return tdf


# ----------------------------
# Strategy
# ----------------------------
class CricketScalper:
    def __init__(
        self,
        start_odds: float = 1.50,
        tick: float = 0.01,
        tp_ticks: int = 1,
        sl_ticks: int = 2,
        wicket_cooldown_balls: int = 6,
        stake: float = 2.0,
    ):
        self.start_odds = start_odds
        self.tick = tick
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.wicket_cooldown_balls = wicket_cooldown_balls
        self.stake = stake
        self.reset()

    def reset(self):
        self.odds = self.start_odds
        self.position = None
        self.cooldown_balls_left = 0
        self.rows = []
        self.trades = []

    @staticmethod
    def _clamp(x: float) -> float:
        return float(np.clip(x, 1.05, 10.0))

    def _move_odds(self, runs: int, wicket: bool, boundary: bool):
        # Heurística simples e tunável
        if wicket:
            self.odds = self._clamp(self.odds + 0.02)
        elif boundary:
            self.odds = self._clamp(self.odds - 0.012)
        else:
            if runs == 0:
                self.odds = self._clamp(self.odds + 0.002)  # dot ball
            elif runs == 1:
                self.odds = self._clamp(self.odds - 0.002)
            elif runs == 2:
                self.odds = self._clamp(self.odds - 0.003)
            elif runs == 3:
                self.odds = self._clamp(self.odds - 0.004)
            else:
                self.odds = self._clamp(self.odds - 0.005)

    def _pnl_per_tick(self, entry: float) -> float:
        # aproximação para impacto por tick relativa ao preço de entrada
        return self.stake * (self.tick / (entry * max(entry - self.tick, 1e-6)))

    def on_ball(self, r: pd.Series):
        ts = r["timestamp"]
        runs = int(r["runs"])
        wicket = bool(r["wicket"])
        boundary = bool(r["boundary"])

        # 1) mercado reage
        self._move_odds(runs, wicket, boundary)
        back = round(self.odds, 2)
        lay = round(self.odds + self.tick, 2)

        # 2) cooldown por wicket
        if wicket:
            self.cooldown_balls_left = max(self.cooldown_balls_left, self.wicket_cooldown_balls)

        # 3) gerir posição aberta
        if self.position is not None:
            entry = self.position["entry"]
            tp = self.position["tp"]
            sl = self.position["sl"]
            exit_reason = None
            exit_odds = None
            if lay <= tp:
                exit_reason = "TP"
                exit_odds = lay
            elif lay >= sl:
                exit_reason = "SL"
                exit_odds = lay
            if exit_reason:
                ticks_moved = int(round((entry - exit_odds) / self.tick))
                pnl = ticks_moved * self._pnl_per_tick(entry)
                self.trades.append({
                    "entry_time": self.position["entry_time"],
                    "exit_time": ts,
                    "entry_back": entry,
                    "exit_lay": exit_odds,
                    "ticks": ticks_moved,
                    "pnl": pnl,
                    "reason": exit_reason,
                    "ball": f"O{r['over']}.{r['ball_idx']}",
                    "raw": r["raw"]
                })
                self.position = None

        # 4) entrada: boundary + sem cooldown + flat
        if self.position is None and self.cooldown_balls_left == 0 and boundary:
            entry_back = back
            tp = round(entry_back - self.tp_ticks * self.tick, 2)
            sl = round(entry_back + self.sl_ticks * self.tick, 2)
            self.position = {"entry": entry_back, "tp": tp, "sl": sl, "entry_time": ts}

        # 5) avança cooldown
        if self.cooldown_balls_left > 0:
            self.cooldown_balls_left -= 1

        # 6) logging de série temporal
        self.rows.append({
            "timestamp": ts,
            "back": back,
            "lay": lay,
            "runs": runs,
            "wicket": int(wicket),
            "boundary": int(boundary),
            "cooldown": self.cooldown_balls_left,
            "pos": 1 if self.position is not None else 0
        })

    def run(self, tdf: pd.DataFrame):
        for _, r in tdf.iterrows():
            self.on_ball(r)

        # fecha posição no EOD (se existir)
        if self.position is not None and len(self.rows) > 0:
            last = self.rows[-1]
            entry = self.position["entry"]
            exit_lay = last["lay"]
            ticks_moved = int(round((entry - exit_lay) / self.tick))
            pnl = ticks_moved * self._pnl_per_tick(entry)
            self.trades.append({
                "entry_time": self.position["entry_time"],
                "exit_time": last["timestamp"],
                "entry_back": entry,
                "exit_lay": exit_lay,
                "ticks": ticks_moved,
                "pnl": pnl,
                "reason": "EOD",
                "ball": "EOD",
                "raw": ""
            })
            self.position = None

        return pd.DataFrame(self.rows), pd.DataFrame(self.trades)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to your exported Google Sheet CSV")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--start-odds", type=float, default=1.50)
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--tp", type=int, default=1, help="TP in ticks")
    ap.add_argument("--sl", type=int, default=2, help="SL in ticks")
    ap.add_argument("--stake", type=float, default=2.0)
    ap.add_argument("--cooldown", type=int, default=6, help="balls to skip after a wicket")
    ap.add_argument("--ball-secs", type=int, default=4, help="seconds to advance per ball (timeline spacing)")
    ap.add_argument("--quiet", action="store_true", help="Suppress final prints (Done/Artifacts)")
    args = ap.parse_args()

    def log(msg=""):
        if not args.quiet:
            print(msg)

    os.makedirs(args.out, exist_ok=True)

    # Load & normalize
    df_raw = load_sheet(args.csv)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    if "Team" not in df_raw.columns or "Over" not in df_raw.columns:
        rename = {}
        for c in df_raw.columns:
            cl = c.lower().strip()
            if cl == "team":
                rename[c] = "Team"
            elif cl == "over":
                rename[c] = "Over"
        if rename:
            df_raw = df_raw.rename(columns=rename)

    ball_cols = [c for c in df_raw.columns if str(c).isdigit()]
    keep_cols = ["Team", "Over"] + ball_cols
    df = df_raw[keep_cols].copy()

    # Build timeline
    timeline = build_timeline_from_sheet(df, ball_seconds=args.ball_secs)

    # Simulate
    scalper = CricketScalper(
        start_odds=args.start_odds,
        tick=args.tick,
        tp_ticks=args.tp,
        sl_ticks=args.sl,
        wicket_cooldown_balls=args.cooldown,
        stake=args.stake,
    )
    rows_df, trades_df = scalper.run(timeline)

    # Save artifacts
    events_csv = os.path.join(args.out, "events_timeline.csv")
    trades_csv = os.path.join(args.out, "trade_log.csv")
    plot_png = os.path.join(args.out, "odds_plot.png")
    summary_txt = os.path.join(args.out, "summary.txt")

    rows_df.to_csv(events_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    # Plot
    if not rows_df.empty:
        plt.figure(figsize=(11, 4))
        plt.plot(rows_df["timestamp"], rows_df["back"], label="Back")
        plt.plot(rows_df["timestamp"], rows_df["lay"], label="Lay")
        if not trades_df.empty:
            plt.scatter(trades_df["entry_time"], trades_df["entry_back"], marker="^", label="Entries")
            plt.scatter(trades_df["exit_time"], trades_df["exit_lay"], marker="v", label="Exits")
        plt.title("Cricket scalping – event-driven backtest")
        plt.xlabel("Time")
        plt.ylabel("Odds")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_png)
        plt.close()
    else:
        # cria imagem vazia para não falhar links a artefactos
        plt.figure(figsize=(11, 2))
        plt.title("No data to plot yet")
        plt.tight_layout()
        plt.savefig(plot_png)
        plt.close()

    # Summary
    wins = int((trades_df["reason"] == "TP").sum()) if not trades_df.empty else 0
    losses = int((trades_df["reason"] == "SL").sum()) if not trades_df.empty else 0
    eod = int((trades_df["reason"] == "EOD").sum()) if not trades_df.empty else 0
    total_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0

    with open(summary_txt, "w") as f:
        f.write(f"trades={len(trades_df)}\n")
        f.write(f"wins={wins}\n")
        f.write(f"losses={losses}\n")
        f.write(f"eod={eod}\n")
        f.write(f"total_pnl={total_pnl:.4f}\n")

    # Final prints (silenciáveis)
    log("✅ Done")
    log("Artifacts:")
    log(f" - {events_csv}")
    log(f" - {trades_csv}")
    log(f" - {plot_png}")
    log(f" - {summary_txt}")


if __name__ == "__main__":
    main()
