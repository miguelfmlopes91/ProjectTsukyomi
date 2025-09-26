#!/usr/bin/env python3
import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from typing import Optional, Tuple, List, Dict
import csv
from io import StringIO
import re
import pandas as pd

SUMMARY = "summary.txt"
TRADE_LOG = "trade_log.csv"

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PATH = os.path.join(HERE, "cricket_scalper_sim.py")  # absolute path to sim
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---------- Helpers ----------

def resolve_csv_path(csv_arg: str) -> Optional[str]:
    """Return an existing path for the CSV, trying a couple of sensible locations."""
    candidates = [
        csv_arg,
        os.path.join(REPO_ROOT, csv_arg),
        os.path.join(REPO_ROOT, "src", csv_arg),
        os.path.join(REPO_ROOT, "src", "simulators", csv_arg),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def sha1_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def read_summary(outdir: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
    path = os.path.join(outdir, SUMMARY)
    if not os.path.exists(path):
        return None, None, None, None
    vals = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    try:
        trades = int(vals.get("trades", "0"))
        wins   = int(vals.get("wins", "0"))
        losses = int(vals.get("losses", "0"))
        pnl    = float(vals.get("total_pnl", "0"))
        return trades, wins, losses, pnl
    except Exception:
        return None, None, None, None

def read_last_trades(outdir: str, n: int = 3) -> str:
    path = os.path.join(outdir, TRADE_LOG)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r") as f:
            lines = f.read().splitlines()
        if len(lines) <= 1:
            return ""
        hdr, body = lines[0], lines[1:]
        tail = body[-n:]
        return "\n".join([hdr] + tail)
    except Exception:
        return ""

# ---------- Pretty printing ----------

def _fmt_float(x, prec=4):
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

def _render_table(rows: List[Dict[str, str]], headers: List[str], max_width: int = 22) -> str:
    if not rows:
        return "(sem dados)"
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            cell = "" if r.get(h) is None else str(r.get(h))
            cell = cell if len(cell) <= max_width else (cell[:max_width-1] + "…")
            if len(cell) > widths[h]:
                widths[h] = len(cell)
    def pad(h, s):
        s = "" if s is None else str(s)
        s = s if len(s) <= max_width else (s[:max_width-1] + "…")
        return s.ljust(widths[h])
    top = "| " + " | ".join(pad(h, h) for h in headers) + " |"
    sep = "|-" + "-|-".join("-" * widths[h] for h in headers) + "-|"
    body = ["| " + " | ".join(pad(h, r.get(h, "")) for h in headers) + " |" for r in rows]
    return "\n".join([top, sep] + body)

def pretty_print(trades, wins, losses, pnl, delta, last_trades_text, first=False):
    now = time.strftime("%H:%M:%S")
    delta_str = "—" if (delta is None) else f"{delta:+.4f}"
    prefix = "primeira run | " if first else ""
    print(f"🕒 {now} | {prefix}Δ desde última run: {delta_str}")
    summary_rows = [
        {"metric": "trades",     "value": str(trades or 0)},
        {"metric": "wins",       "value": str(wins or 0)},
        {"metric": "losses",     "value": str(losses or 0)},
        {"metric": "eod",        "value": str(max(0, (trades or 0) - (wins or 0) - (losses or 0)))},
        {"metric": "total_pnl",  "value": _fmt_float(pnl or 0.0, prec=4)},
    ]
    print(_render_table(summary_rows, headers=["metric","value"]))
    print()
    if last_trades_text and last_trades_text.strip():
        lines = last_trades_text.splitlines()
        if len(lines) > 1:
            header = lines[0].split(",")
            rows = [dict(zip(header, l.split(","))) for l in lines[1:]]
            # only show last 3 trades
            rows = rows[-3:]
            print("Últimos trades:")
            print(_render_table(rows, headers=header, max_width=18))
            print()
        else:
            print("Últimos trades (raw):")
            print(last_trades_text)
            print()

# ---------- CSV diagnostics ----------

def load_sheet_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] <= 2:
            raise ValueError("Few columns with comma, try semicolon")
        return df
    except Exception:
        return pd.read_csv(path, sep=";")

def count_boundaries(df: pd.DataFrame) -> int:
    ball_cols = [c for c in df.columns if str(c).isdigit()]
    if not ball_cols:
        return 0
    vals = (df[ball_cols].astype(str).apply(lambda s: s.str.strip().str.lower()))
    def is_boundary(cell: str) -> bool:
        m = re.findall(r"\d+", cell)
        return any(x in ("4","6") for x in m)
    return int(vals.applymap(is_boundary).values.sum())

def count_wickets(df: pd.DataFrame) -> int:
    ball_cols = [c for c in df.columns if str(c).isdigit()]
    if not ball_cols:
        return 0
    vals = (df[ball_cols].astype(str).apply(lambda s: s.str.strip().str.lower()))
    return int(vals.applymap(lambda x: ("w" in x) and (not x.startswith("wd"))).values.sum())

# ---------- Runner to call the simulator ----------

def run_sim(csv_path: str, args, quiet=True):
    cmd = f'''python "{SIM_PATH}" \
      --csv "{csv_path}" \
      --out "{args.out}" \
      --tp {args.tp} --sl {args.sl} --cooldown {args.cooldown} \
      --start-odds {args.start_odds} --tick {args.tick} --ball-secs {args.ball_secs}'''
    if quiet:
        subprocess.run(shlex.split(cmd), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(shlex.split(cmd), check=False)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Caminho para o CSV com a timeline (pode ter espaços).")
    ap.add_argument("--out", default="out_live", help="Diretório de saída onde o sim grava artefactos.")
    ap.add_argument("--interval", type=int, default=5, help="Segundos entre checks.")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--sl", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--start-odds", type=float, default=1.50)
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--ball-secs", type=int, default=4)
    ap.add_argument("--first-run", action="store_true",
                    help="Executa simulação logo no arranque (mesmo sem alterações).")
    args = ap.parse_args()

    # Resolve CSV path (be forgiving if user passed just the filename)
    csv_path = resolve_csv_path(args.csv)

    print("👀 watcher a correr… (Ctrl+C para parar)")
    last_csv_sig = None
    last_output_sig = None
    last_pnl = None

    # ---- Run once immediately if requested (or if we can resolve the CSV) ----
    if args.first_run:
        if not csv_path:
            print(f"⏳ à espera do ficheiro '{args.csv}'… (tenta 'src/{args.csv}' se estiver em src/)")
        else:
            print(f"🚀 primeira execução com: {csv_path}")
            try:
                df_raw = load_sheet_auto(csv_path)
                print(f"📄 CSV parsed: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
                print(f"🏏 boundaries: {count_boundaries(df_raw)} | wickets: {count_wickets(df_raw)}")
            except Exception as e:
                print(f"⚠️ CSV diagnostics failed: {e}")
            run_sim(csv_path, args, quiet=False)  # let the sim print its line(s)
            trades, wins, losses, pnl = read_summary(args.out)
            last_trades = read_last_trades(args.out)
            pretty_print(trades, wins, losses, pnl, delta=None, last_trades_text=last_trades, first=True)
            last_pnl = pnl
            last_output_sig = hashlib.sha1(f"{pnl}|{sha1_file(os.path.join(args.out, TRADE_LOG))}".encode()).hexdigest() if pnl is not None else None
            last_csv_sig = sha1_file(csv_path) if csv_path else None

    # ---- Watch loop ----
    while True:
        try:
            # Re-resolve in case file appears later
            if not csv_path:
                csv_path = resolve_csv_path(args.csv)
                if not csv_path:
                    print(f"⏳ à espera do ficheiro '{args.csv}'…")
                    time.sleep(args.interval)
                    continue

            csv_sig = sha1_file(csv_path)

            if csv_sig and csv_sig != last_csv_sig:
                print(f"🕒 {time.strftime('%H:%M:%S')} | mudança detetada → a simular…")
                try:
                    df_raw = load_sheet_auto(csv_path)
                    print(f"📄 CSV parsed: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
                    print(f"🏏 boundaries: {count_boundaries(df_raw)} | wickets: {count_wickets(df_raw)}")
                except Exception as e:
                    print(f"⚠️ CSV diagnostics failed: {e}")

                run_sim(csv_path, args, quiet=True)

                trades, wins, losses, pnl = read_summary(args.out)
                out_concat = f"{pnl}|{sha1_file(os.path.join(args.out, TRADE_LOG))}"
                output_sig = hashlib.sha1(out_concat.encode()).hexdigest()

                delta = (pnl - last_pnl) if (pnl is not None and last_pnl is not None) else None
                if delta is not None and abs(delta) < 1e-6 and output_sig == last_output_sig:
                    print(f"🕒 {time.strftime('%H:%M:%S')} | CSV updated, nothing mudou nos resultados")
                else:
                    last_trades = read_last_trades(args.out)
                    pretty_print(trades, wins, losses, pnl, delta, last_trades)

                last_pnl = pnl
                last_output_sig = output_sig
                last_csv_sig = csv_sig

            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 watcher terminado.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ erro: {e}")
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
