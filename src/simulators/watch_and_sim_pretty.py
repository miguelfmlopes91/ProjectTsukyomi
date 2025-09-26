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

SUMMARY = "summary.txt"
TRADE_LOG = "trade_log.csv"

def sha1_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

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

# ---------- Pretty formatting helpers ----------

def _fmt_float(x, prec=4):
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return str(x)

def _render_table(rows: List[Dict[str, str]], headers: List[str], max_width: int = 22) -> str:
    """Renderiza lista de dicts como tabela ASCII alinhada."""
    if not rows:
        return "(sem dados)"
    # calcular larguras por coluna
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

def _parse_csv_text(text: str, limit: int = 3) -> List[Dict[str,str]]:
    """Converte CSV em lista de dicts (primeiras `limit` linhas)."""
    f = StringIO(text.strip())
    reader = csv.DictReader(f)
    rows = []
    for i, row in enumerate(reader):
        if i >= limit:
            break
        rows.append(row)
    return rows

def pretty_print(trades, wins, losses, pnl, delta, last_trades_text, first=False):
    now = time.strftime("%H:%M:%S")
    delta_str = "—" if (delta is None) else f"{delta:+.4f}"
    prefix = "primeira run | " if first else ""
    print(f"🕒 {now} | {prefix}Δ desde última run: {delta_str}")

    # resumo
    summary_rows = [
        {"metric": "trades",     "value": str(trades or 0)},
        {"metric": "wins",       "value": str(wins or 0)},
        {"metric": "losses",     "value": str(losses or 0)},
        {"metric": "eod",        "value": str(max(0, (trades or 0) - (wins or 0) - (losses or 0)))},
        {"metric": "total_pnl",  "value": _fmt_float(pnl or 0.0, prec=4)},
    ]
    print(_render_table(summary_rows, headers=["metric","value"]))
    print()

    # últimos trades
    if last_trades_text and last_trades_text.strip():
        try:
            rows = _parse_csv_text(last_trades_text, limit=3)
            cols_pref = ["entry_time","exit_time","entry_back","exit_lay","ticks","pnl","reason","ball","raw"]
            cols = [c for c in cols_pref if rows and c in rows[0]]
            if cols:
                print("Últimos trades:")
                print(_render_table(rows, headers=cols, max_width=22))
                print()
            else:
                # fallback se colunas inesperadas
                print("Últimos trades (raw):")
                print(last_trades_text)
                print()
        except Exception:
            print("Últimos trades (raw):")
            print(last_trades_text)
            print()

# ---------- Runner ----------

def run_sim(args, quiet=True):
    cmd = f'''python cricket_scalper_sim.py \
      --csv "{args.csv}" \
      --out "{args.out}" \
      --tp {args.tp} --sl {args.sl} --cooldown {args.cooldown} \
      --start-odds {args.start_odds} --tick {args.tick} --ball-secs {args.ball_secs}'''
    if quiet:
        cmd += " --quiet"
        subprocess.run(shlex.split(cmd), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(shlex.split(cmd), check=False)

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
                    help="Imprime um resumo na primeira execução (mesmo sem delta).")
    args = ap.parse_args()

    last_csv_sig = None
    last_output_sig = None
    last_pnl = None

    print("👀 watcher a correr… (Ctrl+C para parar)")
    while True:
        try:
            if not os.path.exists(args.csv):
                time.sleep(args.interval)
                continue

            csv_sig = sha1_file(args.csv)

            if csv_sig and csv_sig != last_csv_sig:
                print(f"🕒 {time.strftime('%H:%M:%S')} | mudança detetada → a simular…")
                run_sim(args, quiet=True)

                trades, wins, losses, pnl = read_summary(args.out)
                out_concat = f"{pnl}|{sha1_file(os.path.join(args.out, TRADE_LOG))}"
                output_sig = hashlib.sha1(out_concat.encode()).hexdigest()

                if args.first_run and last_output_sig is None:
                    last_trades = read_last_trades(args.out)
                    delta = None
                    pretty_print(trades, wins, losses, pnl, delta, last_trades, first=True)
                    last_pnl = pnl
                    last_output_sig = output_sig
                    last_csv_sig = csv_sig
                else:
                    delta = (pnl - last_pnl) if (pnl is not None and last_pnl is not None) else None
                    if delta is not None and abs(delta) < 1e-6 and output_sig == last_output_sig:
                        print(f"🕒 {time.strftime('%H:%M:%S')} | CSV updated, nothing has changed")
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
