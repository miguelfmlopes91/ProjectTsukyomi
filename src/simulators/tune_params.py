#!/usr/bin/env python3
"""
Grid-search simples para afinar TP/SL/cooldown do cricket_scalper_sim.py.

Uso:
  python src/simulators/tune_params.py \
    --csv "src/live - Timeline.csv" \
    --out tune_out

Argumentos opcionais para alinhar com o simulador:
  --start-odds 1.5 --tick 0.01 --ball-secs 4
"""

import argparse
import itertools
import os
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PATH = os.path.join(HERE, "cricket_scalper_sim.py")

def run_once(csv_path, outdir, tp, sl, cooldown, start_odds, tick, ball_secs):
    combo_out = os.path.join(outdir, f"tp{tp}_sl{sl}_cd{cooldown}")
    os.makedirs(combo_out, exist_ok=True)

    cmd = [
        sys.executable, SIM_PATH,
        "--csv", csv_path,
        "--out", combo_out,
        "--tp", str(tp),
        "--sl", str(sl),
        "--cooldown", str(cooldown),
        "--start-odds", str(start_odds),
        "--tick", str(tick),
        "--ball-secs", str(ball_secs),
    ]

    # Se um combo der erro, não queremos parar todo o grid; capturamos o código de saída.
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        return {
            "tp": tp, "sl": sl, "cooldown": cooldown,
            "trades": 0, "wins": 0, "losses": 0, "eod": 0, "total_pnl": float("-inf"),
            "status": f"ERROR ({ret.returncode})",
            "stderr": (ret.stderr or "").strip()[:300]
        }

    # Ler summary.txt (o simulador escreve trades, wins, losses, total_pnl)
    summary_file = os.path.join(combo_out, "summary.txt")
    metrics = {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
    try:
        with open(summary_file, "r") as f:
            for line in f:
                line = line.strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k in ("trades", "wins", "losses"):
                    metrics[k] = int(v)
                elif k == "total_pnl":
                    metrics[k] = float(v)
    except FileNotFoundError:
        return {
            "tp": tp, "sl": sl, "cooldown": cooldown,
            "trades": 0, "wins": 0, "losses": 0, "eod": 0, "total_pnl": float("-inf"),
            "status": "NO_SUMMARY", "stderr": ""
        }

    trades = metrics["trades"]
    wins   = metrics["wins"]
    losses = metrics["losses"]
    eod    = max(0, trades - wins - losses)  # calcula eod aqui
    total_pnl = metrics["total_pnl"]

    return {
        "tp": tp, "sl": sl, "cooldown": cooldown,
        "trades": trades, "wins": wins, "losses": losses, "eod": eod,
        "total_pnl": total_pnl,
        "status": "OK", "stderr": ""
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV da timeline (caminho tal como usarias no simulador).")
    ap.add_argument("--out", default="tune_out", help="Diretório de saída para os resultados do grid.")
    # knobs adicionais (mantém default igual ao simulador)
    ap.add_argument("--start-odds", type=float, default=1.50)
    ap.add_argument("--tick", type=float, default=0.01)
    ap.add_argument("--ball-secs", type=int, default=4)
    # grelha básica (podes alterar aqui sem mexer no código)
    ap.add_argument("--tps", default="1,2", help="Lista de TP ticks (ex: 1,2,3)")
    ap.add_argument("--sls", default="1,2,3", help="Lista de SL ticks (ex: 1,2,3)")
    ap.add_argument("--cds", default="4,6,8,10", help="Lista de cooldown (bolas) (ex: 4,6,8,10)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # parse listas
    tps = [int(x) for x in args.tps.split(",") if x.strip()]
    sls = [int(x) for x in args.sls.split(",") if x.strip()]
    cds = [int(x) for x in args.cds.split(",") if x.strip()]

    print(f"🔎 Tuning contra: {args.csv}")
    print(f"   Grid: TP={tps} | SL={sls} | CD={cds}")
    print(f"   Extra: start_odds={args.start_odds} | tick={args.tick} | ball_secs={args.ball_secs}")

    rows = []
    for tp, sl, cd in itertools.product(tps, sls, cds):
        print(f" → tp={tp} sl={sl} cooldown={cd} …")
        res = run_once(
            args.csv, args.out, tp, sl, cd,
            start_odds=args.start_odds, tick=args.tick, ball_secs=args.ball_secs
        )
        rows.append(res)

    df = pd.DataFrame(rows)
    # leaderboard: ordenar por total_pnl desc; em caso de erro o total_pnl é -inf, vai para o fim
    df_sorted = df.sort_values(["total_pnl", "trades"], ascending=[False, False])
    leaderboard = os.path.join(args.out, "leaderboard.csv")
    df_sorted.to_csv(leaderboard, index=False)
    print(f"✅ Tuning concluído. Leaderboard em: {leaderboard}")

if __name__ == "__main__":
    main()
