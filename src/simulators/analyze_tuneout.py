import pandas as pd

# carregar leaderboard
lb = pd.read_csv("tune_out/leaderboard.csv")

# ordenar por PnL (maior → menor)
lb_sorted = lb.sort_values("total_pnl", ascending=False)

print("Top 5 setups:\n")
print(lb_sorted.head(5)[["tp","sl","cooldown","trades","wins","losses","total_pnl"]])

print("\nPiores 5 setups:\n")
print(lb_sorted.tail(5)[["tp","sl","cooldown","trades","wins","losses","total_pnl"]])

print("\nEstatísticas gerais:")
print("Média de PnL:", lb["total_pnl"].mean())
print("Melhor PnL:", lb["total_pnl"].max())
print("Pior PnL:", lb["total_pnl"].min())
