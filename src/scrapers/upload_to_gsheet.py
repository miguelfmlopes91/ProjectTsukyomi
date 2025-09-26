import requests
import pandas as pd
import os
import time
from pathlib import Path

def main():
    # === CONFIGURAÇÕES ===
    match_id = 62657839  # altera para o jogo que quiseres monitorar
    ROOT = Path(__file__).resolve().parent.parent
    final_path = ROOT / "live - Timeline.csv"

    # === OBTÉM DADOS DO JOGO ===
    BASE_URL = "https://lsc.fn.sportradar.com/common/en/Etc:UTC/cricket"
    score_url = f"{BASE_URL}/get_scorecard/{match_id}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://sportcenter.sir.sportradar.com/",
        "Origin": "https://sportcenter.sir.sportradar.com"
    }

    resp = requests.get(score_url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    innings = data['doc'][0]['data']['score']['innings']
    ball_by_ball = data['doc'][0]['data']['score']['ballByBallSummaries']

    # === MAPEAR TIMES E PREPARAR DADOS ===
    team_map = {}
    for idx, inn in enumerate(innings):
        key = f"{['first', 'second'][idx]}Innings"
        team_map[key] = inn['teamName']

    over_dict = {}
    max_balls = 0

    for over in ball_by_ball:
        over_num = over["overNumber"]
        for key in ["firstInnings", "secondInnings"]:
            balls_str = over.get(key)
            if not balls_str:
                continue

            team = team_map.get(key, key)
            balls = balls_str.split(",")

            runs_total, wickets, boundaries = 0, 0, 0
            clean_balls = []

            for b in balls:
                b = b.strip().lower()
                clean_balls.append(b)

                if b in ["w", "w1", "1w", "2w"]:
                    wickets += 1
                    val = int(b.replace("w", "") or "0")
                else:
                    try:
                        val = int(b)
                    except:
                        val = 0

                runs_total += val
                if val in [4, 6]:
                    boundaries += 1

            max_balls = max(max_balls, len(clean_balls))

            over_dict.setdefault(over_num, {})
            over_dict[over_num][team] = {
                "Team": team,
                "Over": over_num,
                "Balls": clean_balls,
                "Total": runs_total,
                "W": wickets,
                "BOUN.": boundaries,
                "Dif.": "",
                "Dif.2": ""
            }

    # === CONSTRUIR A TABELA FINAL ===
    rows = []
    teams = list(set(row["Team"] for over in over_dict.values() for row in over.values()))
    teams = sorted(teams)

    teamA = teams[0] if len(teams) > 0 else "Team A"
    teamB = teams[1] if len(teams) > 1 else "Team B (TBD)"

    for over in range(1, 21):
        rowA = over_dict.get(over, {}).get(teamA, {"Team": teamA, "Over": over, "Balls": [], "Total": 0, "W": 0, "BOUN.": 0, "Dif.": "", "Dif.2": ""})
        rowB = over_dict.get(over, {}).get(teamB, {"Team": teamB, "Over": over, "Balls": [], "Total": 0, "W": 0, "BOUN.": 0, "Dif.": "", "Dif.2": ""})

        diff = rowA["Total"] - rowB["Total"]
        rowB["Dif."] = diff
        rowB["Dif.2"] = abs(diff)

        def flatten(row):
            base = {"Team": row["Team"], "Over": row["Over"]}
            for i in range(max_balls):
                base[str(i + 1)] = row["Balls"][i] if i < len(row["Balls"]) else ""
            base.update({
                "Total": row["Total"],
                "W": row["W"],
                "BOUN.": row["BOUN."],
                "Dif.": row["Dif."],
                "Dif.2": row["Dif.2"]
            })
            return base

        rows.append(flatten(rowA))
        rows.append(flatten(rowB))

    # === ESCREVER CSV LOCAL ===
    df = pd.DataFrame(rows)
    tmp_path = f"{final_path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, final_path)  # atómico
    print(f"✅ Timeline atualizada para match {match_id} em '{final_path}' às {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
