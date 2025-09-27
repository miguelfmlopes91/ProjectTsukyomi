#!/usr/bin/env python3
"""
Proper SportCenter Multi-Game Scraper
Uses the actual get_events API endpoint to discover matches
"""

import requests
import pandas as pd
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ProperSportCenterScraper:
    def __init__(self, output_dir: str = "data/games"):
        self.base_url = "https://lsc.fn.sportradar.com/common/en/Etc:UTC/cricket"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Your exact working headers
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://sportcenter.sir.sportradar.com/",
            "Origin": "https://sportcenter.sir.sportradar.com"
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_active_matches(self) -> List[Dict]:
        """Get active matches using the get_events API endpoint"""
        events_url = f"{self.base_url}/get_events/"

        try:
            print(f"📡 Fetching active matches from: {events_url}")
            response = self.session.get(events_url, timeout=15)
            response.raise_for_status()

            data = response.json()

            if not data or 'doc' not in data:
                print("⚠️  No data in get_events response")
                return []

            matches = []
            sport_events = data['doc'][0]['data']['sportEvents']

            print(f"📊 Found {len(sport_events)} total events")

            for event in sport_events:
                try:
                    # Extract match information
                    match_id = event['premiumCricketEventId']
                    event_name = event['eventName']
                    format_type = event['format']
                    competition = event['competition']
                    status = event['status']

                    # Filter for T20 matches (more trading opportunities)
                    if '20 Overs' in format_type:
                        matches.append({
                            'match_id': str(match_id),
                            'event_name': event_name,
                            'format': format_type,
                            'competition': competition,
                            'status': status,
                            'scheduled_start': event.get('scheduledStart', ''),
                            'event_id': event['eventId']
                        })
                        print(f"  ✅ {event_name} (ID: {match_id}) - {format_type}")

                except Exception as e:
                    print(f"  ⚠️  Error parsing event: {e}")
                    continue

            print(f"🏏 Filtered to {len(matches)} T20 matches")
            return matches

        except Exception as e:
            print(f"❌ Error fetching events: {e}")
            return []

    def get_match_scorecard(self, match_id: str) -> Optional[dict]:
        """Get match scorecard data (your existing logic)"""
        score_url = f"{self.base_url}/get_scorecard/{match_id}"

        try:
            response = self.session.get(score_url, timeout=15)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"  ⚠️  Error fetching scorecard for {match_id}: {e}")
            return None

    def process_match_to_csv(self, match_id: str, match_info: Dict) -> Optional[str]:
        """Process match data to CSV using your exact logic"""
        print(f"📊 Processing: {match_info['event_name']} (ID: {match_id})")

        # Get scorecard data
        data = self.get_match_scorecard(match_id)
        if not data:
            return None

        try:
            # Your exact processing logic from upload_to_gsheet.py
            innings = data['doc'][0]['data']['score']['innings']
            ball_by_ball = data['doc'][0]['data']['score']['ballByBallSummaries']

            # Check if match has actual ball-by-ball data
            if not ball_by_ball:
                print(f"  ⚠️  No ball-by-ball data available")
                return None

            # MAPEAR TIMES E PREPARAR DADOS (your exact code)
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
                                val = int(b) if b.isdigit() else 0
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

            # CONSTRUIR A TABELA FINAL (your exact code)
            rows = []
            teams = list(set(row["Team"] for over in over_dict.values() for row in over.values()))
            teams = sorted(teams)

            teamA = teams[0] if len(teams) > 0 else "Team A"
            teamB = teams[1] if len(teams) > 1 else "Team B (TBD)"

            for over in range(1, 21):
                rowA = over_dict.get(over, {}).get(teamA, {"Team": teamA, "Over": over, "Balls": [], "Total": 0, "W": 0,
                                                           "BOUN.": 0, "Dif.": "", "Dif.2": ""})
                rowB = over_dict.get(over, {}).get(teamB, {"Team": teamB, "Over": over, "Balls": [], "Total": 0, "W": 0,
                                                           "BOUN.": 0, "Dif.": "", "Dif.2": ""})

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

            df = pd.DataFrame(rows)

            # Validate the data has actual ball content
            ball_cols = [c for c in df.columns if str(c).isdigit()]
            if not ball_cols or df[ball_cols].isna().all().all():
                print(f"  ⚠️  No actual ball data found")
                return None

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = match_info['event_name'].replace(' ', '_').replace('v', 'vs')
            # Remove special characters for filename safety
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ['_', '-'])[:60]

            filename = f"{timestamp}_{match_id}_{safe_name}.csv"
            filepath = self.output_dir / filename

            df.to_csv(filepath, index=False)

            # Calculate stats
            total_boundaries = sum(df[col].astype(str).str.contains('4|6', na=False).sum() for col in ball_cols)
            total_wickets = sum(df[col].astype(str).str.contains('w', na=False).sum() for col in ball_cols)

            print(f"  ✅ Saved: {filepath}")
            print(f"     📊 Boundaries: {total_boundaries}, Wickets: {total_wickets}")

            return str(filepath)

        except Exception as e:
            print(f"  ❌ Processing error: {e}")
            return None

    def scrape_all_active_matches(self, max_matches: int = 10) -> List[str]:
        """Main method: get all active matches and process them"""
        print("🚀 Starting proper SportCenter multi-game scraping...")

        # Get active matches from API
        matches = self.get_active_matches()

        if not matches:
            print("⚠️  No active matches found")
            return []

        saved_files = []
        processed = 0

        # Process up to max_matches
        matches_to_process = matches[:max_matches]

        for i, match in enumerate(matches_to_process):
            print(f"\n📊 Processing {i + 1}/{len(matches_to_process)}...")

            filepath = self.process_match_to_csv(match['match_id'], match)

            if filepath:
                saved_files.append(filepath)
                processed += 1

            # Rate limiting - be nice to the API
            time.sleep(2)

        print(f"\n🎯 Scraping complete!")
        print(f"✅ Successfully processed {processed}/{len(matches_to_process)} matches")
        print(f"📁 Files saved to: {self.output_dir}")

        return saved_files

    def update_live_match(self, match_id: str = "62657839"):
        """Update single live match (your original functionality)"""
        data = self.get_match_scorecard(match_id)
        if not data:
            print(f"⚠️  No data for match {match_id}")
            return

        # Process and save as live timeline
        match_info = {'event_name': f'Match_{match_id}', 'format': '20 Overs'}
        filepath = self.process_match_to_csv(match_id, match_info)

        if filepath:
            # Copy to your original live timeline location
            live_path = Path(__file__).parent.parent / "live - Timeline.csv"
            df = pd.read_csv(filepath)
            df.to_csv(live_path, index=False)
            print(f"✅ Updated live timeline: {live_path}")


def create_automated_test_script(saved_files: List[str], output_dir: str):
    """Create script to automatically test all scraped games"""
    if not saved_files:
        return None

    script_content = f'''#!/bin/bash
echo "🏏 Testing {len(saved_files)} cricket games with improved parameters..."

mkdir -p results
total_games={len(saved_files)}
current=0
summary_file="results/all_games_summary.txt"

echo "Game,Trades,Wins,Losses,Total_PnL" > "$summary_file"

for game in {output_dir}/*.csv; do
    current=$((current + 1))
    echo ""
    echo "🎯 Testing game $current/$total_games: $(basename "$game")"

    game_name=$(basename "$game" .csv)
    output_dir="results/$game_name"

    # Test with improved parameters
    python src/simulators/cricket_scalper_sim.py \\
        --csv "$game" \\
        --out "$output_dir" \\
        --tp 2 --sl 1 --cooldown 6 \\
        --start-odds 1.50

    # Extract results and add to summary
    if [ -f "$output_dir/summary.txt" ]; then
        trades=$(grep "trades=" "$output_dir/summary.txt" | cut -d'=' -f2)
        wins=$(grep "wins=" "$output_dir/summary.txt" | cut -d'=' -f2)
        losses=$(grep "losses=" "$output_dir/summary.txt" | cut -d'=' -f2)
        pnl=$(grep "total_pnl=" "$output_dir/summary.txt" | cut -d'=' -f2)

        echo "$game_name,$trades,$wins,$losses,$pnl" >> "$summary_file"

        echo "  📊 Results: $trades trades, $wins wins, $losses losses, PnL: $pnl"
    fi
done

echo ""
echo "🎉 All games tested!"
echo "📋 Summary saved to: $summary_file"
echo ""
echo "📈 Quick Analysis:"
python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$summary_file')
    print(f'Average PnL: {{df[\\\"Total_PnL\\\"].mean():.4f}}')
    print(f'Best game: {{df.loc[df[\\\"Total_PnL\\\"].idxmax(), \\\"Game\\\"]}} ({{df[\\\"Total_PnL\\\"].max():.4f}})')
    print(f'Win rate: {{(df[\\\"Wins\\\"] > 0).mean() * 100:.1f}}% of games had winning trades')
except:
    print('Run the analysis manually')
"
'''

    script_path = "test_all_cricket_games.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Proper SportCenter Multi-Game Scraper")
    parser.add_argument("--mode", choices=['all', 'live'], default='all')
    parser.add_argument("--max-matches", type=int, default=10)
    parser.add_argument("--match-id", default="62657839")
    parser.add_argument("--output-dir", default="data/games")

    args = parser.parse_args()

    scraper = ProperSportCenterScraper(output_dir=args.output_dir)

    if args.mode == 'all':
        saved_files = scraper.scrape_all_active_matches(max_matches=args.max_matches)

        if saved_files:
            print(f"\n📋 SUCCESSFULLY SCRAPED GAMES:")
            for i, file in enumerate(saved_files, 1):
                print(f"  {i}. {Path(file).name}")

            # Create automated testing script
            script_path = create_automated_test_script(saved_files, args.output_dir)
            if script_path:
                print(f"\n🚀 AUTOMATED TESTING READY:")
                print(f"  Created script: {script_path}")
                print(f"  Run with: ./{script_path}")
                print(f"  This will test all games and create a summary report")
        else:
            print("⚠️  No games were successfully scraped")

    elif args.mode == 'live':
        scraper.update_live_match(args.match_id)


if __name__ == "__main__":
    main()