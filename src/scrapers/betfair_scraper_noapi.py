# src/scrapers/betfair_scraper_noapi.py
"""
Betfair Cricket Odds Scraper (No API account required)
Scrapes odds from Betfair public pages using Selenium

Requirements:
pip install selenium beautifulsoup4 requests-html
"""

import time
import pandas as pd
import sqlite3
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import json
import random
from typing import Dict, List, Optional
from pathlib import Path
import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager


def _sleep(s):
    time.sleep(s + random.uniform(0.2, 0.6))

class BetfairCricketScraper:
    def __init__(self, headless: bool = True, db_path: str = "data/betfair_odds.db"):
        """Initialize the scraper"""
        self.db_path = db_path
        self.setup_database()

        # UC chrome options
        chrome_opts = uc.ChromeOptions()
        if headless:
            # new headless is less detectable
            chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        chrome_opts.add_argument("--disable-blink-features=AutomationControlled")
        chrome_opts.add_argument("--window-size=1366,900")
        chrome_opts.add_argument("--lang=en-GB")
        chrome_opts.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

        self.driver = uc.Chrome(driver_executable_path=ChromeDriverManager().install(),
                                options=chrome_opts)
        # small stealth tweak
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")


    def setup_database(self):
        """Create SQLite database for storing odds"""
        Path(self.db_path).parent.mkdir(exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cricket_odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                match_name TEXT,
                market_type TEXT,
                selection_name TEXT,
                back_price REAL,
                lay_price REAL,
                volume REAL,
                url TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp_match 
            ON cricket_odds(timestamp, match_name)
        ''')

        conn.commit()
        conn.close()

    def _consent_and_settle(self):
        # Try several common consent buttons
        for sel in [
            "//button[contains(., 'Accept') or contains(., 'accept')]",
            "//button[contains(., 'Agree') or contains(., 'agree')]",
            "//button[contains(., 'I Accept')]",
            "//*[@id='onetrust-accept-btn-handler']",
        ]:
            try:
                btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, sel))
                )
                btn.click()
                _sleep(0.8)
                break
            except Exception:
                pass

    def _scroll_page(self, times=4):
        for _ in range(times):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            _sleep(1.0)

    def _debug_dump(self, tag="cricket"):
        Path("out_debug").mkdir(exist_ok=True)
        html_path = Path(f"out_debug/{tag}.html")
        png_path = Path(f"out_debug/{tag}.png")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self.driver.page_source)
        try:
            self.driver.save_screenshot(str(png_path))
        except Exception:
            pass
        print(f"🧪 Saved debug snapshot: {html_path}, {png_path}")

    def find_cricket_matches(self) -> List[Dict]:
        """Find active cricket matches on Betfair (robust selectors + scrolling)."""
        cricket_urls = [
            "https://www.betfair.com/exchange/plus/cricket",
            "https://www.betfair.com/sport/cricket",
        ]
        matches = []

        for url in cricket_urls:
            try:
                print(f"🔍 Checking {url}...")
                self.driver.get(url)
                _sleep(2.5)
                self._consent_and_settle()
                self._scroll_page(times=5)  # force lazy load

                # Collect all <a> hrefs using JS (faster than find_elements when many nodes)
                links = self.driver.execute_script(
                    "return Array.from(document.querySelectorAll('a')).map(a => [a.innerText, a.href]);"
                ) or []

                # Filter candidates
                cand = []
                for text, href in links:
                    text_l = (text or "").strip()
                    href_l = (href or "")
                    if not href_l:
                        continue
                    if ("cricket" in href_l and (
                            "/exchange/plus/cricket/event/" in href_l or
                            "/exchange/plus/cricket/market/" in href_l or
                            "/sport/cricket" in href_l
                    )):
                        # crude "vs" detection in text
                        if " v " in text_l.lower() or " vs " in text_l.lower() or "match odds" in text_l.lower():
                            cand.append((text_l, href_l))

                # Deduplicate by href
                seen = set()
                for name, href in cand:
                    if href in seen:
                        continue
                    seen.add(href)
                    # Clean name
                    name = name.replace("\n", " ").strip()
                    if name:
                        matches.append({"name": name, "url": href, "source": url})

                if matches:
                    print(f"📊 Found {len(matches)} matches on {url}")
                    # Don’t stop early; move on to next url for more
                else:
                    # Dump page to debug
                    print("⚠️  No matches found on this variant; saving snapshot.")
                    self._debug_dump(tag="cricket_page")

            except Exception as e:
                print(f"⚠️  Error accessing {url}: {e}")
                self._debug_dump(tag="cricket_error")

        # Heuristic: return top N unique
        # Often same match appears with multiple routes; keep first 5.
        # If still empty, the debug dump will tell us what the DOM looks like.
        unique = []
        seen_href = set()
        for m in matches:
            if m["url"] in seen_href:
                continue
            seen_href.add(m["url"])
            unique.append(m)
        return unique[:5]

    def scrape_match_odds(self, match_url: str, match_name: str) -> List[Dict]:
        """Scrape odds for a specific cricket match"""
        odds_data = []

        try:
            print(f"🎯 Scraping odds for: {match_name}")
            self.driver.get(match_url)

            # Random delay to avoid detection
            time.sleep(random.uniform(2, 4))

            # Wait for odds to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "bet-button"))
                )
            except TimeoutException:
                print("⚠️  Odds not loaded, trying alternative selectors...")

            # Multiple strategies to find odds
            odds_selectors = [
                ".bet-button",
                ".price",
                "[data-testid*='price']",
                ".odds",
                ".back-selection-button",
                ".lay-selection-button"
            ]

            for selector in odds_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        print(f"✅ Found {len(elements)} odds elements with selector: {selector}")
                        break
                except:
                    continue

            # Parse page source with BeautifulSoup for better parsing
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Look for common betting patterns
            bet_buttons = soup.find_all(['button', 'div'], class_=lambda x: x and any(
                keyword in str(x).lower() for keyword in ['bet', 'back', 'lay', 'price', 'odds']
            ))

            current_time = datetime.now()

            # Extract odds data
            for i, button in enumerate(bet_buttons[:20]):  # Limit to avoid spam
                try:
                    # Try to extract price
                    price_text = button.get_text(strip=True)

                    # Look for decimal odds pattern (e.g., "1.50", "2.25")
                    import re
                    price_match = re.search(r'\b(\d+\.\d{2})\b', price_text)

                    if price_match:
                        price = float(price_match.group(1))

                        # Determine if back or lay based on context
                        button_classes = ' '.join(button.get('class', []))
                        is_back = 'back' in button_classes.lower()
                        is_lay = 'lay' in button_classes.lower()

                        # Try to find selection name (team/outcome)
                        selection_name = "Unknown"
                        parent = button.find_parent()
                        if parent:
                            selection_text = parent.get_text(strip=True)
                            # Look for team names or "Draw"
                            if any(word in selection_text.lower() for word in ['v', 'vs', 'draw', 'tie']):
                                selection_name = selection_text[:50]  # Truncate

                        odds_data.append({
                            'timestamp': current_time,
                            'match_name': match_name,
                            'market_type': 'match_odds',
                            'selection_name': selection_name,
                            'back_price': price if is_back or not is_lay else None,
                            'lay_price': price if is_lay else None,
                            'volume': None,  # Not available from scraping
                            'url': match_url
                        })

                except Exception as e:
                    continue

            # Alternative: Look for JSON data in page source
            json_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.*?});'
            json_match = re.search(json_pattern, self.driver.page_source, re.DOTALL)

            if json_match:
                try:
                    json_data = json.loads(json_match.group(1))
                    # Parse JSON for odds data (structure varies)
                    print("📊 Found JSON data, parsing...")
                    # This would require reverse engineering Betfair's data structure
                except:
                    pass

        except Exception as e:
            print(f"❌ Error scraping {match_name}: {e}")

        return odds_data

    def save_odds_to_db(self, odds_data: List[Dict]):
        """Save scraped odds to database"""
        if not odds_data:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for odds in odds_data:
            cursor.execute('''
                INSERT INTO cricket_odds 
                (timestamp, match_name, market_type, selection_name, 
                 back_price, lay_price, volume, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                odds['timestamp'], odds['match_name'], odds['market_type'],
                odds['selection_name'], odds['back_price'], odds['lay_price'],
                odds['volume'], odds['url']
            ))

        conn.commit()
        conn.close()

        print(f"💾 Saved {len(odds_data)} odds records to database")

    def run_continuous_scraping(self, interval_minutes: int = 2):
        """Run continuous odds scraping"""
        print(f"🚀 Starting continuous cricket odds scraping...")
        print(f"📊 Checking every {interval_minutes} minutes (Ctrl+C to stop)")

        try:
            while True:
                print(f"\n🕒 {datetime.now().strftime('%H:%M:%S')} - Starting scraping cycle...")

                # Find matches
                matches = self.find_cricket_matches()

                if not matches:
                    print("⚠️  No cricket matches found")
                else:
                    print(f"📊 Found {len(matches)} matches")

                    # Scrape each match
                    for match in matches[:3]:  # Limit to 3 matches to avoid overload
                        odds_data = self.scrape_match_odds(match['url'], match['name'])

                        if odds_data:
                            self.save_odds_to_db(odds_data)
                            print(f"✅ {match['name']}: {len(odds_data)} odds collected")
                        else:
                            print(f"⚠️  {match['name']}: No odds found")

                        # Random delay between matches
                        time.sleep(random.uniform(3, 8))

                print(f"😴 Waiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n👋 Scraping stopped by user")
        except Exception as e:
            print(f"❌ Scraping error: {e}")
        finally:
            self.close()

    def export_odds_csv(self, output_path: str = "data/betfair_odds_export.csv"):
        """Export scraped odds to CSV"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT timestamp, match_name, market_type, selection_name,
                   back_price, lay_price, volume
            FROM cricket_odds
            ORDER BY timestamp DESC
        '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Create output directory
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"📊 Exported {len(df)} odds records to {output_path}")
        return df

    def get_latest_odds(self, match_name_filter: str = None) -> pd.DataFrame:
        """Get latest odds from database"""
        conn = sqlite3.connect(self.db_path)

        if match_name_filter:
            query = '''
                SELECT * FROM cricket_odds
                WHERE match_name LIKE ?
                ORDER BY timestamp DESC
                LIMIT 50
            '''
            df = pd.read_sql_query(query, conn, params=[f'%{match_name_filter}%'])
        else:
            query = '''
                SELECT * FROM cricket_odds
                ORDER BY timestamp DESC
                LIMIT 100
            '''
            df = pd.read_sql_query(query, conn)

        conn.close()
        return df

    def close(self):
        """Close the browser driver"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            print("🔒 Browser closed")


# Alternative: Simple requests-based scraper for odds comparison sites
class OddsCheckerScraper:
    """Scrape cricket odds from odds comparison sites"""

    def __init__(self):
        self.session = None
        self.setup_session()

    def setup_session(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def scrape_oddschecker_cricket(self) -> List[Dict]:
        """Scrape cricket odds from Oddschecker"""
        odds_data = []

        try:
            url = "https://www.oddschecker.com/cricket"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find cricket matches
                match_rows = soup.find_all('tr', class_=lambda x: x and 'match-row' in str(x))

                for row in match_rows[:5]:  # Limit to 5 matches
                    try:
                        # Extract match name
                        match_element = row.find('td', class_='match-name')
                        if match_element:
                            match_name = match_element.get_text(strip=True)

                            # Extract odds
                            odds_cells = row.find_all('td', class_=lambda x: x and 'odds' in str(x))

                            for cell in odds_cells:
                                odds_text = cell.get_text(strip=True)
                                # Parse odds (e.g., "3/1", "1.50")
                                import re
                                decimal_odds = re.search(r'(\d+\.\d{2})', odds_text)
                                if decimal_odds:
                                    odds_data.append({
                                        'timestamp': datetime.now(),
                                        'match_name': match_name,
                                        'odds': float(decimal_odds.group(1)),
                                        'source': 'oddschecker'
                                    })

                    except Exception as e:
                        continue

        except Exception as e:
            print(f"❌ Error scraping Oddschecker: {e}")

        return odds_data


# Usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Betfair cricket odds")
    parser.add_argument("--mode", choices=['continuous', 'single', 'export'],
                        default='single', help="Scraping mode")
    parser.add_argument("--interval", type=int, default=2,
                        help="Minutes between scraping cycles")
    parser.add_argument("--headless", action='store_true',
                        help="Run browser in headless mode")

    args = parser.parse_args()

    if args.mode == 'single':
        scraper = BetfairCricketScraper(headless=args.headless)
        try:
            matches = scraper.find_cricket_matches()

            if matches:
                print(f"📊 Found {len(matches)} matches")
                # Scrape first match as example
                match = matches[0]
                odds_data = scraper.scrape_match_odds(match['url'], match['name'])
                scraper.save_odds_to_db(odds_data)

                # Export to CSV
                df = scraper.export_odds_csv()
                print(f"✅ Exported data shape: {df.shape}")
            else:
                print("⚠️  No matches found")

        finally:
            scraper.close()

    elif args.mode == 'continuous':
        scraper = BetfairCricketScraper(headless=args.headless)
        scraper.run_continuous_scraping(interval_minutes=args.interval)

    elif args.mode == 'export':
        scraper = BetfairCricketScraper()
        df = scraper.export_odds_csv()
        print(f"📊 Exported {len(df)} records")
        scraper.close()