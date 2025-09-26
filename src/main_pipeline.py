# src/main_pipeline.py
import schedule
import time
from datetime import datetime
import logging
from utils.logger import setup_logger
from scrapers.betfair_scraper_noapi import BetfairCricketScraper
from ml.cricket_ml_pipeline import CricketMLPipeline
from database.db_manager import CricketDatabaseManager

logger = setup_logger("main_pipeline")


class CricketTradingPipeline:
    def __init__(self):
        self.scraper = BetfairCricketScraper(headless=True)
        self.ml_pipeline = CricketMLPipeline()
        self.db_manager = CricketDatabaseManager()

    def daily_data_collection(self):
        """Run daily data collection"""
        logger.info("Starting daily data collection...")
        try:
            # Scrape odds data
            matches = self.scraper.find_cricket_matches()

            for match in matches[:3]:
                odds_data = self.scraper.scrape_match_odds(
                    match['url'], match['name']
                )
                self.scraper.save_odds_to_db(odds_data)

            logger.info(f"Collected data from {len(matches)} matches")

        except Exception as e:
            logger.error(f"Data collection failed: {e}")

    def weekly_model_retraining(self):
        """Retrain ML models weekly"""
        logger.info("Starting weekly model retraining...")
        try:
            results = self.ml_pipeline.run_training_pipeline()
            logger.info(f"Retrained {len(results)} models")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def run_scheduler(self):
        """Run the automated pipeline scheduler"""
        logger.info("Starting cricket trading pipeline scheduler...")

        # Schedule tasks
        schedule.every(2).hours.do(self.daily_data_collection)
        schedule.every().sunday.at("02:00").do(self.weekly_model_retraining)

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user")
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying


if __name__ == "__main__":
    pipeline = CricketTradingPipeline()
    pipeline.run_scheduler()