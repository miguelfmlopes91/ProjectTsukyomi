# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Paths
    ROOT_DIR = Path(__file__).parent
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"
    LOGS_DIR = ROOT_DIR / "logs"

    # Database
    DATABASE_PATH = DATA_DIR / "cricket_data.db"

    # ML Parameters
    ML_LOOKBACK_WINDOWS = [5, 10, 20, 50]
    ML_VALIDATION_SPLITS = 3
    ML_EARLY_STOPPING_ROUNDS = 50

    # Trading Parameters
    DEFAULT_BANKROLL = 1000.0
    MAX_DAILY_LOSS = 50.0
    MAX_POSITION_RISK = 5.0

    # Scraping
    SCRAPING_INTERVAL_MINUTES = 2
    MAX_MATCHES_PER_CYCLE = 3

    # API Keys (if you get Betfair API)
    BETFAIR_USERNAME = os.getenv('BETFAIR_USERNAME')
    BETFAIR_PASSWORD = os.getenv('BETFAIR_PASSWORD')
    BETFAIR_APP_KEY = os.getenv('BETFAIR_APP_KEY')