# src/database/db_manager.py
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging


class CricketDatabaseManager:
    def __init__(self, db_path: str = "data/cricket_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.setup_database()

    def setup_database(self):
        """Initialize all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Simulation results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                game_id TEXT,
                strategy_params TEXT,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                total_pnl REAL,
                max_drawdown REAL,
                win_rate REAL,
                avg_trade_pnl REAL
            )
        ''')

        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                auc_score REAL,
                feature_importance TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store_simulation_result(self, result: Dict):
        """Store simulation result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO simulation_results 
            (game_id, strategy_params, total_trades, wins, losses, 
             total_pnl, max_drawdown, win_rate, avg_trade_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['game_id'],
            str(result['strategy_params']),
            result['total_trades'],
            result['wins'],
            result['losses'],
            result['total_pnl'],
            result.get('max_drawdown', 0),
            result.get('win_rate', 0),
            result.get('avg_trade_pnl', 0)
        ))

        conn.commit()
        conn.close()