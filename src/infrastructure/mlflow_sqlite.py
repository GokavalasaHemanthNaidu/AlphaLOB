import sqlite3
import os
import json
import time

DB_PATH = "alphalob_mlflow.db"

def init_mlflow_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for general metrics like latency
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            metric_name TEXT,
            value REAL
        )
    ''')
    
    # Table for tracking signal distributions to detect drift
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signal_distribution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            dir_up_prob REAL,
            dir_down_prob REAL,
            spread_compress_prob REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def log_metric(metric_name: str, value: float):
    """
    Mimics mlflow.log_metric()
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO metrics (timestamp, metric_name, value) VALUES (?, ?, ?)",
        (time.time(), metric_name, value)
    )
    conn.commit()
    conn.close()

def log_prediction_distribution(up_prob: float, down_prob: float, spread_prob: float):
    """
    Logs the raw probabilities of predictions to track data drift over time.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO signal_distribution (timestamp, dir_up_prob, dir_down_prob, spread_compress_prob) VALUES (?, ?, ?, ?)",
        (time.time(), up_prob, down_prob, spread_prob)
    )
    conn.commit()
    conn.close()

def get_recent_metrics(metric_name: str, limit: int = 1000):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT value FROM metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT ?",
        (metric_name, limit)
    )
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results

def get_recent_distributions(limit: int = 1000):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT dir_up_prob, dir_down_prob, spread_compress_prob FROM signal_distribution ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return results
