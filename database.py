import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "electricity_history.db"


def create_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                family_size INTEGER NOT NULL,
                fans INTEGER NOT NULL,
                lights INTEGER NOT NULL,
                ac_count INTEGER NOT NULL,
                ac_hours_per_day REAL NOT NULL,
                fridge INTEGER NOT NULL,
                washing_machine INTEGER NOT NULL,
                geyser INTEGER NOT NULL,
                tv_hours_per_day REAL NOT NULL,
                season TEXT NOT NULL,
                predicted_units REAL NOT NULL,
                predicted_bill REAL NOT NULL,
                bill_category TEXT NOT NULL,
                usage_group TEXT NOT NULL
            )
        """)


def save_prediction(data):
    create_database()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, family_size, fans, lights,
                ac_count, ac_hours_per_day, fridge,
                washing_machine, geyser, tv_hours_per_day,
                season, predicted_units, predicted_bill,
                bill_category, usage_group
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            data["family_size"],
            data["fans"],
            data["lights"],
            data["ac_count"],
            data["ac_hours_per_day"],
            data["fridge"],
            data["washing_machine"],
            data["geyser"],
            data["tv_hours_per_day"],
            data["season"],
            data["predicted_units"],
            data["predicted_bill"],
            data["bill_category"],
            data["usage_group"],
        ))


def get_predictions():
    create_database()

    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT * FROM predictions
            ORDER BY id DESC
        """).fetchall()