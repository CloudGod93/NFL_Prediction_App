# src/data_handler.py

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime, date, timedelta

def get_schedule(year, week):
    """Fetches the NFL schedule for a specific year and week."""
    try:
        print(f"Fetching schedule for {year}, Week {week}...")
        schedule = nfl.import_schedules([year])
        if schedule.empty or 'week' not in schedule.columns:
            return pd.DataFrame()
        return schedule[schedule['week'] == week]
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return pd.DataFrame()

def get_training_data(start_year, end_year):
    """Downloads all necessary data for model training."""
    try:
        years = list(range(start_year, end_year + 1))
        print(f"Downloading data for years: {years}")
        pbp = nfl.import_pbp_data(years)
        weekly = nfl.import_weekly_data(years)
        schedule = nfl.import_schedules(years)
        return pbp, weekly, schedule
    except Exception as e:
        print(f"Error downloading training data: {e}")
        return None, None, None

def get_current_nfl_week():
    """
    Calculates the current NFL season year and week.
    A new week is considered to start on Tuesday.
    """
    today = date.today()
    year = today.year

    first_day_of_sept = date(year, 9, 1)
    days_to_thursday = (3 - first_day_of_sept.weekday() + 7) % 7
    kickoff_thursday = first_day_of_sept + timedelta(days=days_to_thursday)

    start_of_week_1 = kickoff_thursday - timedelta(days=2)

    if today < start_of_week_1:
        return year, 1

    days_since_start = (today - start_of_week_1).days
    current_week = (days_since_start // 7) + 1

    if current_week > 18:
        current_week = 18

    return year, current_week