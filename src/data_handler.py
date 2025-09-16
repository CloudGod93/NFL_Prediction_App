# src/data_handler.py

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime, date, timedelta

def get_schedule(year, week):
    """Fetches the NFL schedule for a specific year and week."""
    try:
        schedule = nfl.import_schedules([year])
        if schedule.empty or 'week' not in schedule.columns: return pd.DataFrame()
        return schedule[schedule['week'] == week]
    except Exception as e:
        print(f"Error fetching schedule: {e}"); return pd.DataFrame()

def get_training_data(start_year, end_year):
    """Downloads all necessary data for model training."""
    try:
        years = list(range(start_year, end_year + 1))
        pbp = nfl.import_pbp_data(years); weekly = nfl.import_weekly_data(years)
        schedule = nfl.import_schedules(years)
        return pbp, weekly, schedule
    except Exception as e:
        print(f"Error downloading training data: {e}"); return None, None, None

def get_team_stats(year):
    """
    NEW: Fetches and calculates key team stats for a given season.
    This is used for the Scouting Report feature.
    """
    try:
        roster = nfl.import_rosters([year])
        team_stats = roster.groupby('team').agg(
            wins=('wins', 'first'),
            losses=('losses', 'first'),
            ties=('ties', 'first')
        ).reset_index()

        weekly = nfl.import_weekly_data([year])
        season_stats = weekly.groupby('recent_team').agg(
            points_for=('fantasy_points_ppr', 'sum'),
            pass_yds=('passing_yards', 'sum'),
            rush_yds=('rushing_yards', 'sum'),
            turnovers=('interceptions', 'sum') + weekly.groupby('recent_team')['rushing_fumbles_lost'].sum()
        ).reset_index()
        season_stats['games_played'] = weekly.groupby('recent_team')['week'].nunique().values
        
        # Merge stats
        full_stats = pd.merge(team_stats, season_stats, left_on='team', right_on='recent_team', how='left')
        return full_stats.set_index('team')
    except Exception as e:
        print(f"Error fetching team stats: {e}")
        return pd.DataFrame()


def get_current_nfl_week():
    """Calculates the current NFL season year and week. (A new week starts on Tuesday)"""
    today = date.today(); year = today.year
    first_day_of_sept = date(year, 9, 1)
    days_to_thursday = (3 - first_day_of_sept.weekday() + 7) % 7
    kickoff_thursday = first_day_of_sept + timedelta(days=days_to_thursday)
    start_of_week_1 = kickoff_thursday - timedelta(days=2)

    if today < start_of_week_1: return year, 1
    
    current_week = ((today - start_of_week_1).days // 7) + 1
    return year, min(current_week, 18)