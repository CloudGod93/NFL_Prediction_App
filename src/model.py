# src/model.py

import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import os

class NFLGamePredictor:
    def __init__(self, model_path='nfl_model.joblib'):
        self.model = None
        self.model_path = model_path
        # Features selected from your notebook's RFE process
        self.best_features = [
            'home_passing_ypg', 'home_points_pg', 'home_passing_tds_pg',
            'home_turnovers_pg', 'away_passing_tds_pg', 'away_turnovers_pg',
            'scoring_advantage', 'turnover_advantage', 'is_playoff', 'season'
        ]
        self.load_model()

    def model_exists(self):
        """Checks if a trained model file exists."""
        return os.path.exists(self.model_path)

    def load_model(self):
        """Loads the model from the file if it exists."""
        if self.model_exists():
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
        else:
            print("No pre-trained model found.")

    def save_model(self):
        """Saves the current model to a file."""
        if self.model:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")

    def build_dataset_for_training(self, pbp, weekly, schedule):
        """Builds the full feature dataset from raw data."""
        game_records = []
        for _, game in schedule.iterrows():
            season, week, home, away = game['season'], game['week'], game['home_team'], game['away_team']
            if pd.isna(home) or pd.isna(away) or pd.isna(game.get('home_score')):
                continue

            team_features = self._create_team_features(weekly, season, week, is_training=True)
            if not team_features: continue

            game_feats = self._create_game_features(home, away, team_features, season, week, is_playoff=(game.get('game_type') != 'REG'))
            if game_feats is None: continue

            game_feats['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
            game_records.append(game_feats)
        return pd.DataFrame(game_records)

    def train(self, training_df):
        """Trains the ensemble model and resolves convergence warnings."""
        X = training_df[self.best_features].fillna(0)
        y = training_df['home_win']

        # Create a pipeline for Logistic Regression to scale data and increase iterations
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=1, random_state=42, max_iter=1000)) # Increased max_iter
        ])

        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200, random_state=42, verbosity=0, eval_metric='logloss')
        
        # Use the new pipeline in the Voting Classifier
        voting_clf = VotingClassifier(estimators=[('rf', rf), ('lr', lr_pipeline), ('xgb', xgb_model)], voting='soft')
        self.model = CalibratedClassifierCV(voting_clf, method='isotonic', cv=3)
        self.model.fit(X, y)
        self.save_model()

    def predict(self, games_df, weekly_data, season, week):
        """Makes predictions for a dataframe of upcoming games."""
        if self.model is None: raise ValueError("Model not loaded.")

        team_features = self._create_team_features(weekly_data, season, week, is_training=False)
        
        feature_list = []
        for _, game in games_df.iterrows():
            features = self._create_game_features(game['home_team'], game['away_team'], team_features, season, week)
            if features: feature_list.append(features)
        
        if not feature_list: return pd.DataFrame()

        X_pred = pd.DataFrame(feature_list)[self.best_features].fillna(0)
        home_win_probs = self.model.predict_proba(X_pred)[:, 1]

        results = games_df.copy()
        results['home_win_prob'] = home_win_probs
        results['winner'] = results.apply(lambda r: r['home_team'] if r['home_win_prob'] > 0.5 else r['away_team'], axis=1)
        results['confidence'] = results.apply(lambda r: r['home_win_prob'] if r['winner'] == r['home_team'] else 1 - r['home_win_prob'], axis=1)
        return results

    def _create_team_features(self, weekly, season, week, is_training=False):
        """Helper to create aggregated team stats."""
        if is_training:
            # For training, use data only from weeks prior to the current game's week
            data = weekly[(weekly['season'] == season) & (weekly['week'] < week)]
        else:
            # For predicting, use the entire previous season's data for context
            data = weekly[weekly['season'] == season - 1]
        
        if data.empty: return {}

        team_stats = data.groupby('recent_team').agg(
            passing_yards_pg=('passing_yards', 'mean'),
            fantasy_points_pg=('fantasy_points', 'mean'),
            passing_tds_pg=('passing_tds', 'mean'),
            interceptions_pg=('interceptions', 'mean'),
            fumbles_lost_pg=('rushing_fumbles_lost', 'mean')
        ).fillna(0)
        team_stats['turnovers_pg'] = team_stats['interceptions_pg'] + team_stats['fumbles_lost_pg']
        return team_stats.to_dict('index')

    def _create_game_features(self, home, away, team_features, season, week, is_playoff=False):
        """Helper to create the feature set for a single game."""
        if home not in team_features or away not in team_features: return None
        
        home_stats, away_stats = team_features[home], team_features[away]
        return {
            'home_passing_ypg': home_stats['passing_yards_pg'],
            'home_points_pg': home_stats['fantasy_points_pg'],
            'home_passing_tds_pg': home_stats['passing_tds_pg'],
            'home_turnovers_pg': home_stats['turnovers_pg'],
            'away_passing_tds_pg': away_stats['passing_tds_pg'],
            'away_turnovers_pg': away_stats['turnovers_pg'],
            'scoring_advantage': home_stats['fantasy_points_pg'] - away_stats['fantasy_points_pg'],
            'turnover_advantage': away_stats['turnovers_pg'] - home_stats['turnovers_pg'],
            'is_playoff': 1 if is_playoff else 0,
            'season': season,
        }