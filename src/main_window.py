# src/main_window.py

import sys
import os
import pandas as pd
import nfl_data_py as nfl
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QPushButton, QListWidget, QListWidgetItem,
                             QLabel, QProgressBar, QTabWidget, QSpacerItem, QSizePolicy,
                             QScrollArea, QStackedWidget, QDialog, QGridLayout)
from PyQt6.QtGui import QIcon, QPixmap, QFont, QColor
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from src.data_handler import get_schedule, get_current_nfl_week, get_training_data, get_team_stats
from src.model import NFLGamePredictor
from src.team_data import TEAM_NAMES, TEAM_COLORS

# --- NEW: Scouting Report Dialog ---
class ScoutingReportDialog(QDialog):
    def __init__(self, away_abbr, home_abbr, year, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scouting Report")
        self.setMinimumWidth(600)

        # Fetch data
        team_stats_df = get_team_stats(year)
        away_stats = team_stats_df.loc[away_abbr] if away_abbr in team_stats_df.index else None
        home_stats = team_stats_df.loc[home_abbr] if home_abbr in team_stats_df.index else None

        # UI Layout
        layout = QGridLayout(self)
        
        # Headers
        away_color, home_color = TEAM_COLORS.get(away_abbr, ('#FFF', '#000')), TEAM_COLORS.get(home_abbr, ('#FFF', '#000'))
        away_header = self.create_header(away_abbr, away_color)
        home_header = self.create_header(home_abbr, home_color)
        layout.addWidget(away_header, 0, 0)
        layout.addWidget(QLabel("vs"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(home_header, 0, 2)
        
        # Stats
        stats_to_show = {
            "Record": lambda s: f"{s.get('wins', 0)}-{s.get('losses', 0)}-{s.get('ties', 0)}",
            "Points For/Game": lambda s: f"{s.get('points_for', 0) / s.get('games_played', 1):.1f}",
            "Pass Yds/Game": lambda s: f"{s.get('pass_yds', 0) / s.get('games_played', 1):.1f}",
            "Rush Yds/Game": lambda s: f"{s.get('rush_yds', 0) / s.get('games_played', 1):.1f}",
            "Turnovers": lambda s: f"{s.get('turnovers', 0)}"
        }
        
        row = 1
        for title, func in stats_to_show.items():
            away_val = func(away_stats) if away_stats is not None else "N/A"
            home_val = func(home_stats) if home_stats is not None else "N/A"
            layout.addWidget(QLabel(away_val), row, 0, Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(QLabel(f"<b>{title}</b>"), row, 1, Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(QLabel(home_val), row, 2, Qt.AlignmentFlag.AlignCenter)
            row += 1

    def create_header(self, team_abbr, colors):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        logo = QLabel(); logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name = QLabel(TEAM_NAMES.get(team_abbr, team_abbr)); name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        for ext in ['png', 'webp', 'jpg', 'jpeg']:
            logo_file = os.path.join('assets', f"{team_abbr.upper()}.{ext}")
            if os.path.exists(logo_file):
                pixmap = QPixmap(logo_file).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo.setPixmap(pixmap); break
        
        widget.setStyleSheet(f"background-color: {colors[0]}; color: {colors[1]}; border-radius: 8px;")
        name.setStyleSheet(f"background-color: transparent; color: {colors[1]}; font-weight: bold; font-size: 16px;")
        layout.addWidget(logo); layout.addWidget(name)
        return widget


# --- Custom Widget for Displaying Games with Logos in the List ---
class GameListItem(QWidget):
    # ... (same as before) ...
    def __init__(self, away_team_abbr, home_team_abbr, asset_path='assets'):
        super().__init__(); layout = QHBoxLayout(); layout.setContentsMargins(5, 5, 5, 5)
        def create_logo_label(abbr):
            logo_label = QLabel()
            for ext in ['png', 'webp', 'jpg', 'jpeg']:
                logo_file = os.path.join(asset_path, f"{abbr.upper()}.{ext}")
                if os.path.exists(logo_file):
                    pixmap = QPixmap(logo_file).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    logo_label.setPixmap(pixmap); break
            return logo_label
        away_logo, home_logo = create_logo_label(away_team_abbr), create_logo_label(home_team_abbr)
        away_name = QLabel(f"{TEAM_NAMES.get(away_team_abbr, away_team_abbr):<20}")
        home_name = QLabel(f"{TEAM_NAMES.get(home_team_abbr, home_team_abbr):<20}")
        at_label = QLabel("@"); at_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(away_logo); layout.addWidget(away_name); layout.addWidget(at_label)
        layout.addWidget(home_logo); layout.addWidget(home_name)
        layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.setLayout(layout)

# --- Custom Widget for Displaying Prediction Results ---
class PredictionResultWidget(QWidget):
    # ... (updated to include Upset/Lock logic) ...
    def __init__(self, away_abbr, home_abbr, winner_abbr, confidence, away_wins, home_wins, asset_path='assets'):
        super().__init__()
        self.setStyleSheet("""
            QWidget { background-color: #2c3136; border-radius: 8px; }
            QLabel { font-size: 16px; color: #e0e0e0; }
            #winner_name { font-weight: bold; color: #4CAF50; }
            #tag { font-size: 10px; font-weight: bold; padding: 2px 5px; border-radius: 3px; }
            #lock_tag { background-color: #4CAF50; color: white; }
            #upset_tag { background-color: #FFC107; color: black; }
        """)
        main_layout = QVBoxLayout(self); main_layout.setSpacing(10)
        
        # Top row for matchup and tags
        top_layout = QHBoxLayout()
        away_logo = self.create_logo_label(away_abbr, winner_abbr, asset_path)
        away_name = QLabel(TEAM_NAMES.get(away_abbr, away_abbr)); away_name.setObjectName("winner_name" if away_abbr == winner_abbr else "")
        at_label = QLabel("@")
        home_logo = self.create_logo_label(home_abbr, winner_abbr, asset_path)
        home_name = QLabel(TEAM_NAMES.get(home_abbr, home_abbr)); home_name.setObjectName("winner_name" if home_abbr == winner_abbr else "")
        
        top_layout.addWidget(away_logo); top_layout.addWidget(away_name, 1)
        top_layout.addSpacerItem(QSpacerItem(10, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        top_layout.addWidget(at_label)
        top_layout.addSpacerItem(QSpacerItem(10, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        top_layout.addWidget(home_logo); top_layout.addWidget(home_name, 1)

        # Upset/Lock Logic
        tag_label = QLabel()
        is_upset = (winner_abbr == away_abbr and away_wins < home_wins) or \
                   (winner_abbr == home_abbr and home_wins < away_wins)
        if confidence > 0.75:
            tag_label.setText("🔒 LOCK"); tag_label.setObjectName("lock_tag")
        elif is_upset:
            tag_label.setText("🚨 UPSET"); tag_label.setObjectName("upset_tag")
        if tag_label.text(): top_layout.addWidget(tag_label, 0, Qt.AlignmentFlag.AlignRight)

        # Bottom row for confidence bar and spread
        bottom_layout = QHBoxLayout()
        confidence_label = QLabel(f"Confidence: <b>{confidence:.1%}</b>")
        bar_container = QWidget(); bar_container.setFixedHeight(10)
        bar_layout = QHBoxLayout(bar_container); bar_layout.setContentsMargins(0,0,0,0)
        bar_color = "#4CAF50" if confidence > 0.7 else "#FFC107" if confidence > 0.6 else "#013369"
        confidence_bar = QWidget(); confidence_bar.setStyleSheet(f"background-color: {bar_color}; border-radius: 4px;")
        bar_layout.addWidget(confidence_bar, int(confidence * 100)); bar_layout.addStretch(100 - int(confidence * 100))
        
        # Placeholder for Vegas Spread
        spread_label = QLabel("Spread: -7.5"); spread_label.setStyleSheet("color: #aaa; font-style: italic;")

        bottom_layout.addWidget(confidence_label); bottom_layout.addWidget(bar_container, 1)
        bottom_layout.addSpacerItem(QSpacerItem(20, 20)); bottom_layout.addWidget(spread_label)
        main_layout.addLayout(top_layout); main_layout.addLayout(bottom_layout)

    def create_logo_label(self, team_abbr, winner_abbr, asset_path): # ... (same as before) ...
        logo_label = QLabel(); logo_label.setFixedSize(50, 50); opacity = "1.0" if team_abbr == winner_abbr else "0.3"
        logo_label.setStyleSheet(f"opacity: {opacity};")
        for ext in ['png', 'webp', 'jpg', 'jpeg']:
            logo_file = os.path.join(asset_path, f"{team_abbr.upper()}.{ext}")
            if os.path.exists(logo_file):
                pixmap = QPixmap(logo_file).scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(pixmap); break
        return logo_label

# --- Loading Screen Widget ---
class LoadingWidget(QWidget): # ... (same as before) ...
    def __init__(self, text="Loading..."):
        super().__init__(); layout = QVBoxLayout(self); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QLabel(text); self.label.setFont(QFont("Segoe UI", 18)); self.label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(self.label)
    def setText(self, text): self.label.setText(text)

# --- Worker Threads ---
class TrainingWorker(QThread): # ... (same as before) ...
    finished = pyqtSignal(); progress = pyqtSignal(int, str)
    def __init__(self, predictor, start_year, end_year):
        super().__init__(); self.predictor, self.start_year, self.end_year = predictor, start_year, end_year
    def run(self):
        self.progress.emit(10, f"Downloading all data from {self.start_year}-{self.end_year}...")
        pbp, weekly, schedule = get_training_data(self.start_year, self.end_year)
        if pbp is None or weekly is None or schedule is None:
            self.progress.emit(100, "❌ Error: Failed to download data. Check internet connection."); self.finished.emit(); return
        self.progress.emit(40, "Building feature dataset..."); training_df = self.predictor.build_dataset_for_training(pbp, weekly, schedule)
        self.progress.emit(70, "Training new ensemble model..."); self.predictor.train(training_df)
        self.progress.emit(100, "✅ Training complete! Model saved."); self.finished.emit()

class PredictionWorker(QThread): # ... (same as before) ...
    finished = pyqtSignal(pd.DataFrame); progress = pyqtSignal(str)
    def __init__(self, predictor, games, year, week):
        super().__init__(); self.predictor, self.games, self.year, self.week = predictor, games, year, week
    def run(self):
        self.progress.emit("Loading historical data for context...")
        try:
            weekly_data = nfl.import_weekly_data(years=range(2015, self.year))
            if weekly_data.empty: self.progress.emit("Error: Could not load historical weekly data."); self.finished.emit(pd.DataFrame()); return
        except Exception as e: self.progress.emit(f"Error fetching data: {e}"); self.finished.emit(pd.DataFrame()); return
        self.progress.emit("Generating predictions..."); results = self.predictor.predict(self.games, weekly_data, self.year, self.week)
        self.finished.emit(results)

# --- Main Application Window ---
class MainWindow(QMainWindow):
    # ... (updated to handle double clicks for scouting reports) ...
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NFL Game Predictor"); self.setGeometry(100, 100, 900, 700)
        for ext in ['png', 'webp', 'jpg', 'jpeg']:
            icon_path = os.path.join('assets', f'logo.{ext}');
            if os.path.exists(icon_path): self.setWindowIcon(QIcon(icon_path)); break
        
        self.predictor = NFLGamePredictor()
        
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget(); self.predict_tab, self.train_tab = QWidget(), QWidget()
        self.tabs.addTab(self.predict_tab, "🏈 Predict Games"); self.tabs.addTab(self.train_tab, "⚙️ Train Model")
        main_layout.addWidget(self.tabs)
        
        self.setup_predict_ui(); self.setup_train_ui(); self.apply_stylesheet()

    def setup_predict_ui(self):
        layout = QVBoxLayout(self.predict_tab)
        top_layout, current_year, current_week = QHBoxLayout(), *get_current_nfl_week()
        
        self.year_combo = QComboBox(); self.year_combo.addItems([str(y) for y in range(2015, current_year + 2)])
        self.year_combo.setCurrentText(str(current_year))
        self.week_combo = QComboBox(); self.week_combo.addItems([f"Week {w}" for w in range(1, 19)])
        self.week_combo.setCurrentIndex(current_week - 1)
        self.load_schedule_button = QPushButton("Load Schedule")
        top_layout.addWidget(QLabel("Year:")); top_layout.addWidget(self.year_combo); top_layout.addWidget(QLabel("Week:"))
        top_layout.addWidget(self.week_combo); top_layout.addWidget(self.load_schedule_button)

        self.main_stack = QStackedWidget()
        self.game_list_widget = QListWidget(); self.game_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.game_list_widget.itemDoubleClicked.connect(self.show_scouting_report) # Connect double click
        self.loading_widget = LoadingWidget("Predicting Games...")
        self.results_area = QScrollArea(); self.results_area.setWidgetResizable(True)
        self.main_stack.addWidget(self.game_list_widget); self.main_stack.addWidget(self.loading_widget); self.main_stack.addWidget(self.results_area)

        pred_layout = QHBoxLayout()
        self.predict_week_button, self.predict_selected_button = QPushButton("Predict Entire Week"), QPushButton("Predict Selected Games")
        pred_layout.addWidget(self.predict_week_button); pred_layout.addWidget(self.predict_selected_button)

        layout.addLayout(top_layout); layout.addWidget(self.main_stack, 1); layout.addLayout(pred_layout)
        self.load_schedule_button.clicked.connect(self.update_schedule_display)
        self.predict_week_button.clicked.connect(self.predict_week); self.predict_selected_button.clicked.connect(self.predict_selected)
        
        self.update_schedule_display(); self.update_prediction_button_state()

    def setup_train_ui(self): # ... (same as before) ...
        layout = QVBoxLayout(self.train_tab)
        info_label = QLabel("Retrain the prediction model using the latest complete season data. This downloads several years of data and can take 5-10 minutes.")
        info_label.setWordWrap(True)
        self.train_button = QPushButton(); self.model_status_label = QLabel(); self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_training_ui_state()
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.progress_label = QLabel(); self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label); layout.addWidget(self.model_status_label); layout.addWidget(self.train_button)
        layout.addWidget(self.progress_bar); layout.addWidget(self.progress_label)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.train_button.clicked.connect(self.start_training)

    def update_schedule_display(self): # ... (same as before) ...
        self.main_stack.setCurrentWidget(self.game_list_widget); self.statusBar().showMessage("Loading schedule...")
        self.game_list_widget.clear(); year, week = int(self.year_combo.currentText()), self.week_combo.currentIndex() + 1
        self.schedule = get_schedule(year, week)
        if self.schedule.empty: self.statusBar().showMessage(f"No schedule found for {year} Week {week}.", 5000); return
        for index, row in self.schedule.iterrows():
            item = QListWidgetItem(self.game_list_widget)
            game_widget = GameListItem(row['away_team'], row['home_team'])
            item.setSizeHint(game_widget.sizeHint()); self.game_list_widget.setItemWidget(item, game_widget)
            item.setData(Qt.ItemDataRole.UserRole, index)
        self.statusBar().showMessage("Schedule loaded.", 3000)

    def show_scouting_report(self, item):
        game_index = item.data(Qt.ItemDataRole.UserRole)
        game_data = self.schedule.loc[game_index]
        year = int(self.year_combo.currentText())
        dialog = ScoutingReportDialog(game_data['away_team'], game_data['home_team'], year, self)
        dialog.exec()
    
    def start_training(self): # ... (same as before) ...
        self.train_button.setEnabled(False); self.progress_bar.setVisible(True)
        self.training_worker = TrainingWorker(self.predictor, 2015, get_current_nfl_week()[0] - 1)
        self.training_worker.progress.connect(self.update_training_progress); self.training_worker.finished.connect(self.training_finished)
        self.training_worker.start()

    def update_training_progress(self, value, text): # ... (same as before) ...
        self.progress_bar.setValue(value); self.progress_label.setText(text)
    def training_finished(self): # ... (same as before) ...
        self.predictor.load_model(); self.update_training_ui_state(); self.update_prediction_button_state()

    def update_training_ui_state(self): # ... (same as before) ...
        if self.predictor.model_exists():
            self.model_status_label.setText("✅ Model is trained and ready."); self.train_button.setText("Retrain Model")
        else:
            self.model_status_label.setText("⚠️ No model found. Please train the initial model."); self.train_button.setText("Train Initial Model")
        self.train_button.setEnabled(True)

    def update_prediction_button_state(self): # ... (same as before) ...
        is_enabled = self.predictor.model_exists()
        self.predict_week_button.setEnabled(is_enabled); self.predict_selected_button.setEnabled(is_enabled)

    def predict_week(self): # ... (same as before) ...
        if hasattr(self, 'schedule') and not self.schedule.empty: self.run_predictions(self.schedule)
    def predict_selected(self): # ... (same as before) ...
        indices = [self.game_list_widget.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.game_list_widget.count()) if self.game_list_widget.item(i).isSelected()]
        if indices: self.run_predictions(self.schedule.loc[indices])

    def run_predictions(self, games): # ... (same as before) ...
        if not self.predictor.model_exists():
            self.statusBar().showMessage("❌ Error: Model not trained. Go to 'Train Model' tab."); return
        self.set_prediction_buttons_enabled(False); self.main_stack.setCurrentWidget(self.loading_widget)
        self.prediction_worker = PredictionWorker(self.predictor, games, int(self.year_combo.currentText()), self.week_combo.currentIndex() + 1)
        self.prediction_worker.finished.connect(self.display_results); self.prediction_worker.progress.connect(self.statusBar().showMessage)
        self.prediction_worker.start()

    def display_results(self, results_df): # ... (updated to get win data for upset logic) ...
        results_container = QWidget(); results_layout = QVBoxLayout(results_container); results_layout.setSpacing(15)
        year = int(self.year_combo.currentText())
        team_stats_df = get_team_stats(year)

        if results_df.empty:
            results_layout.addWidget(QLabel("Could not generate predictions. Team data might be missing or there was an error."))
        else:
            for _, row in results_df.iterrows():
                away_wins = team_stats_df.loc[row['away_team']]['wins'] if row['away_team'] in team_stats_df.index else 0
                home_wins = team_stats_df.loc[row['home_team']]['wins'] if row['home_team'] in team_stats_df.index else 0
                result_card = PredictionResultWidget(row['away_team'], row['home_team'], row['winner'], row['confidence'], away_wins, home_wins)
                results_layout.addWidget(result_card)
        
        self.results_area.setWidget(results_container)
        self.main_stack.setCurrentWidget(self.results_area)
        self.set_prediction_buttons_enabled(True); self.statusBar().showMessage("Predictions complete!", 3000)
    
    def set_prediction_buttons_enabled(self, enabled): # ... (same as before) ...
        self.predict_week_button.setEnabled(enabled); self.predict_selected_button.setEnabled(enabled)

    def apply_stylesheet(self): self.setStyleSheet("""
        QWidget { background-color: #1a1d21; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }
        QTabWidget::pane { border: 1px solid #4a4a4a; }
        QTabBar::tab { background: #2c3136; color: #e0e0e0; padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px; font-weight: bold; }
        QTabBar::tab:selected { background: #013369; color: white; }
        QPushButton { background-color: #013369; color: white; border-radius: 5px; padding: 10px; font-size: 14px; font-weight: bold; border: 1px solid #D50A0A; }
        QPushButton:hover { background-color: #004b8d; }
        QPushButton:disabled { background-color: #4a4a4a; color: #888; border: 1px solid #555; }
        QComboBox { background-color: #2c3136; border: 1px solid #555; padding: 5px; border-radius: 3px; }
        QListWidget { background-color: #262a2e; border: 1px solid #4a4a4a; border-radius: 5px; }
        QListWidget::item { border-bottom: 1px solid #3a3f44; }
        QListWidget::item:selected { background-color: #013369; border: 1px solid #D50A0A; }
        QScrollArea { border: none; background-color: #1a1d21; }
        QLabel { font-size: 14px; }
        QProgressBar { border: 1px solid #555; border-radius: 5px; text-align: center; color: white; font-weight: bold; }
        QProgressBar::chunk { background-color: #D50A0A; }
        QStatusBar { font-size: 12px; }
    """)