"""
Global configuration for the Hyperliquid Sentiment Analysis project.
"""

from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent

# Directories
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Input Files
HISTORICAL_DATA = DATA_DIR / "historical_data.csv"
FEAR_GREED_DATA = DATA_DIR / "fear_greed_index.csv"

# Random Seed
RANDOM_STATE = 42

# Figure Settings
FIG_DPI = 300
FIG_SIZE = (12, 6)

# Logging
LOG_FILE = LOGS_DIR / "project.log"