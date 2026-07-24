"""
Utility functions used across the project.
"""

import logging
from pathlib import Path

from config import LOG_FILE

# Create log folder if it doesn't exist
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("Hyperliquid")


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def log(message: str):
    logger.info(message)