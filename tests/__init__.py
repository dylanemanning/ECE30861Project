import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOG_LEVEL", "0")  # Suppress logs during tests
os.environ.setdefault("LOG_FILE", "/dev/null")  # Don't write log files during tests