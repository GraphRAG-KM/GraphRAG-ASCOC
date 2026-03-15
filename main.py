import importlib
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

main_entry = importlib.import_module("app_main").main_entry

if __name__ == "__main__":
    main_entry()
