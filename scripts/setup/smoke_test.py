"""Command-line entry point for the short tracking pipeline check."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mot.tracking.smoke import main


if __name__ == "__main__":
    main()
