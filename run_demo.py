from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from urban_forest_ai.cli import main


if __name__ == "__main__":
    main()
