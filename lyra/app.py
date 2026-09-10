
from __future__ import annotations

import sys


def main() -> int:
    from lyra.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
