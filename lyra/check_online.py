from __future__ import annotations

import sys

from lyra import network


def main() -> int:
    try:
        print(network.count(network.DEFAULT_URL))
    except network.NetworkError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
