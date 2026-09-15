from __future__ import annotations

import sys

from lyra import network


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    day = args[0] if args else "today"
    try:
        stamp, n = network.unique(network.DEFAULT_URL, day)
    except network.NetworkError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{stamp} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
