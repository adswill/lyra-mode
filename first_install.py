from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

W = 19
frac = [0.0]


def draw() -> None:
    n = min(W, max(0, int(frac[0] * W + 1e-9)))
    sys.stdout.write("\r[" + "#" * n + " " * (W - n) + "]")
    sys.stdout.flush()


def pump(proc: subprocess.Popen[bytes]) -> None:
    pct = re.compile(rb"(\d{1,3})%")
    assert proc.stdout is not None
    for chunk in iter(lambda: proc.stdout.read(256), b""):
        found = pct.findall(chunk)
        if found:
            frac[0] = max(frac[0], min(0.99, int(found[-1]) / 100.0))
        if b"Successfully installed" in chunk:
            frac[0] = 1.0


def main() -> int:
    req = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-r", req],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    th = threading.Thread(target=pump, args=(proc,), daemon=True)
    th.start()
    t0 = time.monotonic()
    draw()
    while proc.poll() is None:
        frac[0] = max(frac[0], min(0.95, (time.monotonic() - t0) / 240.0))
        draw()
        time.sleep(0.15)
    th.join(timeout=1)
    if proc.returncode == 0:
        frac[0] = 1.0
    draw()
    sys.stdout.write("\n")
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
