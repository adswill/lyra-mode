
from __future__ import annotations

import socket
import threading

from lyra.const import RF_DIAL_HZ


class RigError(RuntimeError):
    pass


class DummyRig:

    name = "Test"

    def __init__(self) -> None:
        self.connected = False
        self.frequency = int(RF_DIAL_HZ)
        self.mode = "USB"
        self.ptt = False
        self.history: list[tuple[str, object]] = []

    def connect(self) -> None:
        self.connected = True
        self.history.append(("connect", True))

    def disconnect(self) -> None:
        self.set_ptt(False)
        self.connected = False
        self.history.append(("disconnect", True))

    def set_frequency(self, hz: int) -> None:
        if not self.connected:
            raise RigError("Test rig is not connected")
        self.frequency = int(hz)
        self.history.append(("frequency", self.frequency))

    def set_mode(self, mode: str, width: int = 3000) -> None:
        if not self.connected:
            raise RigError("Test rig is not connected")
        self.mode = str(mode).upper()
        self.history.append(("mode", (self.mode, int(width))))

    def set_ptt(self, on: bool) -> None:
        self.ptt = bool(on) if self.connected else False
        self.history.append(("ptt", self.ptt))


class RigctldRig:

    name = "Hamlib rigctld"

    def __init__(self, host: str = "127.0.0.1", port: int = 4532) -> None:
        self.host = host
        self.port = int(port)
        self.connected = False
        self.ptt = False
        self._sock: socket.socket | None = None
        self._file = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self.disconnect()
        try:
            sock = socket.create_connection((self.host, self.port), timeout=3.0)
            sock.settimeout(3.0)
            self._sock = sock
            self._file = sock.makefile("rwb", buffering=0)
            self.connected = True
            
            reply = self._command("f", expect_report=False)
            int(float(reply))
        except Exception as exc:
            self.disconnect()
            raise RigError(f"Could not connect to rigctld at {self.host}:{self.port}: {exc}") from exc

    def disconnect(self) -> None:
        if self.connected:
            try:
                self.set_ptt(False)
            except Exception:
                pass
        self.connected = False
        self.ptt = False
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._file = None
        self._sock = None

    def _command(self, command: str, *, expect_report: bool = True) -> str:
        if not self.connected or self._file is None:
            raise RigError("rigctld is not connected")
        with self._lock:
            try:
                self._file.write((command.rstrip() + "\n").encode("ascii"))
                line = self._file.readline().decode("ascii", errors="replace").strip()
            except Exception as exc:
                self.connected = False
                raise RigError(f"rigctld communication failed: {exc}") from exc
        if not line:
            raise RigError("rigctld returned no response")
        if expect_report:
            if not line.startswith("RPRT "):
                raise RigError(f"Unexpected rigctld response: {line}")
            code = int(line.split()[-1])
            if code != 0:
                raise RigError(f"rigctld rejected {command.split()[0]!r} (error {code})")
        return line

    def set_frequency(self, hz: int) -> None:
        self._command(f"F {int(hz)}")

    def set_mode(self, mode: str, width: int = 3000) -> None:
        self._command(f"M {str(mode).upper()} {int(width)}")

    def set_ptt(self, on: bool) -> None:
        self._command(f"T {1 if on else 0}")
        self.ptt = bool(on)

