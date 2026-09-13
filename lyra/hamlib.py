
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lyra.rig import RigError


_STATUSES = ("Alpha", "Beta", "Stable", "Untested", "Bugs", "Deprecated")


def _vendor_root() -> Path:
    return Path(__file__).resolve().parent.parent / "vendor" / "hamlib"


def bundled_hamlib_dir() -> Path | None:
    machine = platform.machine().lower()
    if sys.platform == "win32":
        folders, name = ("windows",), "rigctld.exe"
    elif sys.platform == "darwin":
        name = "rigctld"
        if machine in ("arm64", "aarch64"):
            folders = ("macos-arm64", "macos")
        else:
            folders = ("macos-x86_64", "macos")
    else:
        name = "rigctld"
        if machine in ("aarch64", "arm64"):
            folders = ("linux-arm64", "linux")
        else:
            folders = ("linux-x86_64", "linux")
    for folder_name in folders:
        folder = _vendor_root() / folder_name
        for cand in (folder / name, folder / "bin" / name):
            if cand.is_file():
                return cand.parent
    return None


def _rigctld_env(exe: str) -> dict[str, str]:
    env = os.environ.copy()
    libdir = str(Path(exe).resolve().parent)
    if sys.platform.startswith("linux"):
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = libdir if not prev else libdir + os.pathsep + prev
    return env


def _rigctld_from_hint(hint: str) -> str | None:
    path = Path(hint)
    if path.is_file() and path.name.lower().startswith("rigctld"):
        return str(path)
    for name in ("rigctld.exe", "rigctld"):
        cand = path / name if path.is_dir() else path / "bin" / name
        if cand.is_file():
            return str(cand)
    return None


def find_rigctld(hint: str = "") -> str | None:
    hint = (hint or "").strip().strip('"')
    if hint:
        found = _rigctld_from_hint(hint)
        if found:
            return found
    bundled = bundled_hamlib_dir()
    if bundled is not None:
        name = "rigctld.exe" if sys.platform == "win32" else "rigctld"
        return str(bundled / name)
    found = shutil.which("rigctld")
    if found:
        return found
    if sys.platform == "win32":
        roots = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("LOCALAPPDATA", "")),
        ]
        for root in roots:
            if not root:
                continue
            for match in root.glob("hamlib*/bin/rigctld.exe"):
                return str(match)
            for match in root.glob("hamlib-*/bin/rigctld.exe"):
                return str(match)
    return None


def list_serial_ports() -> list[str]:
    if sys.platform == "win32":
        found: list[str] = []
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(65536)
            n = ctypes.windll.kernel32.QueryDosDeviceW(None, buf, 65536)
            if n:
                for name in buf[:n].split("\x00"):
                    if name.startswith("COM") and name[3:].isdigit():
                        found.append(name)
        except Exception:
            found = [f"COM{i}" for i in range(1, 16)]
        if not found:
            found = [f"COM{i}" for i in range(1, 16)]
        return sorted(set(found), key=lambda s: int(s[3:]))
    paths: list[str] = []
    for pat in (
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/tty.usb*",
        "/dev/cu.usb*",
        "/dev/cu.SLAB*",
        "/dev/ttyS*",
    ):
        paths.extend(glob.glob(pat))
    return sorted(set(paths))


def list_models(rigctld: str) -> list[tuple[int, str]]:
    try:
        raw = subprocess.check_output(
            [rigctld, "-l"],
            stderr=subprocess.STDOUT,
            timeout=12,
            cwd=str(Path(rigctld).parent),
            env=_rigctld_env(rigctld),
        )
    except Exception as exc:
        raise RigError(f"Could not list hamlib radios: {exc}") from exc
    models: list[tuple[int, str]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        status_i = next((i for i, p in enumerate(parts) if p in _STATUSES), -1)
        if status_i < 3:
            continue
        rid = int(parts[0])
        mfg = parts[1]
        model = " ".join(parts[2 : status_i - 1]).strip() or parts[2]
        models.append((rid, f"{rid}  {mfg} {model}"))
    return models


class LocalRigctld:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.port = 4532

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(
        self,
        *,
        exe: str,
        model: int,
        device: str,
        baud: int,
        port: int,
        conf: str = "",
    ) -> None:
        self.stop()
        cmd = [
            exe,
            "-m",
            str(int(model)),
            "-r",
            str(device),
            "-s",
            str(int(baud)),
            "-T",
            "127.0.0.1",
            "-t",
            str(int(port)),
        ]
        conf = conf.strip()
        if conf:
            cmd.extend(["-C", conf])
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=str(Path(exe).parent),
                env=_rigctld_env(exe),
            )
        except Exception as exc:
            raise RigError(f"Could not start rigctld: {exc}") from exc
        self.port = int(port)
        time.sleep(0.45)
        if self.proc.poll() is not None:
            err = b""
            if self.proc.stderr is not None:
                err = self.proc.stderr.read() or b""
            text = err.decode("utf-8", "replace").strip() or f"exit {self.proc.returncode}"
            self.proc = None
            raise RigError(f"rigctld failed: {text}")

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception:
                pass
