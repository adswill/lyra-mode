
from __future__ import annotations

import math
import threading

import numpy as np

from lyra.const import SAMPLE_RATE

VIRTUAL_HINTS = (
    "blackhole",
    "vb-audio",
    "cable",
    "voicemeeter",
    "soundflower",
    "loopback",
    "virtual",
    "stereo mix",
)


def load_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", "sounddevice"]
        )
        import sounddevice as sd

        return sd


def list_inputs(sd=None) -> list[tuple[int, str, int]]:
    sd = sd or load_sounddevice()
    out = []
    for i, d in enumerate(sd.query_devices()):
        ch = int(d.get("max_input_channels") or 0)
        if ch < 1:
            continue
        out.append((i, str(d["name"]), ch))
    return out


def default_input_index(sd=None) -> int | None:
    sd = sd or load_sounddevice()
    devices = list_inputs(sd)
    if not devices:
        return None
    for i, name, _ch in devices:
        n = name.lower()
        if any(h in n for h in VIRTUAL_HINTS):
            if "2ch" in n or "2 ch" in n or "cable output" in n or "output" in n:
                return i
    for i, name, _ch in devices:
        if any(h in name.lower() for h in VIRTUAL_HINTS):
            return i
    try:
        raw = sd.default.device
        idx = int(raw[0] if isinstance(raw, (list, tuple)) else raw)
        if idx >= 0:
            return idx
    except Exception:
        pass
    return devices[0][0]


def _mono(frames: np.ndarray) -> np.ndarray:
    x = np.asarray(frames)
    if x.ndim == 1 or x.shape[1] == 1:
        return np.asarray(x, dtype=np.float32).reshape(-1)
    l = x[:, 0]
    r = x[:, 1]
    e_l = float(np.mean(l * l))
    e_r = float(np.mean(r * r))
    return np.asarray(r if e_r > 4.0 * e_l else l, dtype=np.float32)


def _resample_48k(x: np.ndarray, src_hz: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if int(src_hz) == SAMPLE_RATE or len(x) < 16:
        return x
    from scipy.signal import resample_poly

    g = math.gcd(int(src_hz), SAMPLE_RATE)
    return np.asarray(resample_poly(x, SAMPLE_RATE // g, int(src_hz) // g), dtype=np.float64)


class AudioTap:

    def __init__(self, device: int | None, seconds: float = 6.0):
        self.device = device
        self.seconds = seconds
        self._stream = None
        self._sd = None
        self.channels = 1
        self.name = "default"
        self.capture_rate = SAMPLE_RATE
        self.overruns = 0
        self._lock = threading.Lock()
        self._ring = np.zeros(1, dtype=np.float32)
        self._w = 0
        self._n = 0

    def start(self) -> None:
        sd = load_sounddevice()
        self._sd = sd
        dev = self.device if self.device is not None else default_input_index(sd)
        if dev is None:
            raise RuntimeError("No input audio device found.")
        info = sd.query_devices(dev)
        self.name = str(info["name"])
        self.channels = min(2, int(info["max_input_channels"]))
        self.device = int(dev)
        rate = int(round(float(info.get("default_samplerate") or SAMPLE_RATE)))
        if rate < 8000:
            rate = SAMPLE_RATE
        try:
            sd.check_input_settings(
                device=dev,
                samplerate=rate,
                channels=self.channels,
                dtype="float32",
            )
        except Exception:
            rate = SAMPLE_RATE
        self.capture_rate = rate
        n = max(int(self.seconds * self.capture_rate), 4096)
        self._ring = np.zeros(n, dtype=np.float32)
        self._w = 0
        self._n = 0
        self.overruns = 0

        def callback(indata, frames, time_info, status) -> None:
            if status:
                self.overruns += 1
            mono = _mono(indata)
            k = int(mono.shape[0])
            if k < 1:
                return
            buf = self._ring
            cap = int(buf.shape[0])
            with self._lock:
                w = self._w
                if w + k <= cap:
                    buf[w : w + k] = mono
                else:
                    a = cap - w
                    buf[w:] = mono[:a]
                    buf[: k - a] = mono[a:]
                self._w = (w + k) % cap
                self._n = min(cap, self._n + k)

        self._stream = sd.InputStream(
            device=dev,
            samplerate=self.capture_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=2048,
            latency=2.0,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            cap = int(self._ring.shape[0])
            n = int(self._n)
            w = int(self._w)
            if n < int(0.25 * self.capture_rate):
                return None
            if n < cap:
                native = np.array(self._ring[:n], dtype=np.float64)
            else:
                native = np.empty(cap, dtype=np.float64)
                native[: cap - w] = self._ring[w:]
                native[cap - w :] = self._ring[:w]
        audio = _resample_48k(native, self.capture_rate)
        if len(audio) < SAMPLE_RATE:
            return None
        return audio

    def latest(self, n: int) -> np.ndarray | None:
        need = max(int(n), 512)
        if self.capture_rate != SAMPLE_RATE:
            need = int(round(need * self.capture_rate / SAMPLE_RATE))
        with self._lock:
            cap = int(self._ring.shape[0])
            have = int(self._n)
            w = int(self._w)
            if have < 512:
                return None
            take = min(need, have)
            native = np.empty(take, dtype=np.float64)
            if have < cap:
                native[:] = self._ring[have - take : have]
            else:
                start = (w - take) % cap
                if start + take <= cap:
                    native[:] = self._ring[start : start + take]
                else:
                    a = cap - start
                    native[:a] = self._ring[start:]
                    native[a:] = self._ring[: take - a]
        audio = _resample_48k(native, self.capture_rate)
        return audio[-n:] if len(audio) >= n else audio

    def rms(self) -> float:
        audio = self.latest(SAMPLE_RATE)
        if audio is None:
            return 0.0
        return float(np.sqrt(np.mean(audio**2)))
