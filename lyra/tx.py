
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from lyra.capture import load_sounddevice
from lyra.codec import frame_iq
from lyra.const import CHANNELS, PIN_HZ, SAMPLE_RATE


def list_outputs(sd=None) -> list[tuple[int, str, int]]:
    sd = sd or load_sounddevice()
    out = []
    for i, device in enumerate(sd.query_devices()):
        channels = int(device.get("max_output_channels") or 0)
        if channels > 0:
            out.append((i, str(device["name"]), channels))
    return out


def default_output_index(sd=None) -> int | None:
    sd = sd or load_sounddevice()
    devices = list_outputs(sd)
    if not devices:
        return None
    try:
        raw = sd.default.device
        idx = int(raw[1] if isinstance(raw, (list, tuple)) else raw)
        if idx >= 0 and any(i == idx for i, _name, _ch in devices):
            return idx
    except Exception:
        pass
    return devices[0][0]


def _resample(audio: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    src = int(src_hz)
    dst = int(dst_hz)
    if src == dst or len(x) < 16:
        return x
    g = math.gcd(src, dst)
    return np.asarray(resample_poly(x, dst // g, src // g), dtype=np.float32)


def _frames_for_output(audio: np.ndarray, channels: int, play_hz: int) -> np.ndarray:
    mono = _resample(audio, SAMPLE_RATE, play_hz)
    nch = max(1, int(channels))
    if nch == 1:
        return np.ascontiguousarray(mono.reshape(-1, 1))
    return np.ascontiguousarray(np.repeat(mono.reshape(-1, 1), nch, axis=1))


def _output_attempts(sd, device: int) -> list[dict]:
    info = sd.query_devices(int(device))
    max_ch = max(1, int(info.get("max_output_channels") or 1))
    native = int(round(float(info.get("default_samplerate") or SAMPLE_RATE)))
    channels = []
    if max_ch >= 2:
        channels.append(2)
    if 1 not in channels:
        channels.append(1 if max_ch >= 1 else max_ch)
    if max_ch not in channels and max_ch > 2:
        channels.append(max_ch)
    rates = []
    for rate in (SAMPLE_RATE, native):
        if rate > 0 and rate not in rates:
            rates.append(rate)
    attempts = []
    for latency in (None, "high", "low"):
        for rate in rates:
            for nch in channels:
                kw = {
                    "device": int(device),
                    "samplerate": int(rate),
                    "channels": int(nch),
                    "dtype": "float32",
                    "blocksize": 0 if latency is None else 512,
                }
                if latency is not None:
                    kw["latency"] = latency
                attempts.append(kw)
    return attempts


def _open_output_stream(sd, device: int, audio: np.ndarray):
    last = None
    seen: set[tuple] = set()
    for kw in _output_attempts(sd, device):
        key = (kw["samplerate"], kw["channels"], kw.get("latency"), kw["blocksize"])
        if key in seen:
            continue
        seen.add(key)
        stream = None
        try:
            stream = sd.OutputStream(**kw)
            stream.start()
            frames = _frames_for_output(audio, kw["channels"], kw["samplerate"])
            return stream, frames
        except Exception as exc:
            last = exc
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
    hint = " macOS often needs stereo output, not mono." if sys.platform == "darwin" else ""
    raise RuntimeError(f"Could not open audio output.{hint} {last}") from last


TUNE_S = 30.0


def build_tune_audio(*, channel: int, level: float = 0.70) -> np.ndarray:
    idx = int(channel) - 1
    if idx < 0 or idx >= len(CHANNELS):
        raise ValueError(f"channel must be 1–{len(CHANNELS)}")
    fa, fb = CHANNELS[idx]
    n = int(round(TUNE_S * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    audio = (np.sin(2.0 * np.pi * fa * t) + np.sin(2.0 * np.pi * fb * t)).astype(np.float32)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    audio = np.asarray(audio * (float(level) / peak), dtype=np.float32)
    ramp_n = min(int(round(0.005 * SAMPLE_RATE)), n // 4)
    if ramp_n > 1:
        phase = np.linspace(0.0, 0.5 * np.pi, ramp_n, dtype=np.float64)
        fade = np.sin(phase).astype(np.float32) ** 2
        audio[:ramp_n] *= fade
        audio[-ramp_n:] *= fade[::-1]
    return audio


def build_tx_audio(
    info_bits,
    *,
    mode: str,
    channel: int,
    level: float = 0.70,
) -> np.ndarray:
    idx = int(channel) - 1
    if idx < 0 or idx >= len(CHANNELS):
        raise ValueError(f"channel must be 1–{len(CHANNELS)}")
    mode = mode.upper()
    iq = frame_iq(info_bits, mode=mode)
    center = 0.5 * (CHANNELS[idx][0] + CHANNELS[idx][1])
    shift = center - PIN_HZ
    if abs(shift) > 0.01:
        t = np.arange(len(iq), dtype=np.float64) / SAMPLE_RATE
        iq = iq * np.exp(1j * 2.0 * np.pi * shift * t)
    audio = np.real(iq).astype(np.float32)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    audio = np.asarray(audio * (float(level) / peak), dtype=np.float32)
    
    
    ramp_n = min(int(round(0.005 * SAMPLE_RATE)), len(audio) // 4)
    if ramp_n > 1:
        phase = np.linspace(0.0, 0.5 * np.pi, ramp_n, dtype=np.float64)
        audio[:ramp_n] *= np.sin(phase).astype(np.float32) ** 2
        active = np.flatnonzero(np.abs(audio) > 1e-8)
        if len(active):
            end = int(active[-1]) + 1
            start = max(ramp_n, end - ramp_n)
            n_out = end - start
            if n_out > 1:
                phase_out = np.linspace(0.0, 0.5 * np.pi, n_out, dtype=np.float64)
                audio[start:end] *= np.cos(phase_out).astype(np.float32) ** 2
    return audio


def write_tx_wav(path: Path, audio: np.ndarray) -> Path:
    path = Path(path)
    pcm = np.clip(np.asarray(audio) * 32767.0, -32767, 32767).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, pcm)
    return path


class TxSession:

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stream = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        audio: np.ndarray,
        *,
        output_device: int | None,
        rig,
        callback: Callable[[str, bool], None],
    ) -> None:
        if self.busy:
            raise RuntimeError("A transmission is already active")
        self._stop.clear()
        frames = None
        stream = None
        if output_device is not None:
            sd = load_sounddevice()
            stream, frames = _open_output_stream(
                sd, int(output_device), np.asarray(audio, dtype=np.float32)
            )
            with self._lock:
                self._stream = stream
        self._thread = threading.Thread(
            target=self._run,
            args=(np.asarray(audio, dtype=np.float32), frames, stream, rig, callback),
            daemon=True,
        )
        self._thread.start()

    def _run(self, audio, frames, stream, rig, callback) -> None:
        ok = False
        message = "Transmission stopped"
        try:
            rig.set_ptt(True)
            callback("PTT ON", False)
            if self._stop.wait(0.02):
                return
            if stream is None or frames is None:
                end = time.monotonic() + len(audio) / SAMPLE_RATE
                while not self._stop.is_set() and time.monotonic() < end:
                    time.sleep(0.02)
            else:
                step = int(stream.blocksize or 0) or 1024
                nch = int(frames.shape[1])
                for pos in range(0, len(frames), step):
                    if self._stop.is_set():
                        break
                    chunk = frames[pos : pos + step]
                    if len(chunk) < step:
                        pad = np.zeros((step, nch), dtype=np.float32)
                        pad[: len(chunk)] = chunk
                        chunk = pad
                    stream.write(chunk)
                stream.stop()
                stream.close()
                with self._lock:
                    self._stream = None
            ok = not self._stop.is_set()
            message = "Transmission complete" if ok else "Transmission stopped"
        except Exception as exc:
            message = f"TX error: {type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                stream = self._stream
                self._stream = None
            if stream is not None:
                try:
                    stream.abort()
                    stream.close()
                except Exception:
                    pass
            try:
                rig.set_ptt(False)
            except Exception as exc:
                ok = False
                message += f"; PTT release failed: {exc}"
            callback(message, True)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

