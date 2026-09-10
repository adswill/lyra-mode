
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import wavfile

from lyra.capture import load_sounddevice
from lyra.codec import frame_iq
from lyra.const import CHANNEL_MODES, CHANNELS, PIN_HZ, SAMPLE_RATE


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
    if CHANNEL_MODES[idx] != mode:
        allowed = "1–5" if mode == "F" else "6–10"
        raise ValueError(f"Lyra {mode} uses channels {allowed}")
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
        self._thread = threading.Thread(
            target=self._run,
            args=(np.asarray(audio, dtype=np.float32), output_device, rig, callback),
            daemon=True,
        )
        self._thread.start()

    def _run(self, audio, output_device, rig, callback) -> None:
        ok = False
        message = "Transmission stopped"
        try:
            rig.set_ptt(True)
            callback("PTT ON", False)
            if self._stop.wait(0.10):
                return
            if output_device is None:
                
                end = time.monotonic() + len(audio) / SAMPLE_RATE
                while not self._stop.is_set() and time.monotonic() < end:
                    time.sleep(0.02)
            else:
                sd = load_sounddevice()
                stream = sd.OutputStream(
                    device=int(output_device),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=2048,
                    latency="high",
                )
                with self._lock:
                    self._stream = stream
                stream.start()
                for pos in range(0, len(audio), 2048):
                    if self._stop.is_set():
                        break
                    stream.write(audio[pos : pos + 2048].reshape(-1, 1))
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

