
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from lyra.capture import AudioTap, default_input_index, list_inputs, load_sounddevice
from lyra.codec import decode_usb_all, decode_usb_audio, heartbeat_bits
from lyra import codec as lyra_codec
from lyra.const import SAMPLE_RATE
from lyra.modem import audio_tone_peaks


def _read_wav_for_decode(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(str(path))
    x = np.asarray(data)
    if x.ndim == 2 and x.shape[1] >= 2:
        iq = x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64)
        if int(sr) != SAMPLE_RATE:
            g = np.gcd(int(sr), SAMPLE_RATE)
            iq = resample_poly(iq, SAMPLE_RATE // g, int(sr) // g)
        audio = np.real(iq)
    else:
        audio = x.astype(np.float64).reshape(-1)
        if int(sr) != SAMPLE_RATE:
            g = np.gcd(int(sr), SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, int(sr) // g)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return audio / peak


def _pick_device(sd, want: str | None) -> int:
    devices = list_inputs(sd)
    if not devices:
        raise SystemExit("No input audio devices.")
    if want:
        w = want.lower()
        for i, name, _ch in devices:
            if w in name.lower() or w == str(i):
                return i
        print("No input matching", want)
        for i, name, ch in devices:
            print(f"  #{i}  {name}  ({ch} ch)")
        raise SystemExit(1)
    idx = default_input_index(sd)
    if idx is None:
        raise SystemExit("No input audio devices.")
    return idx


def main() -> int:
    p = argparse.ArgumentParser(description="Lyra USB decoder (terminal)")
    p.add_argument("--device", default=None, help="input name substring or device number")
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--seconds", type=float, default=12.0, help="decode window")
    p.add_argument("--wav", type=Path, default=None, help="decode a WAV file and exit")
    args = p.parse_args()

    block, _info = heartbeat_bits("K1ABC", "FN20")
    if args.wav is not None:
        audio = _read_wav_for_decode(args.wav)
        rows = decode_usb_all(audio, block)
        if not rows:
            print("no decode")
            print(lyra_codec.LAST_STATUS or "")
            return 2
        for rec in rows:
            kind, a, b = rec["row"]
            print(
                f"{rec['fa']:.0f}+{rec['fb']:.0f}  Lyra {rec['mode']}  "
                f"{' '.join(p for p in (kind, a, b) if p)}"
            )
        print(lyra_codec.LAST_STATUS or "")
        return 0

    sd = load_sounddevice()
    if args.list:
        for i, name, ch in list_inputs(sd):
            print(f"#{i}  {name}  ({ch} ch)")
        return 0

    dev = _pick_device(sd, args.device)
    tap = AudioTap(dev, seconds=args.seconds)
    tap.start()
    print(
        f"Lyra decoder  |  {tap.name}  (#{tap.device}, {tap.channels} ch)  "
        f"device {tap.capture_rate} Hz"
    )
    print(f"SDR++ USB demod → this input. Decoder at {SAMPLE_RATE} Hz. Ctrl+C to stop.\n")

    last_print = None
    last_t = 0.0
    try:
        while True:
            time.sleep(0.45)
            audio = tap.snapshot()
            rms = tap.rms()
            pk = "—"
            extra = ""
            hot = ""
            if audio is not None:
                peaks = audio_tone_peaks(audio)
                pk = "  ".join(f"{f:.0f}Hz" for f in peaks) if peaks else "—"
                rows = decode_usb_all(audio, block)
                now = time.monotonic()
                for rec in rows:
                    got = rec["row"]
                    key = (got, round(rec["fa"]), round(rec["fb"]))
                    if key == last_print and now - last_t < 1.5:
                        continue
                    last_print = key
                    last_t = now
                    print(
                        f"\n{time.strftime('%H:%M:%S')}  "
                        f"{rec['fa']:.0f}+{rec['fb']:.0f}  "
                        f"{' '.join(got)}",
                        flush=True,
                    )
                extra = f"  {lyra_codec.LAST_STATUS}" if lyra_codec.LAST_STATUS else ""
                hot = "  HOT" if rms > 0.22 else ""
            bar = min(20, int(min(rms, 0.25) * 80))
            print(
                f"\r  in {'█' * bar}{'░' * (20 - bar)}  {rms:.4f}  peaks {pk}{hot}{extra}   ",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        tap.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
