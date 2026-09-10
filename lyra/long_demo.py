from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from lyra.const import BIT_RATE, CHANNELS, CONV_TAIL, INFO_BITS, SAMPLE_RATE
from lyra.fec import crc16
from lyra.pack import pack_73, pack_cq, pack_reply, pack_rpt, pack_rr73, unpack_row


RAIL_A_HZ = 1460.0
RAIL_B_HZ = 1540.0
CENTER_HZ = 1500.0
MARK_S = 0.20
BB_RATE = 400.0
SPS = int(round(BB_RATE / BIT_RATE))
BT = 0.3
UW_BITS = 16
STEPS = INFO_BITS + 16 + CONV_TAIL
CODED_BITS = 4 * STEPS
RAIL_DATA = CODED_BITS // 2
RAIL_BITS = UW_BITS + RAIL_DATA
FRAME_S = MARK_S + RAIL_BITS / BIT_RATE
GENERATORS = (0o171, 0o133, 0o165, 0o117)
N_STATE = 64
_FILTER = butter(6, 48.0, btype="low", fs=BB_RATE, output="sos")
UW_A = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=np.int8)
UW_B = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=np.int8)
L_CENTERS = tuple(0.5 * (CHANNELS[i][0] + CHANNELS[i][1]) for i in range(5, 10))
SEARCH_CENTERS = (CENTER_HZ,) + L_CENTERS
LAST_STATUS = ""
LAST_PLAN: list[tuple[float, float]] = []
LAST_ROWS: list[dict] = []


def _parity(value: int) -> int:
    value ^= value >> 16
    value ^= value >> 8
    value ^= value >> 4
    value ^= value >> 2
    value ^= value >> 1
    return value & 1


def _code(info: np.ndarray) -> np.ndarray:
    source = np.concatenate((info[:INFO_BITS], crc16(info[:INFO_BITS])))
    register = 0
    output = []
    for bit in list(source) + [0] * CONV_TAIL:
        register = ((register << 1) | int(bit)) & 0x7F
        for generator in GENERATORS:
            output.append(_parity(register & generator))
    return np.asarray(output, dtype=np.int8)


def _shuffle(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    return x.reshape(18, 22, *x.shape[1:]).transpose(1, 0, *range(2, x.ndim + 1)).reshape((-1,) + x.shape[1:])


def _unshuffle(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    return x.reshape(22, 18, *x.shape[1:]).transpose(1, 0, *range(2, x.ndim + 1)).reshape((-1,) + x.shape[1:])


_STATES = np.arange(N_STATE, dtype=np.int16)
_PRED = np.column_stack((_STATES >> 1, (_STATES >> 1) | 0x20))
_OUT = np.empty((N_STATE, 2, 4), dtype=np.int8)
for _nst in range(N_STATE):
    _bit = _nst & 1
    for _j, _st in enumerate(_PRED[_nst]):
        _reg = ((int(_st) << 1) | _bit) & 0x7F
        _OUT[_nst, _j] = [_parity(_reg & g) for g in GENERATORS]


def _viterbi(llr: np.ndarray) -> np.ndarray:
    metrics = np.zeros((STEPS, 4, 2), dtype=np.float64)
    llr = np.clip(np.asarray(llr, dtype=np.float64).reshape(-1)[:CODED_BITS], -8.0, 8.0)
    if len(llr) < CODED_BITS:
        llr = np.concatenate((llr, np.zeros(CODED_BITS - len(llr))))
    metrics[:, :, 0] = -0.5 * llr.reshape(STEPS, 4)
    metrics[:, :, 1] = 0.5 * llr.reshape(STEPS, 4)
    path = np.full(N_STATE, 1e18)
    path[0] = 0.0
    prev = np.zeros((STEPS, N_STATE), dtype=np.int16)
    for step in range(STEPS):
        branch = np.zeros((N_STATE, 2), dtype=np.float64)
        for k in range(4):
            branch -= metrics[step, k, _OUT[:, :, k]]
        cand = path[_PRED] + branch
        choice = np.argmin(cand, axis=1)
        path = cand[_STATES, choice]
        prev[step] = _PRED[_STATES, choice]
    state = 0
    bits = np.zeros(STEPS, dtype=np.int8)
    for step in range(STEPS - 1, -1, -1):
        bits[step] = state & 1
        state = int(prev[step, state])
    return bits[: INFO_BITS + 16]


def _mark() -> np.ndarray:
    count = int(round(MARK_S * SAMPLE_RATE))
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    k = 200.0 / MARK_S
    pa = 2.0 * np.pi * (1400.0 * t + 0.5 * k * t * t)
    pb = 2.0 * np.pi * (1600.0 * t - 0.5 * k * t * t)
    env = np.ones(count, dtype=np.float64)
    fade = min(int(0.012 * SAMPLE_RATE), count // 6)
    env[:fade] = np.sin(np.linspace(0.0, 0.5 * np.pi, fade)) ** 2
    env[-fade:] = np.cos(np.linspace(0.0, 0.5 * np.pi, fade)) ** 2
    return (((np.exp(1j * pa) + np.exp(1j * pb)) / np.sqrt(2.0)) * env).astype(np.complex64)


def _gmsk(bits: np.ndarray) -> np.ndarray:
    samples = int(round(SAMPLE_RATE / BIT_RATE))
    span = 4 * samples
    t = (np.arange(span) - (span - 1) / 2) / samples
    alpha = np.sqrt(np.log(2.0)) / (2.0 * BT)
    taps = np.exp(-2.0 * np.pi**2 * alpha**2 * t**2)
    taps /= np.sum(taps)
    values = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    frequency = np.convolve(np.repeat(values * 0.25 * BIT_RATE, samples), taps, mode="same")
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    wave = np.exp(1j * phase).astype(np.complex64)
    fade = min(samples, len(wave) // 8)
    wave[:fade] *= (np.sin(np.linspace(0.0, 0.5 * np.pi, fade)) ** 2).astype(np.float32)
    wave[-fade:] *= (np.cos(np.linspace(0.0, 0.5 * np.pi, fade)) ** 2).astype(np.float32)
    return wave


def frame(info_bits) -> np.ndarray:
    info = np.asarray(info_bits, dtype=np.int8).reshape(-1)
    if len(info) < INFO_BITS:
        info = np.concatenate((info, np.zeros(INFO_BITS - len(info), dtype=np.int8)))
    coded = _shuffle(_code(info[:INFO_BITS]))
    rail_a = np.concatenate((UW_A, coded[0::2]))
    rail_b = np.concatenate((UW_B, coded[1::2]))
    base_a = _gmsk(rail_a)
    base_b = _gmsk(rail_b)
    count = min(len(base_a), len(base_b))
    base_a = base_a[:count]
    base_b = base_b[:count]
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    data = (
        base_a * np.exp(2j * np.pi * RAIL_A_HZ * t)
        + base_b * np.exp(2j * np.pi * RAIL_B_HZ * t)
    ) / np.sqrt(2.0)
    return np.concatenate((_mark(), data.astype(np.complex64)))


def _move(signal: np.ndarray, center: float) -> np.ndarray:
    t = np.arange(len(signal), dtype=np.float64) / SAMPLE_RATE
    return signal * np.exp(2j * np.pi * (center - CENTER_HZ) * t)


def _rail(frame_iq: np.ndarray, frequency: float, offset: float) -> np.ndarray:
    t = np.arange(len(frame_iq), dtype=np.float64) / SAMPLE_RATE
    mixed = frame_iq * np.exp(-2j * np.pi * (frequency + offset) * t)
    baseband = resample_poly(mixed, 1, int(round(SAMPLE_RATE / BB_RATE)))
    return sosfiltfilt(_FILTER, baseband)


def _all_llr(baseband: np.ndarray) -> np.ndarray:
    values = np.asarray(baseband, dtype=np.complex128).reshape(-1)
    count = len(values) // SPS
    if count < RAIL_BITS:
        return np.zeros(0, dtype=np.float64)
    symbols = values[: count * SPS].reshape(count, SPS)
    t = np.arange(SPS, dtype=np.float64) / BB_RATE
    deviation = 0.25 * BIT_RATE
    low = np.exp(-2j * np.pi * deviation * t)
    high = np.exp(2j * np.pi * deviation * t)
    score_0 = np.abs(symbols @ np.conj(low)) ** 2
    score_1 = np.abs(symbols @ np.conj(high)) ** 2
    floor = 0.04 * float(np.median(score_0 + score_1)) + 1e-18
    return np.clip(np.log((score_1 + floor) / (score_0 + floor)), -8.0, 8.0)


def _llr(baseband: np.ndarray, start: int) -> np.ndarray:
    first = int(round(MARK_S * BB_RATE)) + start
    values = _all_llr(np.asarray(baseband, dtype=np.complex128)[first:])
    if len(values) < RAIL_BITS:
        return np.zeros(0, dtype=np.float64)
    return values[:RAIL_BITS]


def _uw_score(llr: np.ndarray, uw: np.ndarray) -> int:
    return int(np.count_nonzero((llr[:UW_BITS] > 0).astype(np.int8) == uw))


def _decode_llr(llr_a: np.ndarray, llr_b: np.ndarray):
    if len(llr_a) < RAIL_BITS or len(llr_b) < RAIL_BITS:
        return None
    if max(_uw_score(llr_a, UW_A), _uw_score(llr_b, UW_B)) < 12:
        return None
    joined = np.zeros(CODED_BITS, dtype=np.float64)
    joined[0::2] = llr_a[UW_BITS:]
    joined[1::2] = llr_b[UW_BITS:]
    if _uw_score(llr_a, UW_A) < 12:
        joined[0::2] *= 0.12
    if _uw_score(llr_b, UW_B) < 12:
        joined[1::2] *= 0.12
    decoded = _viterbi(_unshuffle(joined))
    info = decoded[:INFO_BITS]
    if not np.array_equal(decoded[INFO_BITS:], crc16(info)):
        return None
    return unpack_row(info.tolist())


def decode_iq_all(samples: np.ndarray, center: float = CENTER_HZ) -> list:
    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    need = int(round(FRAME_S * SAMPLE_RATE))
    if len(values) < need:
        return []
    uw_a = 2.0 * UW_A.astype(np.float64) - 1.0
    uw_b = 2.0 * UW_B.astype(np.float64) - 1.0
    found = []
    seen = set()
    for offset in (0.0, -2.0, 2.0, -4.0, 4.0):
        rail_a = _rail(values, center - 40.0, offset)
        rail_b = _rail(values, center + 40.0, offset)
        for sub in (0, 2, 4, 6):
            la = _all_llr(np.asarray(rail_a)[sub:])
            lb = _all_llr(np.asarray(rail_b)[sub:])
            if len(la) < RAIL_BITS or len(lb) < RAIL_BITS:
                continue
            score = np.correlate(la, uw_a, mode="valid") + np.correlate(lb, uw_b, mode="valid")
            order = np.argsort(score)[::-1]
            seen_idx = set()
            for index in order[:24]:
                key = int(index) // 2
                if key in seen_idx:
                    continue
                seen_idx.add(key)
                end = int(index) + RAIL_BITS
                if end > min(len(la), len(lb)):
                    continue
                result = _decode_llr(la[int(index) : end], lb[int(index) : end])
                if result is None or result in seen:
                    continue
                seen.add(result)
                found.append(result)
        if found:
            break
    return found


def decode_iq(samples: np.ndarray, center: float = CENTER_HZ):
    got = decode_iq_all(samples, center=center)
    return got[0] if got else None


def decode_audio(samples: np.ndarray, center: float = CENTER_HZ):
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    return decode_iq(hilbert(audio - np.mean(audio)), center=center)


def _as_iq(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if np.iscomplexobj(values):
        return values.astype(np.complex128).reshape(-1)
    real = values.astype(np.float64).reshape(-1)
    return hilbert(real - np.mean(real))


def _active_centers(iq: np.ndarray) -> list[float]:
    values = np.asarray(iq, dtype=np.complex128).reshape(-1)
    n = min(len(values), 16384)
    if len(values) <= n:
        pieces = [values]
    else:
        mid = len(values) // 2
        pieces = [values[:n], values[mid - n // 2 : mid + n // 2], values[-n:]]
    spec = None
    freqs = np.fft.fftfreq(n, 1.0 / SAMPLE_RATE)
    for piece in pieces:
        if len(piece) < n:
            continue
        part = np.abs(np.fft.fft(piece[:n] * np.hanning(n))) ** 2
        spec = part if spec is None else spec + part
    if spec is None:
        return [float(CENTER_HZ)]
    scored = []
    for center in SEARCH_CENTERS:
        band = (np.abs(freqs - (center - 40.0)) <= 25.0) | (
            np.abs(freqs - (center + 40.0)) <= 25.0
        )
        scored.append((float(np.sum(spec[band])), float(center)))
    scored.sort(reverse=True)
    floor = scored[0][0] * 0.08 + 1e-18
    picked = [center for energy, center in scored if energy >= floor][:4]
    l_scores = [(energy, center) for energy, center in scored if center in L_CENTERS]
    for _energy, center in l_scores[:2]:
        if center not in picked:
            picked.append(center)
    return picked or [scored[0][1]]


def read_wav(path: str | Path) -> np.ndarray:
    path = Path(path)
    sr, data = wavfile.read(str(path))
    values = np.asarray(data)
    if values.ndim == 2 and values.shape[1] >= 2:
        iq = values[:, 0].astype(np.float64) + 1j * values[:, 1].astype(np.float64)
        if int(sr) != SAMPLE_RATE:
            g = np.gcd(int(sr), SAMPLE_RATE)
            iq = resample_poly(iq, SAMPLE_RATE // g, int(sr) // g)
        peak = max(float(np.max(np.abs(iq.real))), float(np.max(np.abs(iq.imag)))) + 1e-12
        return iq / peak
    audio = values.astype(np.float64).reshape(-1)
    if int(sr) != SAMPLE_RATE:
        g = np.gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sr) // g)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return audio / peak


def decode_many(audio: np.ndarray) -> list[dict]:
    global LAST_STATUS, LAST_PLAN, LAST_ROWS
    LAST_ROWS = []
    LAST_PLAN = []
    values = np.asarray(audio)
    if values.size < int(0.80 * SAMPLE_RATE):
        LAST_STATUS = "short"
        return []
    streams = [_as_iq(values)]
    if np.iscomplexobj(values):
        real = np.real(values).astype(np.float64).reshape(-1)
        streams.append(hilbert(real - np.mean(real)))
    rows: list[dict] = []
    seen: set[tuple] = set()
    plan: list[tuple[float, float]] = []
    for iq in streams:
        active = [float(c) for c in _active_centers(iq)]
        for center in active:
            fa = float(center - 40.0)
            fb = float(center + 40.0)
            for got in decode_iq_all(iq, center=center):
                key = (got, int(round(center / 40.0)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"row": got, "fa": fa, "fb": fb, "mode": "L"})
                plan.append((fa, fb))
        if rows:
            break
        for center in L_CENTERS:
            if float(center) in active:
                continue
            fa = float(center - 40.0)
            fb = float(center + 40.0)
            for got in decode_iq_all(iq, center=center):
                key = (got, int(round(center / 40.0)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"row": got, "fa": fa, "fb": fb, "mode": "L"})
                plan.append((fa, fb))
        if rows:
            break
    uniq_plan: list[tuple[float, float]] = []
    seen_mid: set[int] = set()
    for fa, fb in plan:
        mid = int(round(0.5 * (fa + fb) / 40.0))
        if mid in seen_mid:
            continue
        seen_mid.add(mid)
        uniq_plan.append((fa, fb))
    LAST_ROWS = rows
    LAST_PLAN = uniq_plan
    if rows:
        tag = "  ".join(f"{a:.0f}+{b:.0f}" for a, b in uniq_plan[:5])
        LAST_STATUS = f"{len(rows)} decode  {tag}"
    else:
        LAST_STATUS = "no CRC"
    return rows


def decode_usb_all(audio: np.ndarray, block: np.ndarray | None = None) -> list[dict]:
    return decode_many(audio)


def build_tx_audio(
    info_bits,
    *,
    mode: str = "L",
    channel: int = 6,
    level: float = 0.70,
) -> np.ndarray:
    if str(mode).upper() != "L":
        raise ValueError("lyra long v2 uses mode L")
    idx = int(channel) - 1
    if idx < 5 or idx >= len(CHANNELS):
        raise ValueError("lyra long v2 uses channels 6–10")
    center = 0.5 * (CHANNELS[idx][0] + CHANNELS[idx][1])
    iq = _move(frame(info_bits), center)
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


def qso() -> np.ndarray:
    messages = (
        (pack_cq("K1ABC", "FN20"), 3000.0),
        (pack_reply("K1ABC", "W1AW", "FN31"), 3500.0),
        (pack_rpt("W1AW", "K1ABC", -8), 3000.0),
        (pack_rpt("K1ABC", "W1AW", -8, True), 3500.0),
        (pack_rr73("W1AW", "K1ABC"), 3000.0),
        (pack_73("K1ABC", "W1AW"), 3500.0),
    )
    gap = np.zeros(SAMPLE_RATE, dtype=np.complex64)
    parts = []
    for bits, center in messages:
        parts.extend((_move(frame(bits), center), gap))
    return np.concatenate(parts)


def make_wav(path: str | Path) -> Path:
    path = Path(path)
    signal = qso() * 0.08
    rng = np.random.default_rng(20260910)
    noise = 0.08 * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    result = signal + noise
    peak = max(float(np.max(np.abs(result.real))), float(np.max(np.abs(result.imag))))
    result *= 0.9 / peak
    stereo = np.column_stack((result.real, result.imag))
    wavfile.write(path, SAMPLE_RATE, np.asarray(stereo * 32767.0, dtype=np.int16))
    return path


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(np.abs(iq) ** 2) + 1e-18)
    n0 = p * (SAMPLE_RATE / 2500.0) / (10 ** (snr_db / 10.0))
    noise = np.sqrt(n0 / 2.0) * (
        rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    )
    return (iq + noise).astype(np.complex64)


if __name__ == "__main__":
    output = Path("example qso wavs") / "lyra_long_demo_qso_iq.wav"
    print(make_wav(output), f"{FRAME_S:.2f}s")
