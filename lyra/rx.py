
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import butter, correlate, hilbert, resample_poly, sosfilt, sosfiltfilt

from lyra.const import (
    BIT_RATE,
    CHIRP_HI_HZ,
    CHIRP_LO_HZ,
    CHIRP_S,
    CODED_BITS,
    COSTAS,
    FRAME_BITS,
    MOD_INDEX,
    SAMPLE_RATE,
    SPACING_HZ,
    TONE_A_HZ,
    TONE_B_HZ,
    CHANNELS,
    CHANNEL_MODES,
)
from lyra.fec import decode_payload, deinterleave, deinterleave_metrics
from lyra.pack import unpack_row

LAST_STATUS = ""
LAST_TONES: tuple[float, float] | None = None
LAST_SHIFT = 0.0
LAST_PLAN: list[tuple[float, float]] = []

_MANUAL_PLAN: list[tuple[float, float]] | None = None
_MANUAL_LOCK = threading.Lock()
MAX_CHANNELS = 10


def set_manual_plan(pairs: list[tuple[float, float]] | None) -> None:
    global _MANUAL_PLAN, LAST_PLAN
    with _MANUAL_LOCK:
        if pairs is None:
            _MANUAL_PLAN = None
        else:
            _MANUAL_PLAN = [(float(a), float(b)) for a, b in pairs]
            LAST_PLAN = list(_MANUAL_PLAN)


def get_manual_plan() -> list[tuple[float, float]] | None:
    with _MANUAL_LOCK:
        return None if _MANUAL_PLAN is None else list(_MANUAL_PLAN)


def set_max_channels(n: int) -> None:
    global MAX_CHANNELS
    MAX_CHANNELS = max(1, min(len(CHANNELS), int(n)))
_LAST_ROW: tuple[str, str, str] | None = None
_LAST_CHIRP_AGE = -1.0
_LAST_MIX: tuple[float, float] | None = None
_SEEN: dict[tuple, float] = {}
LAST_ROWS: list[dict] = []

BB_RATE = 400.0
DECIM = int(round(SAMPLE_RATE / BB_RATE))  
SPS = int(round(BB_RATE / BIT_RATE))  
DEV_HZ = (MOD_INDEX / 2.0) * BIT_RATE
_COSTAS = np.array(COSTAS, dtype=np.int8)
_LP = None
_SMOOTH = None

STFT_N = 2048
STFT_HOP = 1024
_STFT_WIN = np.hanning(STFT_N)
_STFT_F = np.fft.fftfreq(STFT_N, 1.0 / SAMPLE_RATE)


def _lp():
    global _LP
    if _LP is None:
        _LP = butter(6, 55.0, btype="low", fs=SAMPLE_RATE, output="sos")
    return _LP


def _smooth():
    global _SMOOTH
    if _SMOOTH is None:
        fs = SAMPLE_RATE / STFT_HOP
        _SMOOTH = butter(2, 1.2, btype="low", fs=fs, output="sos")
    return _SMOOTH


def _interp_bin(spec: np.ndarray, freqs: np.ndarray, i: int) -> float:
    f = float(freqs[i])
    if 0 < i < len(spec) - 1:
        y0, y1, y2 = float(spec[i - 1]), float(spec[i]), float(spec[i + 1])
        den = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(den) > 1e-18:
            df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            f += float(np.clip((y2 - y0) / den, -1.0, 1.0)) * df
    return f


def _band_energy(spec: np.ndarray, freqs: np.ndarray, f0: float, hw: float = 16.0) -> float:
    m = np.abs(freqs - f0) <= hw
    return float(np.sum(spec[m]))


def _centroid_hz(spec: np.ndarray, freqs: np.ndarray, f0: float, hw: float = 18.0) -> float:
    m = np.abs(freqs - f0) <= hw
    w = spec[m]
    s = float(np.sum(w))
    if s <= 0:
        return float(f0)
    return float(np.sum(freqs[m] * w) / s)


def _twins_from_audio(sl: np.ndarray) -> list[tuple[float, float, float]]:
    sl = np.asarray(sl, dtype=np.float64).reshape(-1)
    n = len(sl)
    if n < 2048:
        return []
    sl = sl - np.mean(sl)
    spec = np.abs(np.fft.rfft(sl * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    lo, hi = 400.0, 2_800.0
    noise = float(np.median(spec[(freqs >= lo) & (freqs <= hi)]) + 1e-18)
    peak_max = float(np.max(spec[(freqs >= lo) & (freqs <= hi)]) + 1e-18)
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(spec) - 1):
        f = float(freqs[i])
        if f < lo or f > hi:
            continue
        if spec[i] <= spec[i - 1] or spec[i] <= spec[i + 1]:
            continue
        e = float(spec[i])
        if e < max(5.0 * noise, 0.03 * peak_max):
            continue
        peaks.append((e, _interp_bin(spec, freqs, i)))
    peaks.sort(reverse=True)
    peaks = peaks[:24]
    if len(peaks) < 2:
        return []
    cands: list[tuple[float, float, float]] = []
    for i, (e1, f1) in enumerate(peaks):
        for e2, f2 in peaks[i + 1 :]:
            a, b = (f1, f2) if f1 < f2 else (f2, f1)
            d = b - a
            err = abs(d - SPACING_HZ)
            if err > 6.0:
                continue
            if min(e1, e2) < 0.03 * peak_max:
                continue
            sc = float(min(e1, e2) / (1.0 + 4.0 * err))
            fa = _centroid_hz(spec, freqs, a, 14.0)
            fb = _centroid_hz(spec, freqs, b, 14.0)
            if fa > fb:
                fa, fb = fb, fa
            cands.append((sc, fa, fb))
    cands.sort(reverse=True)
    kept: list[tuple[float, float, float]] = []
    for sc, fa, fb in cands:
        mid = 0.5 * (fa + fb)
        if any(abs(mid - 0.5 * (x + y)) < 35.0 for _, x, y in kept):
            continue
        kept.append((sc, fa, fb))
        if len(kept) >= 8:
            break
    return [(fa, fb, sc) for sc, fa, fb in kept]


def _rail_energy(spec: np.ndarray, freqs: np.ndarray, f0: float, hw: float = 16.0) -> float:
    return float(np.sum(spec[np.abs(freqs - float(f0)) <= hw]))


def graph_twins(audio: np.ndarray) -> list[tuple[float, float]]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = min(len(x), 8192)
    plan0 = list(CHANNELS)[:MAX_CHANNELS]
    if n < 2048:
        return plan0
    acc: list[np.ndarray] = []
    freqs_ref = None
    min_start = max(0, len(x) - int(2.5 * SAMPLE_RATE))
    start = len(x) - n
    hop = max(n // 2, 1)
    while start >= min_start:
        sl = x[start : start + n]
        sl = sl - np.mean(sl)
        spec = np.abs(np.fft.rfft(sl * np.hanning(len(sl)))) ** 2
        freqs_ref = np.fft.rfftfreq(len(sl), 1.0 / SAMPLE_RATE)
        acc.append(spec)
        start -= hop
    spec = np.mean(np.stack(acc, axis=0), axis=0)
    freqs = freqs_ref
    assert freqs is not None

    def pair_score(fa: float, fb: float) -> float:
        e1 = float(np.interp(fa, freqs, spec))
        e2 = float(np.interp(fb, freqs, spec))
        bal = min(e1, e2) / (max(e1, e2) + 1e-18)
        return min(e1, e2) * (bal ** 2)

    plan = list(CHANNELS)
    scores = [pair_score(a, b) for a, b in plan]
    peak = max(scores) + 1e-18
    kept = [p for p, sc in zip(plan, scores) if sc >= 0.12 * peak]
    if not kept:
        kept = plan
    return kept[:MAX_CHANNELS]


def _is_half_speed_iq(audio: np.ndarray) -> bool:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = min(len(x), 131072)
    if n < 8192:
        return False
    sl = x[-n:] - np.mean(x[-n:])
    spec = np.abs(np.fft.rfft(sl * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)

    def score(plan) -> float:
        return float(
            sum(
                min(
                    np.interp(float(a), freqs, spec),
                    np.interp(float(b), freqs, spec),
                )
                for a, b in plan
            )
        )

    normal = score(CHANNELS)
    half = score(((a * 0.5, b * 0.5) for a, b in CHANNELS))
    return half > 1.5 * (normal + 1e-18)



def _measure_twin_slice(sl: np.ndarray) -> tuple[float, float, float] | None:
    twins = _twins_from_audio(sl)
    if not twins:
        return None
    fa, fb, sc = twins[0]
    return fa, fb, sc


def measure_twins(audio: np.ndarray) -> list[tuple[float, float, float]]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = min(len(x), 32768)
    if n < 2048:
        return []
    hop = max(n // 2, 1)
    min_start = max(0, len(x) - int(8.0 * SAMPLE_RATE))
    found: list[tuple[float, float, float]] = []
    start = len(x) - n
    while start >= min_start:
        found.extend(_twins_from_audio(x[start : start + n]))
        start -= hop
    found.sort(key=lambda t: -t[2])
    kept: list[tuple[float, float, float]] = []
    for fa, fb, sc in found:
        mid = 0.5 * (fa + fb)
        if any(abs(mid - 0.5 * (x + y)) < 35.0 for x, y, _ in kept):
            continue
        kept.append((fa, fb, sc))
        if len(kept) >= 8:
            break
    kept.sort(key=lambda t: t[0])
    return kept


def measure_twin(audio: np.ndarray) -> tuple[float, float, float] | None:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = min(len(x), 32768)
    if n < 2048:
        return None
    hop = max(n // 2, 1)
    min_start = max(0, len(x) - int(8.0 * SAMPLE_RATE))
    start = len(x) - n
    while start >= min_start:
        got = _measure_twin_slice(x[start : start + n])
        if got is not None:
            return got
        start -= hop
    return None


class TwinLock:

    def __init__(self) -> None:
        self.fa: float | None = None
        self.fb: float | None = None
        self._miss = 0
        self._away = 0
        self._guard = threading.Lock()

    def reset(self) -> None:
        global _LAST_ROW, _LAST_CHIRP_AGE, LAST_TONES, _LAST_MIX, _SEEN, LAST_SHIFT, LAST_PLAN
        with self._guard:
            self.fa = self.fb = None
            self._miss = 0
            self._away = 0
        _LAST_ROW = None
        _LAST_CHIRP_AGE = -1.0
        LAST_TONES = None
        _LAST_MIX = None
        _SEEN.clear()
        LAST_SHIFT = 0.0
        LAST_PLAN = []

    def snapshot(self) -> tuple[float, float] | None:
        with self._guard:
            if self.fa is None or self.fb is None:
                return None
            return float(self.fa), float(self.fb)

    def update(self, audio: np.ndarray) -> tuple[float, float] | None:
        meas = measure_twin(audio)
        with self._guard:
            if meas is None:
                self._miss += 1
                if self._miss > 25:
                    self.fa = self.fb = None
                    self._away = 0
                    return None
                if self.fa is None:
                    return None
                return float(self.fa), float(self.fb)
            a, b, _sc = meas
            self._miss = 0
            if self.fa is None:
                self.fa, self.fb = a, b
                self._away = 0
                return float(self.fa), float(self.fb)
            spacing_l = self.fb - self.fa
            spacing_n = b - a
            df = 0.5 * ((a - self.fa) + (b - self.fb))
            mid_l = 0.5 * (self.fa + self.fb)
            mid_n = 0.5 * (a + b)
            ratio = mid_n / mid_l if mid_l > 1.0 else 1.0
            clock_jump = abs(ratio - 0.5) < 0.12 or abs(ratio - 2.0) < 0.25
            if (not clock_jump) and abs(spacing_n - spacing_l) < 12.0 and abs(df) < 18.0:
                self.fa = 0.90 * self.fa + 0.10 * a
                self.fb = 0.90 * self.fb + 0.10 * b
                if self.fa > self.fb:
                    self.fa, self.fb = self.fb, self.fa
                self._away = 0
            else:
                
                self._away += 1
                if clock_jump:
                    self.fa, self.fb = a, b
                    self._away = 0
            return float(self.fa), float(self.fb)


LOCK = TwinLock()


def find_pairs(audio: np.ndarray) -> list[tuple[float, float]]:
    twins = measure_twins(audio)
    if twins:
        LOCK.update(audio)
        return [(fa, fb) for fa, fb, _sc in twins]
    pair = LOCK.update(audio)
    return [pair] if pair else []


def _stft_mags(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = []
    mags = []
    n = len(z)
    for i in range(0, n - STFT_N, STFT_HOP):
        mags.append(np.abs(np.fft.fft(z[i : i + STFT_N] * _STFT_WIN)) ** 2)
        times.append(i + STFT_N / 2.0)
    if not times:
        return np.zeros(0), np.zeros((0, STFT_N))
    return np.asarray(times, dtype=np.float64), np.asarray(mags, dtype=np.float64)


def _walk_pin(times: np.ndarray, mags: np.ndarray, f0: float, n: int, hw: float = 14.0) -> np.ndarray:
    fc = float(f0)
    ftrack = np.empty(len(times))
    for k, mag in enumerate(mags):
        band = (np.abs(_STFT_F - fc) <= hw) & (_STFT_F > 200.0)
        w = mag[band]
        fr = _STFT_F[band]
        s = float(np.sum(w))
        est = float(np.sum(fr * w) / s) if s > 0 else fc
        fc = fc + float(np.clip(est - fc, -8.0, 8.0))
        ftrack[k] = fc
    if len(ftrack) >= 12:
        ftrack = sosfiltfilt(_smooth(), ftrack)
    return np.interp(np.arange(n), times, ftrack)


def _pin_bursts(
    times: np.ndarray, mags: np.ndarray, fpin: float, fa: float, fb: float
) -> list[int]:
    if len(times) < 4:
        return []
    ratio = np.empty(len(mags))
    for k, mag in enumerate(mags):
        ep = float(np.sum(mag[np.abs(_STFT_F - fpin) <= 12.0]))
        ea = float(np.sum(mag[np.abs(_STFT_F - fa) <= 16.0]))
        eb = float(np.sum(mag[np.abs(_STFT_F - fb) <= 16.0]))
        ratio[k] = ep / (ea + eb + 1e-18)
    peak = float(np.max(ratio))
    if peak < 1.35:
        return []
    thr = max(1.25, 0.40 * peak)
    min_gap = 1.8 * SAMPLE_RATE
    hits: list[int] = []
    work = ratio.copy()
    for _ in range(12):
        i = int(np.argmax(work))
        if work[i] < thr:
            break
        hits.append(int(max(0, times[i] - STFT_N / 2.0)))
        t0 = times[i]
        work[np.abs(times - t0) < min_gap] = 0
    hits.sort()
    return hits


def _derotate(bb: np.ndarray, df: float) -> np.ndarray:
    if abs(df) < 0.05:
        return bb
    t = np.arange(len(bb), dtype=np.float64) / BB_RATE
    return bb * np.exp(-1j * 2.0 * np.pi * df * t)


def _mix_nco(z: np.ndarray, fc: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * np.cumsum(fc) / SAMPLE_RATE
    bb = sosfilt(_lp(), z * np.exp(-1j * phase))
    n = (len(bb) // DECIM) * DECIM
    if n < DECIM * SPS:
        return np.zeros(0, dtype=np.complex128)
    out = bb[:n].reshape(-1, DECIM).mean(axis=1)
    return out


def _gmsk_metrics(bb: np.ndarray, start: int) -> np.ndarray:
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)[start:]
    n_sym = len(x) // SPS
    if n_sym < FRAME_BITS // 2:
        return np.zeros((0, 2), dtype=np.float64)
    n = n_sym * SPS
    
    
    tau = np.arange(SPS) / BB_RATE
    up = np.exp(1j * 2 * np.pi * DEV_HZ * tau)
    dn = np.exp(-1j * 2 * np.pi * DEV_HZ * tau)
    symbols = x[:n].reshape(n_sym, SPS)
    m1 = np.abs(symbols @ np.conj(up)) ** 2
    m0 = np.abs(symbols @ np.conj(dn)) ** 2
    return np.column_stack((m0, m1))


def _gmsk_bits(bb: np.ndarray, start: int) -> np.ndarray:
    metrics = _gmsk_metrics(bb, start)
    if not len(metrics):
        return np.zeros(0, dtype=np.int8)
    return np.argmax(metrics, axis=1).astype(np.int8)


def _fm_bits(bb: np.ndarray, start: int) -> np.ndarray:
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)
    ph = np.unwrap(np.angle(x))
    hz = np.diff(ph, prepend=ph[0]) * (BB_RATE / (2.0 * np.pi))
    hz = hz[start:]
    n_sym = len(hz) // SPS
    if n_sym < FRAME_BITS // 2:
        return np.zeros(0, dtype=np.int8)
    a, b = SPS // 4, 3 * SPS // 4
    sl = hz[: n_sym * SPS].reshape(n_sym, SPS)[:, a:b]
    return (np.mean(sl, axis=1) > 0).astype(np.int8)


def _mux(ba: np.ndarray, bb: np.ndarray) -> np.ndarray:
    n = min(len(ba), len(bb))
    out = np.empty(n * 2, dtype=np.int8)
    out[0::2] = ba[:n]
    out[1::2] = bb[:n]
    return out


def _mux_soft(ma: np.ndarray, mb: np.ndarray) -> np.ndarray:
    n = min(len(ma), len(mb))
    out = np.empty((n * 2, 2), dtype=np.float64)
    out[0::2] = ma[:n]
    out[1::2] = mb[:n]
    return out


def _unpack(info: np.ndarray):
    return unpack_row(info.tolist())


def _read_frame(bits: np.ndarray):
    b = np.asarray(bits, dtype=np.int8).reshape(-1)
    if len(b) < FRAME_BITS:
        return 0, None
    pm = 2 * b.astype(np.int16) - 1
    pc = 2 * _COSTAS.astype(np.int16) - 1
    corr = np.correlate(pm, pc, mode="valid")
    best = int((int(np.max(corr)) + 8) // 2)
    
    
    for thresh in (7, 6):
        for off in np.flatnonzero(corr >= thresh):
            off = int(off)
            if off + FRAME_BITS > len(b):
                continue
            coded = deinterleave(b[off + 8 : off + FRAME_BITS], CODED_BITS)
            info = decode_payload(coded)
            if info is None:
                continue
            unpacked = _unpack(info)
            if unpacked:
                return 8, unpacked
    return best, None


def _read_frame_soft(metrics: np.ndarray):
    m = np.asarray(metrics, dtype=np.float64).reshape(-1, 2)
    if len(m) < FRAME_BITS:
        return 0, None
    hard = np.argmax(m, axis=1).astype(np.int8)
    pm = 2 * hard.astype(np.int16) - 1
    pc = 2 * _COSTAS.astype(np.int16) - 1
    corr = np.correlate(pm, pc, mode="valid")
    best = int((int(np.max(corr)) + 8) // 2)

    
    
    den = np.sum(m, axis=1) + 1e-18
    llr = (m[:, 1] - m[:, 0]) / den
    soft_corr = np.correlate(llr, pc.astype(np.float64), mode="valid")
    valid_n = len(m) - FRAME_BITS + 1
    if valid_n <= 0:
        return best, None
    order = np.argsort(soft_corr[:valid_n])[::-1][:2]
    for off in order:
        off = int(off)
        coded = deinterleave_metrics(m[off + 8 : off + FRAME_BITS], CODED_BITS)
        info = decode_payload(coded)
        if info is None:
            continue
        unpacked = _unpack(info)
        if unpacked:
            return 8, unpacked
    return best, None


def _stretch_bb(bb: np.ndarray, scale: float) -> np.ndarray:
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)
    if abs(scale - 1.0) < 1e-6:
        return x
    n = int(round(len(x) * scale))
    if n < SPS * 16:
        return x
    t = np.linspace(0.0, len(x) - 1.0, n)
    idx = np.arange(len(x), dtype=np.float64)
    return np.interp(t, idx, x.real) + 1j * np.interp(t, idx, x.imag)


def _data_s(mode: str) -> float:
    if mode == "L":
        return FRAME_BITS / BIT_RATE
    return (FRAME_BITS / 2.0) / BIT_RATE


def _chirp_templates_at(mid: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(CHIRP_S * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    chirp_hw = 100.0 * (SPACING_HZ / 80.0)
    lo, hi = float(mid) - chirp_hw, float(mid) + chirp_hw

    def tmpl(f0: float, f1: float) -> np.ndarray:
        k = (f1 - f0) / CHIRP_S
        ph = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return np.exp(1j * ph)

    return tmpl(lo, hi), tmpl(hi, lo)


def _corr_abs(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    n = len(template)
    nfft = 1 << int(np.ceil(np.log2(len(x) + n)))
    corr = np.fft.ifft(np.fft.fft(x, nfft) * np.conj(np.fft.fft(template, nfft)))
    return np.abs(corr[: max(1, len(x) - n + 1)])


def _chirp_hits(z: np.ndarray, mids: tuple[float, ...] = (1500.0,)) -> list[tuple[int, str, float]]:
    n = int(round(CHIRP_S * SAMPLE_RATE))
    x = np.asarray(z, dtype=np.complex128).reshape(-1)
    if len(x) < n + 64:
        return []
    cu = None
    cd = None
    for mid in mids:
        up_t, dn_t = _chirp_templates_at(mid)
        u = _corr_abs(x, up_t)
        d = _corr_abs(x, dn_t)
        cu = u if cu is None else np.maximum(cu, u)
        cd = d if cd is None else np.maximum(cd, d)
    assert cu is not None and cd is not None
    noise = float(np.median(np.concatenate([cu, cd])) + 1e-18)
    min_gap = int(0.40 * SAMPLE_RATE)
    work_u, work_d = cu.copy(), cd.copy()
    hits: list[tuple[int, str, float]] = []
    for _ in range(12):
        iu, idn = int(np.argmax(work_u)), int(np.argmax(work_d))
        pu, pd = float(work_u[iu]), float(work_d[idn])
        p = max(pu, pd)
        if p < 3.2 * noise:
            break
        if pd >= pu:
            i, mode, q = idn, "F", pu
        else:
            i, mode, q = iu, "L", pd
        lo = max(0, i - min_gap)
        hi = min(len(work_u), i + min_gap)
        work_u[lo:hi] = 0
        work_d[lo:hi] = 0
        if q / p > 0.92:
            continue
        hits.append((i, mode, p))
    hits.sort()
    return hits


def _chirp_hits_decimated(z: np.ndarray, mid: float) -> list[tuple[int, str, float]]:
    x = np.asarray(z, dtype=np.complex128).reshape(-1)
    if len(x) < int(0.5 * SAMPLE_RATE):
        return []
    t = np.arange(len(x), dtype=np.float64) / SAMPLE_RATE
    centered = x * np.exp(-1j * 2.0 * np.pi * float(mid) * t)
    bb = resample_poly(centered, 1, DECIM)
    n = int(round(CHIRP_S * BB_RATE))
    tt = np.arange(n, dtype=np.float64) / BB_RATE
    hw = 0.5 * (CHIRP_HI_HZ - CHIRP_LO_HZ)

    def template(f0: float, f1: float) -> np.ndarray:
        k = (f1 - f0) / CHIRP_S
        return np.exp(1j * 2.0 * np.pi * (f0 * tt + 0.5 * k * tt * tt))

    up = np.abs(correlate(bb, template(-hw, hw), mode="valid", method="fft"))
    down = np.abs(correlate(bb, template(hw, -hw), mode="valid", method="fft"))
    noise = float(np.median(np.concatenate((up, down))) + 1e-18)
    gap = max(1, int(round(0.40 * BB_RATE)))
    work_u, work_d = up.copy(), down.copy()
    hits: list[tuple[int, str, float]] = []
    for _ in range(12):
        iu, idn = int(np.argmax(work_u)), int(np.argmax(work_d))
        pu, pd = float(work_u[iu]), float(work_d[idn])
        power = max(pu, pd)
        if power < 4.5 * noise:
            break
        if pd >= pu:
            i, mode, other = idn, "F", pu
        else:
            i, mode, other = iu, "L", pd
        lo, hi = max(0, i - gap), min(len(work_u), i + gap)
        work_u[lo:hi] = 0.0
        work_d[lo:hi] = 0.0
        if other / power <= 0.92:
            hits.append((i * DECIM, mode, power))
    hits.sort()
    return hits


def _chirp_mode(z: np.ndarray) -> str | None:
    hits = _chirp_hits(z)
    if not hits:
        return None
    return hits[-1][1]


def _chirp_slope_hz_s(times: np.ndarray, mags: np.ndarray) -> float | None:
    band = (_STFT_F >= 1360.0) & (_STFT_F <= 1640.0) & (_STFT_F > 0.0)
    if not np.any(band) or len(mags) < 8:
        return None
    fr = _STFT_F[band]
    inst = np.empty(len(mags), dtype=np.float64)
    for k, mag in enumerate(mags):
        w = mag[band]
        s = float(np.sum(w))
        inst[k] = float(np.sum(fr * w) / s) if s > 0 else 1500.0
    win = max(8, int(round(CHIRP_S * SAMPLE_RATE / STFT_HOP)))
    tsec = times / SAMPLE_RATE
    best_s = 0.0
    best_abs = 0.0
    for i in range(0, len(inst) - win + 1):
        tt = tsec[i : i + win]
        ff = inst[i : i + win]
        dt = float(tt[-1] - tt[0])
        if dt < 0.10:
            continue
        if float(np.max(ff) - np.min(ff)) < 45.0:
            continue
        slope = float(np.polyfit(tt, ff, 1)[0])
        if abs(slope) > best_abs:
            best_abs = abs(slope)
            best_s = slope
    if best_abs < 280.0:
        return None
    return best_s


def _demod(ba: np.ndarray, bb: np.ndarray, scales: tuple[float, ...] = (1.0,), *, copy: bool = False):
    best = 0
    df_pairs: list[tuple[float, float]] = [(0.0, 0.0)]
    for d in (2.5, 5.0):
        df_pairs.extend(((d, d), (-d, -d), (d, -d), (-d, d)))

    def attempt(fn, hyp_set, pairs, starts) -> tuple[int, tuple | None]:
        nonlocal best
        for scale in scales:
            xa0 = _stretch_bb(ba, scale)
            xb0 = _stretch_bb(bb, scale)
            for dfa, dfb in pairs:
                xa = _derotate(xa0, dfa)
                xb = _derotate(xb0, dfb)
                for start in starts:
                    sa = fn(xa, start)
                    sb = fn(xb, start)
                    if len(sa) < 8 or len(sb) < 8:
                        continue
                    for ia, ib, swap in hyp_set:
                        a, b = sa, sb
                        if swap:
                            a, b = b, a
                        if ia:
                            a = 1 - a
                        if ib:
                            b = 1 - b
                        n = min(len(a), len(b))
                        if copy:
                            cands = (a[:n], b[:n])
                        else:
                            cands = (_mux(a[:n], b[:n]),)
                        for bits in cands:
                            sc, got = _read_frame(bits)
                            if sc > best:
                                best = sc
                            if got:
                                return max(sc, 8), got
        return best, None

    identity = ((False, False, False),)
    starts_even = range(0, SPS, 2)
    starts_all = range(0, SPS)
    sc, got = attempt(_gmsk_bits, identity, ((0.0, 0.0),), starts_even)
    if got:
        return sc, got
    sc, got = attempt(_gmsk_bits, identity, ((0.0, 0.0),), starts_all)
    if got:
        return sc, got
    flips = (
        (False, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
    )
    sc, got = attempt(_gmsk_bits, flips, ((0.0, 0.0),), starts_even)
    if got:
        return sc, got
    sc, got = attempt(_fm_bits, identity, ((0.0, 0.0),), starts_even)
    if got:
        return sc, got
    sc, got = attempt(_gmsk_bits, identity, df_pairs[1:], starts_even)
    if got:
        return sc, got
    return attempt(_fm_bits, identity, df_pairs[1:], starts_even)


def _bandpass_iq(z: np.ndarray, mid: float, hw: float = 180.0) -> np.ndarray:
    x = np.asarray(z, dtype=np.complex128).reshape(-1)
    spec = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1.0 / SAMPLE_RATE)
    spec[np.abs(f - float(mid)) > hw] = 0.0
    spec[f < 0.0] = 0.0
    return np.fft.ifft(spec)


def _demod_slice(z: np.ndarray, mix_a: float, mix_b: float, i0: int, mode: str):
    n_mark = int(round(CHIRP_S * SAMPLE_RATE))
    data_n = int(round(_data_s(mode) * SAMPLE_RATE))
    d0 = i0 + n_mark
    d1 = min(len(z), d0 + data_n)
    if d1 - d0 < int(0.40 * SAMPLE_RATE):
        return 0, None
    ma = float(mix_a)
    mb = float(mix_b)
    mid = 0.5 * (ma + mb)
    ma, mb = mid - 0.5 * SPACING_HZ, mid + 0.5 * SPACING_HZ
    warm = int(0.04 * SAMPLE_RATE)
    tail = int(0.08 * SAMPLE_RATE)
    a0 = max(0, d0 - warm)
    a1 = min(len(z), d1 + tail)
    sl = z[a0:a1]
    skip = max(0, (d0 - a0) // DECIM)
    ba = _mix_nco(sl, np.full(len(sl), ma, dtype=np.float64))
    bb = _mix_nco(sl, np.full(len(sl), mb, dtype=np.float64))
    ba, bb = ba[skip:], bb[skip:]
    if len(ba) < SPS * 16 or len(bb) < SPS * 16:
        return 0, None
    copy = mode == "L"
    sc, cand = _demod_fast(ba, bb, copy=copy)
    if cand:
        return max(sc, 8), cand
    if sc >= 5:
        return _demod(ba, bb, scales=(1.0,), copy=copy)
    return sc, None


def _demod_fast(ba: np.ndarray, bb: np.ndarray, *, copy: bool = False):
    best = 0
    got = None
    combined = None
    if copy:
        
        
        
        
        nc = min(len(ba), len(bb))
        cross = np.vdot(bb[:nc], ba[:nc])
        rot = cross / (abs(cross) + 1e-18)
        combined = (ba[:nc] + bb[:nc] * rot) / np.sqrt(2.0)
    for start in range(0, SPS, 2):
        sa = _gmsk_metrics(ba, start)
        sb = _gmsk_metrics(bb, start)
        if len(sa) < 8 or len(sb) < 8:
            continue
        if copy:
            scmb = _gmsk_metrics(combined, start)
            n = min(len(sa), len(sb), len(scmb))
            base = (scmb[:n], sa[:n] + sb[:n], sa[:n], sb[:n])
            candidates = base + tuple(m[:, ::-1] for m in base)
        else:
            n = min(len(sa), len(sb))
            candidates = (
                _mux_soft(sa[:n], sb[:n]),
                _mux_soft(sa[:n, ::-1], sb[:n, ::-1]),
                _mux_soft(sb[:n], sa[:n]),
                _mux_soft(sb[:n, ::-1], sa[:n, ::-1]),
            )
        for cand_metrics in candidates:
            sc, row = _read_frame_soft(cand_metrics)
            if sc > best:
                best = sc
            if row:
                return max(sc, 8), row
    return best, got


def _try_channel(
    z: np.ndarray,
    mix_a: float,
    mix_b: float,
    _now: int,
    expected_mode: str | None = None,
) -> list[dict]:
    mid = 0.5 * (mix_a + mix_b)
    hits = _chirp_hits_decimated(z, mid)
    if expected_mode:
        hits = [h for h in hits if h[1] == expected_mode]
    frame_n = int(round((CHIRP_S + _data_s(expected_mode or "F")) * SAMPLE_RATE))
    strongest: list[tuple[int, str, float]] = []
    for hit in sorted(hits, key=lambda h: h[2], reverse=True):
        if all(abs(hit[0] - kept[0]) >= frame_n for kept in strongest):
            strongest.append(hit)
    hits = strongest
    hits = sorted(hits, key=lambda h: -h[0])[:4]
    out: list[dict] = []
    best_sc = 0
    for i0, mode, _sc in hits:
        used = expected_mode or mode
        sc, cand = _demod_slice(z, mix_a, mix_b, i0, used)
        if sc > best_sc:
            best_sc = sc
        if not cand and expected_mode is None:
            other = "L" if mode == "F" else "F"
            sc, cand = _demod_slice(z, mix_a, mix_b, i0, other)
            if sc > best_sc:
                best_sc = sc
            used = other
        if not cand:
            continue
        out.append(
            {
                "row": cand,
                "fa": float(mix_a),
                "fb": float(mix_b),
                "mode": used,
                "i0": i0,
            }
        )
        break
    _try_channel.last_costas = max(getattr(_try_channel, "last_costas", 0), best_sc)
    return out


def decode_many(audio: np.ndarray) -> list[dict]:
    global LAST_STATUS, LAST_TONES, LAST_ROWS, _LAST_ROW, _LAST_CHIRP_AGE, _LAST_MIX
    global LAST_SHIFT, LAST_PLAN
    LAST_ROWS = []
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(x) < int(0.80 * SAMPLE_RATE):
        LAST_STATUS = "short"
        return []
    x = x - np.mean(x)
    x = x[-min(len(x), int(8.0 * SAMPLE_RATE)) :]
    mag = np.abs(x)
    scale = float(np.percentile(mag, 99.9)) + 1e-12
    x = np.clip(x / scale, -4.0, 4.0)
    _try_channel.last_costas = 0
    manual = get_manual_plan()
    plan = list(manual) if manual else graph_twins(x)
    LAST_PLAN = list(plan)
    if plan:
        LAST_SHIFT = float(0.5 * (plan[0][0] + plan[0][1]) - 500.0)
    else:
        LAST_SHIFT = 0.0
        LAST_STATUS = "no twins"
        return []
    z80 = hilbert(x).astype(np.complex128)
    now80 = len(z80)
    rows: list[dict] = []
    tasks = []
    for fa, fb in plan:
        mid = 0.5 * (fa + fb)
        idx = min(
            range(len(CHANNELS)),
            key=lambda i: abs(mid - 0.5 * (CHANNELS[i][0] + CHANNELS[i][1])),
        )
        tasks.append((fa, fb, CHANNEL_MODES[idx]))

    def decode_task(task):
        fa, fb, mode = task
        return _try_channel(z80, fa, fb, now80, expected_mode=mode)

    with ThreadPoolExecutor(max_workers=min(4, len(plan))) as pool:
        batches = pool.map(decode_task, tasks)
        for batch in batches:
            rows.extend(batch)
    uniq: list[dict] = []
    seen: set[tuple] = set()
    for r in rows:
        k = (r["row"], int(round(0.5 * (r["fa"] + r["fb"]) / 40.0)))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    LAST_ROWS = uniq
    tag = "  ".join(f"{a:.0f}+{b:.0f}" for a, b in plan[:5]) or "no twins"
    if uniq:
        LAST_TONES = (uniq[0]["fa"], uniq[0]["fb"])
        _LAST_MIX = LAST_TONES
        _LAST_ROW = uniq[0]["row"]
        LAST_STATUS = f"Lyra  {len(uniq)} CRC  {tag}"
    else:
        cs = int(getattr(_try_channel, "last_costas", 0))
        if _is_half_speed_iq(x):
            LAST_STATUS = "IQ IS 1/2 SPEED — reload PCM16 IQ file with Float32 Mode OFF"
        else:
            LAST_STATUS = f"{tag}  costas {cs}/8  no CRC"
    return uniq


def decode(audio: np.ndarray):
    rows = decode_many(audio)
    if not rows:
        return None
    return rows[0]["row"]
