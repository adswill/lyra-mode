
from __future__ import annotations

import numpy as np

from lyra.const import (
    BIT_RATE,
    BT,
    CHIRP_HI_HZ,
    CHIRP_LO_HZ,
    CHIRP_S,
    MOD_INDEX,
    SAMPLE_RATE,
    TONE_A_HZ,
    TONE_B_HZ,
)


def _gauss_taps(sps: int, span: int = 4) -> np.ndarray:
    n = span * sps
    t = (np.arange(n) - (n - 1) / 2) / sps
    alpha = np.sqrt(np.log(2)) / (2 * BT)
    h = np.exp(-2 * np.pi**2 * alpha**2 * t**2)
    return h / h.sum()


def lyra_mark(*, up: bool = True) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(CHIRP_S * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f0 = CHIRP_LO_HZ if up else CHIRP_HI_HZ
    f1 = CHIRP_HI_HZ if up else CHIRP_LO_HZ
    k = (f1 - f0) / CHIRP_S
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    inst = f0 + k * t
    return np.exp(1j * phase).astype(np.complex64), inst.astype(np.float64)


def gmsk_baseband(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sps = int(round(SAMPLE_RATE / BIT_RATE))
    nrz = 2.0 * bits.astype(np.float64) - 1.0
    inst = np.repeat(nrz * (MOD_INDEX / 2.0) * BIT_RATE, sps)
    inst = np.convolve(inst, _gauss_taps(sps), mode="same")
    phase = 2 * np.pi * np.cumsum(inst) / SAMPLE_RATE
    return np.exp(1j * phase).astype(np.complex64), inst


def tone_modulate(bits: np.ndarray, fc: float) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(bits, dtype=np.int8)
    bb, inst = gmsk_baseband(bits)
    t = np.arange(len(bb)) / SAMPLE_RATE
    iq = (bb * np.exp(1j * 2 * np.pi * fc * t)).astype(np.complex64)
    return iq, inst[: len(iq)] + fc


def tone_demod(iq: np.ndarray, fc: float, n_bits: int, start: int = 0) -> np.ndarray:
    m = _tone_metrics(iq, fc, n_bits, start=start)
    if len(m) == 0:
        return np.zeros(0, dtype=np.int8)
    return np.argmax(m, axis=1).astype(np.int8)


def find_caller(audio: np.ndarray) -> float | None:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    sl = x[-min(len(x), 2 * SAMPLE_RATE) :]
    if len(sl) < SAMPLE_RATE // 4:
        return None
    sl = sl - np.mean(sl)
    w = np.hanning(len(sl))
    spec = np.abs(np.fft.rfft(sl * w)) ** 2
    freqs = np.fft.rfftfreq(len(sl), 1.0 / SAMPLE_RATE)
    band = (freqs >= 400.0) & (freqs <= 2500.0)
    if not np.any(band):
        return None
    i = int(np.argmax(np.where(band, spec, 0.0)))
    if spec[i] <= 0:
        return None
    if 0 < i < len(spec) - 1:
        y0, y1, y2 = float(spec[i - 1]), float(spec[i]), float(spec[i + 1])
        den = 2.0 * (2.0 * y1 - y0 - y2)
        delta = ((y2 - y0) / den) if abs(den) > 1e-18 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
    else:
        delta = 0.0
    df = float(freqs[1]) if len(freqs) > 1 else 1.0
    return float(freqs[i] + delta * df)


def twin_modulate(
    bits: np.ndarray,
    gain_a: float = 1.0,
    gain_b: float = 1.0,
    *,
    copy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = np.asarray(bits, dtype=np.int8)
    if copy:
        ba, inst_a = gmsk_baseband(bits)
        bb, inst_b = gmsk_baseband(bits)
    else:
        if len(bits) % 2:
            bits = np.concatenate([bits, np.int8([0])])
        ba, inst_a = gmsk_baseband(bits[0::2])
        bb, inst_b = gmsk_baseband(bits[1::2])
    n = min(len(ba), len(bb))
    ba, bb = ba[:n], bb[:n]
    t = np.arange(n) / SAMPLE_RATE
    ga = float(gain_a if gain_a > 0 else 1.0)
    gb = float(gain_b if gain_b > 0 else 1.0)
    iq = (
        ga * ba * np.exp(1j * 2 * np.pi * TONE_A_HZ * t)
        + gb * bb * np.exp(1j * 2 * np.pi * TONE_B_HZ * t)
    )
    iq = (iq / (np.sqrt(ga * ga + gb * gb) or 1.0)).astype(np.complex64)
    return iq, inst_a[:n] + TONE_A_HZ, inst_b[:n] + TONE_B_HZ


def mux_metrics(
    iq: np.ndarray,
    n_pairs: int,
    *,
    start: int = 0,
    f_a: float | None = None,
    f_b: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ma = _tone_metrics(iq, f_a if f_a is not None else TONE_A_HZ, n_pairs, start=start)
    mb = _tone_metrics(iq, f_b if f_b is not None else TONE_B_HZ, n_pairs, start=start)
    n = min(len(ma), len(mb))
    return ma[:n], mb[:n]


def mux_demod(
    iq: np.ndarray,
    n_pairs: int,
    *,
    start: int = 0,
    f_a: float | None = None,
    f_b: float | None = None,
    swap: bool = False,
    invert_a: bool = False,
    invert_b: bool = False,
) -> np.ndarray:
    ma, mb = mux_metrics(iq, n_pairs, start=start, f_a=f_a, f_b=f_b)
    return _mux_hard(ma, mb, swap=swap, invert_a=invert_a, invert_b=invert_b)


def _mux_hard(
    ma: np.ndarray,
    mb: np.ndarray,
    *,
    swap: bool = False,
    invert_a: bool = False,
    invert_b: bool = False,
) -> np.ndarray:
    if swap:
        ma, mb = mb, ma
    n = min(len(ma), len(mb))
    out = np.zeros(n * 2, dtype=np.int8)
    ba = np.argmax(ma[:n], axis=1)
    bb = np.argmax(mb[:n], axis=1)
    if invert_a:
        ba = 1 - ba
    if invert_b:
        bb = 1 - bb
    out[0::2] = ba
    out[1::2] = bb
    return out


BB_RATE = 400.0
_BB_DECIM = int(round(SAMPLE_RATE / BB_RATE))  
_LP_SOS = None


def _lp_sos():
    global _LP_SOS
    if _LP_SOS is None:
        from scipy.signal import butter

        _LP_SOS = butter(6, 40.0, btype="low", fs=SAMPLE_RATE, output="sos")
    return _LP_SOS


def mix_decimate(iq: np.ndarray, fc: float) -> np.ndarray:
    from scipy.signal import sosfilt

    x = np.asarray(iq, dtype=np.complex128).reshape(-1)
    t = np.arange(len(x)) / SAMPLE_RATE
    bb = sosfilt(_lp_sos(), x * np.exp(-1j * 2 * np.pi * float(fc) * t))
    q = _BB_DECIM
    n = (len(bb) // q) * q
    if n < q:
        return np.zeros(0, dtype=np.complex128)
    return bb[:n].reshape(-1, q).mean(axis=1)


def _bb_metrics(bb: np.ndarray, n_bits: int, start: int = 0, df: float = 0.0) -> np.ndarray:
    sps = int(round(BB_RATE / BIT_RATE))
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)
    if abs(df) > 1e-9:
        t = np.arange(len(x)) / BB_RATE
        x = x * np.exp(-1j * 2 * np.pi * df * t)
    x = x[start:]
    n_bits = min(n_bits, len(x) // sps)
    if n_bits < 1:
        return np.zeros((0, 2), dtype=np.float64)
    n = n_bits * sps
    a, b = sps // 4, 3 * sps // 4
    tau = np.arange(b - a) / BB_RATE
    dev = (MOD_INDEX / 2.0) * BIT_RATE
    up = np.exp(1j * 2 * np.pi * dev * tau)
    dn = np.exp(-1j * 2 * np.pi * dev * tau)
    mid = x[:n].reshape(n_bits, sps)[:, a:b]
    m = np.empty((n_bits, 2), dtype=np.float64)
    m[:, 1] = np.abs(mid @ np.conj(up)) ** 2
    m[:, 0] = np.abs(mid @ np.conj(dn)) ** 2
    return m


def mux_demod_bb(
    ba: np.ndarray,
    bb: np.ndarray,
    n_pairs: int,
    *,
    start: int = 0,
    df: float = 0.0,
    swap: bool = False,
    invert_a: bool = False,
    invert_b: bool = False,
) -> np.ndarray:
    ma = _bb_metrics(ba, n_pairs, start=start, df=df)
    mb = _bb_metrics(bb, n_pairs, start=start, df=df)
    return _mux_hard(ma, mb, swap=swap, invert_a=invert_a, invert_b=invert_b)


def _inst_hz(bb: np.ndarray) -> np.ndarray:
    ph = np.unwrap(np.angle(np.asarray(bb, dtype=np.complex128).reshape(-1)))
    return np.diff(ph, prepend=ph[0]) * (BB_RATE / (2.0 * np.pi))


def gmsk_center_hz(bb: np.ndarray) -> float:
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)
    if len(x) < 32:
        return 0.0
    w = np.hanning(len(x))
    spec = np.abs(np.fft.fft(x * w)) ** 2
    freqs = np.fft.fftfreq(len(x), 1.0 / BB_RATE)
    spec[np.abs(freqs) > 35.0] = 0
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(spec) - 1):
        if spec[i] <= spec[i - 1] or spec[i] <= spec[i + 1]:
            continue
        if spec[i] <= 0:
            continue
        peaks.append((float(spec[i]), float(freqs[i])))
    peaks.sort(reverse=True)
    if len(peaks) >= 2 and abs(peaks[0][1] - peaks[1][1]) > 3.0:
        return 0.5 * (peaks[0][1] + peaks[1][1])
    if peaks:
        return peaks[0][1]
    i = int(np.argmax(spec))
    return float(freqs[i])


def derotate_bb(bb: np.ndarray, df: float | None = None) -> np.ndarray:
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)
    if df is None:
        df = gmsk_center_hz(x)
    if abs(df) < 0.02:
        return x
    t = np.arange(len(x)) / BB_RATE
    return x * np.exp(-1j * 2 * np.pi * df * t)


def _fm_tone_bits(bb: np.ndarray, n_bits: int, start: int = 0) -> np.ndarray:
    sps = int(round(BB_RATE / BIT_RATE))
    x = derotate_bb(bb)
    hz = _inst_hz(x)
    if len(hz) < sps + 2:
        return np.zeros(0, dtype=np.int8)
    win = sps * 5
    if len(hz) > win:
        n = (len(hz) // win) * win
        chunk = hz[:n].reshape(-1, win)
        hz[:n] = (chunk - np.median(chunk, axis=1, keepdims=True)).reshape(-1)
    hz = hz[start:]
    n_bits = min(n_bits, len(hz) // sps)
    if n_bits < 1:
        return np.zeros(0, dtype=np.int8)
    a, b = sps // 4, 3 * sps // 4
    sl = hz[: n_bits * sps].reshape(n_bits, sps)[:, a:b]
    return (np.mean(sl, axis=1) > 0).astype(np.int8)


def mux_demod_fm(
    ba: np.ndarray,
    bb: np.ndarray,
    n_pairs: int,
    *,
    start: int = 0,
    swap: bool = False,
    invert_a: bool = False,
    invert_b: bool = False,
) -> np.ndarray:
    sa = _fm_tone_bits(ba, n_pairs, start=start)
    sb = _fm_tone_bits(bb, n_pairs, start=start)
    n = min(len(sa), len(sb))
    if swap:
        sa, sb = sb, sa
    if invert_a:
        sa = 1 - sa
    if invert_b:
        sb = 1 - sb
    out = np.zeros(n * 2, dtype=np.int8)
    out[0::2] = sa[:n]
    out[1::2] = sb[:n]
    return out


def chirp_indices(iq: np.ndarray, fa: float, fb: float) -> list[int]:
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    n = int(round(CHIRP_S * SAMPLE_RATE))
    if len(x) < n + 64:
        return []
    t = np.arange(n) / SAMPLE_RATE
    f = float(fa) + (float(fb) - float(fa)) * (t / max(CHIRP_S, 1e-9))
    tmpl = np.exp(1j * 2 * np.pi * np.cumsum(f) / SAMPLE_RATE).astype(np.complex64)
    nfft = int(2 ** np.ceil(np.log2(len(x) + n)))
    corr = np.fft.ifft(np.fft.fft(x, nfft) * np.conj(np.fft.fft(tmpl, nfft)))[: len(x) - n + 1]
    mag = np.abs(corr)
    floor = float(np.median(mag) + 1e-18)
    peak = float(np.max(mag))
    if peak < 4.0 * floor:
        return []
    thr = max(0.32 * peak, 3.5 * floor)
    min_gap = int(0.8 * SAMPLE_RATE)
    hits: list[int] = []
    work = mag.copy()
    for _ in range(6):
        i = int(np.argmax(work))
        if work[i] < thr:
            break
        hits.append(i)
        work[max(0, i - min_gap) : i + min_gap] = 0
    hits.sort()
    return hits


def _tone_metrics(iq: np.ndarray, fc: float, n_bits: int, start: int = 0) -> np.ndarray:
    sps = int(round(SAMPLE_RATE / BIT_RATE))
    x = np.asarray(iq, dtype=np.complex128).reshape(-1)[start:]
    n_bits = min(n_bits, len(x) // sps)
    if n_bits < 1:
        return np.zeros((0, 2), dtype=np.float64)
    n = n_bits * sps
    t = (np.arange(n) + start) / SAMPLE_RATE
    bb = x[:n] * np.exp(-1j * 2 * np.pi * fc * t)
    a, b = sps // 4, 3 * sps // 4
    tau = np.arange(b - a) / SAMPLE_RATE
    dev = (MOD_INDEX / 2.0) * BIT_RATE
    up = np.exp(1j * 2 * np.pi * dev * tau)
    dn = np.exp(-1j * 2 * np.pi * dev * tau)
    mid = bb.reshape(n_bits, sps)[:, a:b]
    m = np.empty((n_bits, 2), dtype=np.float64)
    m[:, 1] = np.abs(mid @ np.conj(up)) ** 2
    m[:, 0] = np.abs(mid @ np.conj(dn)) ** 2
    return m


def usb_analytic(audio: np.ndarray) -> np.ndarray:
    from scipy.signal import hilbert

    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) + 1e-12
    return hilbert(x / peak).astype(np.complex64)


def audio_tone_peaks(audio: np.ndarray, n_peaks: int = 2) -> list[float]:
    pair = find_lyra_pair(audio)
    if pair is not None:
        return [pair[0], pair[1]]
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    sl = x[-min(len(x), SAMPLE_RATE) :]
    if len(sl) < 1024:
        return []
    w = np.hanning(len(sl))
    spec = np.abs(np.fft.rfft(sl * w))
    freqs = np.fft.rfftfreq(len(sl), 1.0 / SAMPLE_RATE)
    lo, hi = 400.0, 2800.0
    spec[(freqs < lo) | (freqs > hi)] = 0
    peaks: list[float] = []
    work = spec.copy()
    for _ in range(n_peaks):
        i = int(np.argmax(work))
        if work[i] <= 0:
            break
        peaks.append(float(freqs[i]))
        work[max(0, i - 8) : i + 9] = 0
    return peaks


def find_lyra_pairs(audio: np.ndarray, spacing: float | None = None, n_keep: int = 4) -> list[tuple[float, float, float]]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    
    sl = x[-min(len(x), 8192) :]
    if len(sl) < 2048:
        return []
    sl = sl - np.mean(sl)
    w = np.hanning(len(sl))
    spec = np.abs(np.fft.rfft(sl * w)) ** 2
    freqs = np.fft.rfftfreq(len(sl), 1.0 / SAMPLE_RATE)
    df = float(freqs[1]) if len(freqs) > 1 else 1.0
    lo, hi = 300.0, 3200.0
    band = (freqs >= lo) & (freqs <= hi)
    noise = float(np.median(spec[band]) + 1e-18)
    peak_max = float(np.max(spec[band]) + 1e-18)

    def interp_peak(i: int) -> float:
        fa = float(freqs[i])
        if 0 < i < len(spec) - 1:
            y0, y1, y2 = float(spec[i - 1]), float(spec[i]), float(spec[i + 1])
            den = 2.0 * (2.0 * y1 - y0 - y2)
            delta = ((y2 - y0) / den) if abs(den) > 1e-18 else 0.0
            fa += float(np.clip(delta, -1.0, 1.0)) * df
        return fa

    peaks: list[tuple[float, int]] = []
    for i in range(1, len(spec) - 1):
        f = float(freqs[i])
        if f < lo or f > hi:
            continue
        if spec[i] < max(2.5 * noise, 0.05 * peak_max):
            continue
        if spec[i] <= spec[i - 1] or spec[i] <= spec[i + 1]:
            continue
        peaks.append((float(spec[i]), i))
    peaks.sort(reverse=True)
    peaks = peaks[:20]
    scored: list[tuple[float, float, float]] = []
    for n1, (p1, i1) in enumerate(peaks):
        for p2, i2 in peaks[n1 + 1 :]:
            f1, f2 = interp_peak(i1), interp_peak(i2)
            e1, e2 = p1, p2
            if f1 > f2:
                f1, f2, e1, e2 = f2, f1, e2, e1
            d = f2 - f1
            if d < 75.0 or d > 140.0:
                continue
            if min(e1, e2) < 0.02 * peak_max:
                continue
            
            def centroid(f0: float) -> float:
                m = (np.abs(freqs - f0) <= 18.0) & band
                w = spec[m]
                s = float(np.sum(w))
                if s <= 0:
                    return f0
                return float(np.sum(freqs[m] * w) / s)

            f1, f2 = centroid(f1), centroid(f2)
            if f1 > f2:
                f1, f2 = f2, f1
            d = f2 - f1
            if d < 75.0 or d > 140.0:
                continue
            score = float(min(e1, e2) * np.sqrt(max(e1, e2)))
            scored.append((score, f1, f2))
    scored.sort(reverse=True)
    out: list[tuple[float, float, float]] = []
    for row in scored:
        if any(abs(row[1] - o[1]) < 25 for o in out):
            continue
        out.append(row)
        if len(out) >= n_keep:
            break
    sp = float(spacing or 100.0)
    for _s1, i1 in peaks[:3]:
        fc = interp_peak(i1)
        for fa, fb in ((fc, fc + sp), (fc - sp, fc)):
            if fa < 300.0 or fb > 3200.0 or fb - fa < 75.0:
                continue
            if any(abs(fa - o[1]) < 25 for o in out):
                continue
            out.append((_s1, fa, fb))
            if len(out) >= max(n_keep, 4):
                break
        if len(out) >= max(n_keep, 4):
            break
    return out


def find_lyra_pair(audio: np.ndarray, spacing: float | None = None) -> tuple[float, float] | None:
    pairs = find_lyra_pairs(audio, spacing=spacing, n_keep=1)
    if not pairs:
        return None
    return pairs[0][1], pairs[0][2]


def twin_metrics(iq: np.ndarray, n_bits: int) -> np.ndarray:
    return _tone_metrics(iq, TONE_A_HZ, n_bits) + _tone_metrics(iq, TONE_B_HZ, n_bits)


def twin_demod(iq: np.ndarray, n_bits: int) -> np.ndarray:
    return np.argmax(twin_metrics(iq, n_bits), axis=1).astype(np.int8)


def usb_audio(iq: np.ndarray) -> np.ndarray:
    return np.real(iq).astype(np.float64)


def usb_spectrum(audio: np.ndarray, nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(x) < 256:
        return np.array([200.0, 5_500.0]), np.array([-120.0, -120.0])
    n = int(min(len(x), nfft))
    if n < 256:
        return np.array([200.0, 5_500.0]), np.array([-120.0, -120.0])
    sl = x[-n:]
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(sl * window))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    
    amplitude = spec / (0.5 * float(np.sum(window)) + 1e-18)
    mag = 20.0 * np.log10(amplitude + 1e-12)
    m = (freqs >= 200.0) & (freqs <= 5_500.0)
    return freqs[m], mag[m]
