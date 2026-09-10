
from __future__ import annotations

import numpy as np

from lyra.const import (
    BIT_RATE,
    CAPTURE_S,
    CHIRP_S,
    COSTAS,
    FRAME_BITS,
    INFO_BITS,
    SAMPLE_RATE,
    TONE_A_HZ,
    TONE_B_HZ,
)
from lyra.fec import crc16, decode_payload, encode_payload, interleave
from lyra.modem import (
    BB_RATE,
    lyra_mark,
    chirp_indices,
    find_lyra_pairs,
    mix_decimate,
    mux_demod,
    mux_demod_bb,
    mux_demod_fm,
    twin_modulate,
    usb_analytic,
)
from lyra.pack import (
    pack_73,
    pack_cq,
    pack_reply,
    pack_rpt,
    pack_rr73,
    unpack_cq,
    unpack_row,
)


LAST_STATUS = ""


def heartbeat_bits(call: str, grid: str) -> tuple[np.ndarray, np.ndarray]:
    info = np.array(pack_cq(call, grid), dtype=np.int8)
    coded = interleave(encode_payload(info))
    block = np.concatenate([np.array(COSTAS, dtype=np.int8), coded])
    return block, info


def prefix_len() -> int:
    return FRAME_BITS


def _cycle_iq(block: np.ndarray, gain_a: float, gain_b: float, mode: str = "F"):
    
    air = np.asarray(block[: prefix_len()], dtype=np.int8)
    long = mode.upper() == "L"
    if (not long) and (len(air) % 2):
        air = np.concatenate([air, np.int8([0])])
    mark, fmark = lyra_mark(up=long)
    data, fa, fb = twin_modulate(air, gain_a=gain_a, gain_b=gain_b, copy=long)
    iq = np.concatenate([mark, data])
    za = np.concatenate([fmark, fa])
    zb = np.concatenate([fmark, fb])
    return iq, za, zb, len(mark), len(data)


def encode_heartbeat(
    call: str,
    grid: str,
    seconds: float = CAPTURE_S,
    gain_a=1.0,
    gain_b=1.0,
    mode: str = "F",
):
    block, info = heartbeat_bits(call, grid)
    parts, fa_p, fb_p = [], [], []
    mark_n = data_n = 0
    filled = 0.0
    while filled < seconds:
        iq, za, zb, mark_n, data_n = _cycle_iq(block, gain_a, gain_b, mode=mode)
        parts.append(iq)
        fa_p.append(za)
        fb_p.append(zb)
        filled += len(iq) / SAMPLE_RATE
    iq = np.concatenate(parts)
    fa = np.concatenate(fa_p)
    fb = np.concatenate(fb_p)
    nkeep = int(seconds * SAMPLE_RATE)
    return iq[:nkeep], fa[:nkeep], fb[:nkeep], block, info, mark_n, data_n


def frame_iq(info_bits, mode: str = "F") -> np.ndarray:
    info = np.asarray(info_bits, dtype=np.int8).reshape(-1)
    if len(info) < INFO_BITS:
        info = np.concatenate([info, np.zeros(INFO_BITS - len(info), dtype=np.int8)])
    info = info[:INFO_BITS]
    coded = interleave(encode_payload(info))
    block = np.concatenate([np.array(COSTAS, dtype=np.int8), coded])
    iq, *_rest = _cycle_iq(block, 1.0, 1.0, mode=mode)
    
    
    pad = np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.complex64)
    return np.concatenate([iq.astype(np.complex64), pad])



QSO_GAP_S = 1.00
QSO_F = (
    (0.00, pack_cq, ("K1ABC", "FN20"), 1.00),
    (None, pack_cq, ("K1ABC", "FN20"), 1.00),
    (None, pack_reply, ("K1ABC", "W1AW", "FN31"), 0.90),
    (None, pack_rpt, ("W1AW", "K1ABC", -8, False), 1.00),
    (None, pack_rpt, ("K1ABC", "W1AW", -8, True), 0.90),
    (None, pack_rr73, ("W1AW", "K1ABC"), 1.00),
    (None, pack_73, ("K1ABC", "W1AW"), 0.90),
)

QSO_F2 = (
    (0.00, pack_cq, ("N0QRP", "EM48"), 1.00),
    (None, pack_cq, ("N0QRP", "EM48"), 0.95),
    (None, pack_reply, ("N0QRP", "G0XYZ", "IO91"), 0.85),
    (None, pack_rpt, ("G0XYZ", "N0QRP", 6, False), 1.00),
    (None, pack_rpt, ("N0QRP", "G0XYZ", 6, True), 0.88),
    (None, pack_rr73, ("G0XYZ", "N0QRP"), 0.97),
    (None, pack_73, ("N0QRP", "G0XYZ"), 0.82),
)


def encode_qso(
    seconds: float | None = None,
    script=QSO_F,
    *,
    mode: str = "F",
) -> np.ndarray:
    bursts: list[tuple[float, np.ndarray]] = []
    last_end = 0.0
    for start, pack, args, gain in script:
        iq = frame_iq(pack(*args), mode=mode) * float(gain)
        t0 = float(start) if start is not None else last_end + QSO_GAP_S
        bursts.append((t0, iq))
        last_end = t0 + len(iq) / SAMPLE_RATE
    if seconds is None:
        seconds = last_end + QSO_GAP_S
    n = int(round(seconds * SAMPLE_RATE))
    out = np.zeros(n, dtype=np.complex64)
    for t0, iq in bursts:
        i0 = int(round(t0 * SAMPLE_RATE))
        i1 = min(n, i0 + len(iq))
        if i1 > i0:
            out[i0:i1] += iq[: i1 - i0]
    return out


def encode_qso_f(seconds: float | None = None, script=QSO_F) -> np.ndarray:
    return encode_qso(seconds=seconds, script=script, mode="F")


def encode_qso_l(seconds: float | None = None, script=QSO_F) -> np.ndarray:
    return encode_qso(seconds=seconds, script=script, mode="L")




UNSYNC_F5 = (
    ("K1ABC", "FN20", 0.18, -1000.0, 2.71, 1.00),
    ("W1AW", "FN31", 1.42, -500.0, 3.19, 0.92),
    ("N0QRP", "EM48", 0.61, 0.0, 2.88, 0.97),
    ("G0XYZ", "IO91", 2.04, 500.0, 3.36, 0.85),
    ("K6QSO", "CM87", 0.97, 1000.0, 2.64, 0.90),
)
UNSYNC_FL10 = tuple((*station, "F") for station in UNSYNC_F5) + (
    ("DL1ABC", "JO62", 0.32, 1500.0, 5.20, 0.95, "L"),
    ("F4ABC", "JN18", 1.18, 2000.0, 5.47, 0.90, "L"),
    ("JA1ABC", "PM95", 0.74, 2500.0, 5.71, 0.98, "L"),
    ("VK2ABC", "QF56", 1.63, 3000.0, 5.33, 0.88, "L"),
    ("ZL1ABC", "RF73", 0.09, 3500.0, 5.59, 0.93, "L"),
)
UNSYNC_F = (
    ("K1ABC", "FN20", 0.00, -960.0, 2.70, 1.00),
    ("W1AW", "FN31", 0.55, -360.0, 3.10, 0.95),
    ("N0QRP", "EM48", 1.15, 240.0, 2.85, 0.90),
    ("G0XYZ", "IO91", 1.70, 840.0, 3.40, 0.88),
)


def encode_unsync_f(
    stations: tuple[tuple[str, str, float, float, float, float], ...] = UNSYNC_F,
    seconds: float = 12.0,
) -> np.ndarray:
    n = int(round(seconds * SAMPLE_RATE))
    out = np.zeros(n, dtype=np.complex64)
    for call, grid, t0, df, period, gain in stations:
        block, _info = heartbeat_bits(call, grid)
        burst, *_rest = _cycle_iq(block, 1.0, 1.0, mode="F")
        burst = burst.astype(np.complex64) * float(gain)
        if abs(df) > 0.05:
            t = np.arange(len(burst), dtype=np.float64) / SAMPLE_RATE
            burst = (burst * np.exp(1j * 2.0 * np.pi * df * t)).astype(np.complex64)
        start = float(t0)
        while start < seconds:
            i0 = int(round(start * SAMPLE_RATE))
            i1 = min(n, i0 + len(burst))
            if i0 < n and i1 > i0:
                sl = burst[: i1 - i0]
                out[i0:i1] += sl
            start += period
    return out


def encode_unsync_fl(
    stations=UNSYNC_FL10,
    seconds: float = 60.0,
) -> np.ndarray:
    n = int(round(seconds * SAMPLE_RATE))
    out = np.zeros(n, dtype=np.complex64)
    for call, grid, t0, df, period, gain, mode in stations:
        block, _info = heartbeat_bits(call, grid)
        burst, *_rest = _cycle_iq(block, 1.0, 1.0, mode=mode)
        burst = burst.astype(np.complex64) * float(gain)
        if abs(df) > 0.05:
            t = np.arange(len(burst), dtype=np.float64) / SAMPLE_RATE
            burst = (burst * np.exp(1j * 2.0 * np.pi * df * t)).astype(np.complex64)
        start = float(t0)
        while start < seconds:
            i0 = int(round(start * SAMPLE_RATE))
            i1 = min(n, i0 + len(burst))
            if i0 < n and i1 > i0:
                out[i0:i1] += burst[: i1 - i0]
            start += float(period)
    return out


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(np.abs(iq) ** 2) + 1e-18
    n0 = p * (SAMPLE_RATE / 2500.0) / (10 ** (snr_db / 10))
    noise = np.sqrt(n0 / 2) * (
        rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
    )
    return (iq + noise).astype(np.complex64)


def data_only(iq: np.ndarray, mark_n: int, data_n: int) -> np.ndarray:
    cycle = mark_n + data_n
    chunks = []
    pos = 0
    while pos + cycle <= len(iq):
        chunks.append(iq[pos + mark_n : pos + mark_n + data_n])
        pos += cycle
    if not chunks:
        return iq[mark_n:] if len(iq) > mark_n else iq
    return np.concatenate(chunks)


def _unpack_info(info: np.ndarray) -> tuple[str, str, str] | None:
    return unpack_row(info.tolist())


def _try_block(sl: np.ndarray) -> tuple[str, str, str] | None:
    costas = np.array(COSTAS, dtype=np.int8)
    if len(sl) < prefix_len():
        return None
    if int(np.sum(sl[:8] == costas)) < 7:
        return None
    from lyra.fec import deinterleave

    info = decode_payload(deinterleave(sl[8 : prefix_len()]))
    if info is None:
        return None
    return _unpack_info(info)


def _costas_best(hard: np.ndarray) -> int:
    costas = np.array(COSTAS, dtype=np.int8)
    last = len(hard) - 7
    if last < 1:
        return 0
    return max(int(np.sum(hard[o : o + 8] == costas)) for o in range(last))


def _dual_bandpass(x: np.ndarray, fa: float, fb: float, hw: float = 55.0) -> np.ndarray:
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    keep = ((f >= fa - hw) & (f <= fa + hw)) | ((f >= fb - hw) & (f <= fb + hw))
    spec[~keep] = 0
    return np.fft.irfft(spec, n=n)


def _search_hard(hard: np.ndarray, block: np.ndarray) -> tuple[str, str, str] | None:
    costas = np.array(COSTAS, dtype=np.int8)
    last = max(1, len(hard) - prefix_len() + 1)
    scored: list[tuple[int, int]] = []
    for off in range(last):
        scored.append((int(np.sum(hard[off : off + 8] == costas)), off))
    scored.sort(reverse=True)
    for score, off in scored:
        if score < 7:
            break
        sl = hard[off : off + prefix_len()]
        got = _try_block(sl)
        if got:
            return got
    return None


_EXTRA = (
    {},
    {"invert_a": True, "invert_b": True},
    {"swap": True},
    {"swap": True, "invert_a": True, "invert_b": True},
    {"invert_a": True},
    {"invert_b": True},
)


def _stretch_bb(bb: np.ndarray, scale: float) -> np.ndarray:
    x = np.asarray(bb, dtype=np.complex128).reshape(-1)
    if abs(scale - 1.0) < 1e-6:
        return x
    n = int(round(len(x) * scale))
    if n < 32:
        return x
    t = np.linspace(0.0, len(x) - 1.0, n)
    idx = np.arange(len(x), dtype=np.float64)
    return np.interp(t, idx, x.real) + 1j * np.interp(t, idx, x.imag)


def _demod_bb(ba: np.ndarray, bb: np.ndarray, block: np.ndarray, df: float) -> tuple[int, tuple | None]:
    sps = int(round(BB_RATE / BIT_RATE))
    n_pairs = int(round(len(ba) / BB_RATE * BIT_RATE))
    if n_pairs * 2 < prefix_len():
        return 0, None
    best_sc = 0
    starts = list(range(0, sps, max(1, sps // 4)))
    for start in starts:
        hard = mux_demod_fm(ba, bb, n_pairs, start=start)
        sc = _costas_best(hard)
        if sc > best_sc:
            best_sc = sc
        got = _search_hard(hard, block)
        if got:
            return max(sc, 8), got
        if sc >= 7:
            for kw in _EXTRA[1:]:
                hard = mux_demod_fm(ba, bb, n_pairs, start=start, **kw)
                got = _search_hard(hard, block)
                if got:
                    return max(_costas_best(hard), 8), got
    hard = mux_demod_bb(ba, bb, n_pairs, start=0, df=df)
    got = _search_hard(hard, block)
    if got:
        return max(_costas_best(hard), 8), got
    return best_sc, None


def _refine_pair(iq: np.ndarray, fa0: float, fb0: float, block: np.ndarray) -> tuple[float, float, int, tuple | None]:
    ba = mix_decimate(iq, fa0)
    bb = mix_decimate(iq, fb0)
    if len(ba) < 32 or len(bb) < 32:
        return fa0, fb0, 0, None
    sc, got = _demod_bb(ba, bb, block, 0.0)
    if got:
        return fa0, fb0, max(sc, 8), got
    return fa0, fb0, sc, None


def _drop_harmonics(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for fa, fb in pairs:
        if any(abs(fa - 2.0 * a) < 80 and abs(fb - 2.0 * b) < 80 for a, b in out):
            continue
        out.append((fa, fb))
    return out


def decode_usb_all(audio: np.ndarray, block: np.ndarray | None = None) -> list[dict]:
    global LAST_STATUS
    from lyra import rx

    rows = rx.decode_many(audio)
    LAST_STATUS = rx.LAST_STATUS
    return rows


def decode_usb_audio(audio: np.ndarray, block: np.ndarray | None = None) -> tuple[str, str, str] | None:
    rows = decode_usb_all(audio, block)
    if not rows:
        return None
    return rows[0]["row"]


def decode_stream(iq: np.ndarray, block: np.ndarray, mark_n: int, data_n: int) -> tuple[str, str, str] | None:
    payload = data_only(iq, mark_n, data_n)
    n_pairs = int(round(len(payload) / SAMPLE_RATE * BIT_RATE))
    if n_pairs * 2 < prefix_len():
        return None
    hard = mux_demod(payload, n_pairs)
    bl = len(block)
    last = max(1, len(hard) - prefix_len() + 1)
    for off in range(min(bl, last)):
        sl = hard[off : off + min(bl, len(hard) - off)]
        got = _try_block(sl)
        if got:
            return got
    return None
