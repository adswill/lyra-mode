
from __future__ import annotations

import numpy as np

from lyra.const import CODED_BITS, CONV_TAIL, INFO_BITS

INTERLEAVE_COLS = 12

G0, G1, N_STATE = 0o171, 0o133, 64


def _parity(x: int) -> int:
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def crc16(bits: np.ndarray) -> np.ndarray:
    crc = 0xFFFF
    for b in bits:
        crc ^= int(b) << 15
        crc = ((crc << 1) & 0xFFFF) ^ 0x1021 if crc & 0x8000 else (crc << 1) & 0xFFFF
    return np.array([(crc >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)


def conv_encode(bits: np.ndarray) -> np.ndarray:
    reg, out = 0, []
    for b in list(bits) + [0] * CONV_TAIL:
        reg = ((reg << 1) | int(b)) & 0x7F
        out.append(_parity(reg & G0))
        out.append(_parity(reg & G1))
    return np.array(out, dtype=np.int8)


def encode_payload(info: np.ndarray) -> np.ndarray:
    return conv_encode(np.concatenate([info, crc16(info)]))


def interleave(bits: np.ndarray) -> np.ndarray:
    x = np.asarray(bits, dtype=np.int8).reshape(-1)
    pad = (-len(x)) % INTERLEAVE_COLS
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.int8)])
    return x.reshape(-1, INTERLEAVE_COLS).T.ravel()


def deinterleave(bits: np.ndarray, n: int = CODED_BITS) -> np.ndarray:
    pad = (-n) % INTERLEAVE_COLS
    m = n + pad
    y = np.asarray(bits, dtype=np.int8).reshape(-1)[:m]
    if len(y) < m:
        y = np.concatenate([y, np.zeros(m - len(y), dtype=np.int8)])
    return y.reshape(INTERLEAVE_COLS, -1).T.ravel()[:n]


def deinterleave_metrics(metrics: np.ndarray, n: int = CODED_BITS) -> np.ndarray:
    pad = (-n) % INTERLEAVE_COLS
    m = n + pad
    y = np.asarray(metrics, dtype=np.float64).reshape(-1, 2)[:m]
    if len(y) < m:
        y = np.concatenate((y, np.zeros((m - len(y), 2))), axis=0)
    return y.reshape(INTERLEAVE_COLS, -1, 2).transpose(1, 0, 2).reshape(-1, 2)[:n]


def viterbi(coded: np.ndarray) -> np.ndarray:
    n_pairs = len(coded) // 2
    inf = 1e9
    path_m = np.full(N_STATE, inf)
    path_m[0] = 0.0
    prev = np.zeros((n_pairs, N_STATE), dtype=np.int16)
    prev_bit = np.zeros((n_pairs, N_STATE), dtype=np.int8)
    for t in range(n_pairs):
        y0, y1 = coded[2 * t], coded[2 * t + 1]
        new_m = np.full(N_STATE, inf)
        for st in range(N_STATE):
            pm = path_m[st]
            if pm >= inf / 2:
                continue
            for bit in (0, 1):
                reg = ((st << 1) | bit) & 0x7F
                nst = reg & 0x3F
                e0, e1 = _parity(reg & G0), _parity(reg & G1)
                cost = pm
                if y0 >= 0:
                    cost += 0.0 if int(y0) == e0 else 1.0
                if y1 >= 0:
                    cost += 0.0 if int(y1) == e1 else 1.0
                if cost < new_m[nst]:
                    new_m[nst] = cost
                    prev[t, nst] = st
                    prev_bit[t, nst] = bit
        path_m = new_m
    st = int(np.argmin(path_m))
    bits = np.zeros(n_pairs, dtype=np.int8)
    for t in range(n_pairs - 1, -1, -1):
        bits[t] = prev_bit[t, st]
        st = int(prev[t, st])
    return bits[: INFO_BITS + 16]


def viterbi_soft(metrics: np.ndarray) -> np.ndarray:
    n_pairs = (metrics.shape[0] + 1) // 2
    inf = 1e18
    path_m = np.full(N_STATE, inf)
    path_m[0] = 0.0
    prev = np.zeros((n_pairs, N_STATE), dtype=np.int16)
    states = np.arange(N_STATE, dtype=np.int16)
    bits = (states & 1).astype(np.int8)
    
    predecessors = np.column_stack((states >> 1, (states >> 1) | 0x20))
    outputs = np.empty((N_STATE, 2, 2), dtype=np.int8)
    for nst in range(N_STATE):
        bit = int(bits[nst])
        for j, st in enumerate(predecessors[nst]):
            reg = ((int(st) << 1) | bit) & 0x7F
            outputs[nst, j] = (_parity(reg & G0), _parity(reg & G1))
    for t in range(n_pairs):
        m0 = metrics[2 * t] if 2 * t < len(metrics) else None
        m1 = metrics[2 * t + 1] if 2 * t + 1 < len(metrics) else None
        branch = np.zeros((N_STATE, 2), dtype=np.float64)
        if m0 is not None:
            branch -= m0[outputs[:, :, 0]]
        if m1 is not None:
            branch -= m1[outputs[:, :, 1]]
        candidates = path_m[predecessors] + branch
        choice = np.argmin(candidates, axis=1)
        path_m = candidates[states, choice]
        prev[t] = predecessors[states, choice]
    st = int(np.argmin(path_m))
    bits = np.zeros(n_pairs, dtype=np.int8)
    for t in range(n_pairs - 1, -1, -1):
        bits[t] = st & 1
        st = int(prev[t, st])
    return bits[: INFO_BITS + 16]


def decode_payload(coded: np.ndarray) -> np.ndarray | None:
    need = 2 * (INFO_BITS + 16 + CONV_TAIL)
    if coded.ndim == 2:
        decoded = viterbi_soft(coded)
    else:
        if len(coded) < need:
            coded = np.concatenate([coded, np.full(need - len(coded), -1)])
        decoded = viterbi(coded)
    info, crc = decoded[:INFO_BITS], decoded[INFO_BITS:]
    if not np.array_equal(crc, crc16(info)):
        return None
    return info
