
from __future__ import annotations

A37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A27 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"


T_CQ, T_GRID, T_RPT, T_RRPT, T_RR73, T_73, T_RRR = range(7)


def pack_callsign(call: str) -> int:
    s = call.strip().upper()
    if " " in s:
        s = s.split()[0]
    if not s:
        return 0
    if len(s) < 3:
        raise ValueError(f"callsign too short: {call!r}")
    if len(s) < 6 and len(s) >= 2 and s[1].isdigit():
        s = " " + s
    s = (s + "      ")[:6]
    n = A37.index(s[0])
    n = n * 36 + A36.index(s[1])
    n = n * 10 + (ord(s[2]) - 48)
    for ch in s[3:6]:
        n = n * 27 + A27.index(ch)
    if n >= (1 << 28):
        raise ValueError(f"callsign does not pack: {call!r}")
    return n


def unpack_callsign(n: int) -> str:
    if n <= 0:
        return ""
    chars = [" "] * 6
    for i in range(5, 2, -1):
        n, r = divmod(n, 27)
        chars[i] = A27[r]
    n, r = divmod(n, 10)
    chars[2] = chr(r + 48)
    n, r = divmod(n, 36)
    chars[1] = A36[r]
    chars[0] = A37[n]
    return "".join(chars).strip()


def pack_grid4(grid: str) -> int:
    g = grid.strip().upper()
    if len(g) < 4:
        raise ValueError(f"grid4 required: {grid!r}")
    lon = ord(g[0]) - 65
    lat = ord(g[1]) - 65
    lon2 = ord(g[2]) - 48
    lat2 = ord(g[3]) - 48
    return ((lon * 18 + lat) * 10 + lon2) * 10 + lat2


def unpack_grid4(n: int) -> str:
    n, lat2 = divmod(n, 10)
    n, lon2 = divmod(n, 10)
    n, lat = divmod(n, 18)
    lon = n
    return f"{chr(lon + 65)}{chr(lat + 65)}{lon2}{lat2}"


def _u(bits, v: int, n: int) -> None:
    for i in range(n - 1, -1, -1):
        bits.append((v >> i) & 1)


def _pack77(typ: int, call_a: str, call_b: str, field: int) -> list[int]:
    bits: list[int] = []
    _u(bits, typ, 3)
    _u(bits, pack_callsign(call_a) if call_a else 0, 28)
    _u(bits, pack_callsign(call_b) if call_b else 0, 28)
    _u(bits, int(field) & 0x7FFF, 15)
    _u(bits, 0, 3)
    return bits


def pack_cq(call: str, grid: str) -> list[int]:
    return _pack77(T_CQ, call, "", pack_grid4(grid))


def pack_reply(hiscall: str, mycall: str, grid: str) -> list[int]:
    return _pack77(T_GRID, hiscall, mycall, pack_grid4(grid))


def pack_rpt(hiscall: str, mycall: str, snr: int, roger: bool = False) -> list[int]:
    snr = int(max(-35, min(28, snr)))
    typ = T_RRPT if roger else T_RPT
    return _pack77(typ, hiscall, mycall, snr + 35)


def pack_rr73(hiscall: str, mycall: str) -> list[int]:
    return _pack77(T_RR73, hiscall, mycall, 0)


def pack_73(hiscall: str, mycall: str) -> list[int]:
    return _pack77(T_73, hiscall, mycall, 0)


def pack_rrr(hiscall: str, mycall: str) -> list[int]:
    return _pack77(T_RRR, hiscall, mycall, 0)


def unpack_message(bits: list[int]) -> dict:
    b = list(bits)[:77]
    if len(b) < 77:
        b = b + [0] * (77 - len(b))
    v = 0
    for bit in b:
        v = (v << 1) | int(bit)
    typ = (v >> 74) & 0x7
    call_a = unpack_callsign((v >> 46) & ((1 << 28) - 1))
    call_b = unpack_callsign((v >> 18) & ((1 << 28) - 1))
    field = (v >> 3) & ((1 << 15) - 1)
    if typ == T_CQ:
        return {"type": "CQ", "call": call_a, "grid": unpack_grid4(field)}
    if typ == T_GRID:
        return {"type": "GRID", "to": call_a, "frm": call_b, "grid": unpack_grid4(field)}
    if typ in (T_RPT, T_RRPT):
        snr = int(field) - 35
        tag = f"R{snr:+03d}" if typ == T_RRPT else f"{snr:+03d}"
        return {"type": "RPT", "to": call_a, "frm": call_b, "rpt": tag}
    if typ == T_RR73:
        return {"type": "RR73", "to": call_a, "frm": call_b}
    if typ == T_73:
        return {"type": "73", "to": call_a, "frm": call_b}
    if typ == T_RRR:
        return {"type": "RRR", "to": call_a, "frm": call_b}
    raise ValueError(f"unsupported type {typ}")


def unpack_row(bits: list[int] | bytes) -> tuple[str, str, str] | None:
    try:
        m = unpack_message(list(bits)[:77])
    except ValueError:
        return None
    typ = m.get("type")
    if typ == "CQ":
        return "CQ", str(m["call"]), str(m["grid"])
    if typ == "GRID":
        return str(m["to"]), str(m["frm"]), str(m["grid"])
    if typ == "RPT":
        return str(m["to"]), str(m["frm"]), str(m["rpt"])
    if typ == "RR73":
        return str(m["to"]), str(m["frm"]), "RR73"
    if typ == "73":
        return str(m["to"]), str(m["frm"]), "73"
    if typ == "RRR":
        return str(m["to"]), str(m["frm"]), "RRR"
    return None


def unpack_cq(bits: list[int] | bytes) -> tuple[str, str, str]:
    row = unpack_row(bits)
    if row is None or row[0] != "CQ":
        raise ValueError("not a CQ")
    return row
