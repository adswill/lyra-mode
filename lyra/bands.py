from __future__ import annotations

import re

from lyra.const import RF_DIAL_HZ

DEFAULT_DIAL_HZ = int(RF_DIAL_HZ)
CUSTOM_LABEL = "Custom…"
PRESETS: tuple[tuple[str, int], ...] = (
    ("20m  14.1064 MHz", 14_106_400),
    ("30m  10.1440 MHz", 10_144_000),
    ("15m  21.1100 MHz", 21_110_000),
)


def format_mhz(hz: int) -> str:
    return f"{hz / 1e6:.4f}"


def parse_frequency(text: str) -> int | None:
    raw = str(text or "").strip().lower().replace(",", "")
    if not raw or raw.startswith("custom"):
        return None
    raw = re.sub(r"\b\d+m\b", " ", raw)
    mult = 1_000_000.0
    if "khz" in raw:
        mult = 1_000.0
        raw = raw.replace("khz", " ")
    elif "mhz" in raw:
        raw = raw.replace("mhz", " ")
        mult = 1_000_000.0
    elif re.search(r"(?<![a-z])hz(?![a-z])", raw):
        raw = re.sub(r"(?<![a-z])hz(?![a-z])", " ", raw)
        mult = 1.0
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if match is None:
        return None
    value = float(match.group())
    if mult == 1_000_000.0 and "mhz" not in str(text or "").lower():
        if value >= 1_000_000:
            hz = int(round(value))
        elif value >= 1_000:
            hz = int(round(value * 1_000))
        else:
            hz = int(round(value * 1_000_000))
    else:
        hz = int(round(value * mult))
    if hz < 1_800_000 or hz > 29_700_000:
        return None
    return hz
