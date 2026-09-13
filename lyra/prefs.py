from __future__ import annotations

from PySide6.QtCore import QSettings

from lyra.bands import DEFAULT_DIAL_HZ, parse_frequency

CQ_QUICKEST = "quickest"
CQ_FARTHEST = "farthest"
CQ_NEAREST = "nearest"
CQ_HIGHEST_SNR = "highest_snr"
CQ_LOWEST_SNR = "lowest_snr"
CQ_MANUAL = "manual"
DX_HIDDEN = "hidden"
DX_COUNTRY = "country"
DX_CONTINENT = "continent"
DX_BOTH = "both"

CQ_LABELS = (
    (CQ_QUICKEST, "Quickest response"),
    (CQ_FARTHEST, "Largest distance"),
    (CQ_NEAREST, "Smallest distance"),
    (CQ_HIGHEST_SNR, "Highest dB"),
    (CQ_LOWEST_SNR, "Lowest dB"),
    (CQ_MANUAL, "Manual pick"),
)
DX_LABELS = (
    (DX_HIDDEN, "Off"),
    (DX_COUNTRY, "Country"),
    (DX_CONTINENT, "Continent"),
    (DX_BOTH, "Country and continent"),
)
REPEAT_SKIP = "skip"
REPEAT_AGAIN = "again"
REPEAT_LABELS = (
    (REPEAT_SKIP, "Skip if already worked"),
    (REPEAT_AGAIN, "Answer again"),
)
ANSWER_F = "F"
ANSWER_L = "L"
ANSWER_BOTH = "both"
ANSWER_LABELS = (
    (ANSWER_BOTH, "Lyra F and L"),
    (ANSWER_F, "Lyra F only"),
    (ANSWER_L, "Lyra L only"),
)
COLOR_KEYS = (
    ("color_tx", "your tx", "#d4d4d4"),
    ("color_rx", "received", "#1a1a28"),
    ("color_qso", "this qso", "#2f4f38"),
)


def settings() -> QSettings:
    return QSettings("lyra", "lyra")


def _get(key: str, default: str) -> str:
    value = str(settings().value(key, default) or default).strip()
    return value or default


def cq_pick() -> str:
    value = _get("cq_pick", CQ_FARTHEST)
    allowed = {item[0] for item in CQ_LABELS}
    return value if value in allowed else CQ_FARTHEST


def dx_show() -> str:
    value = _get("dx_show", DX_COUNTRY)
    allowed = {item[0] for item in DX_LABELS}
    return value if value in allowed else DX_COUNTRY


def repeat_qso() -> str:
    value = _get("repeat_qso", REPEAT_SKIP)
    allowed = {item[0] for item in REPEAT_LABELS}
    return value if value in allowed else REPEAT_SKIP


def answer_mode() -> str:
    value = _get("answer_mode", ANSWER_BOTH)
    allowed = {item[0] for item in ANSWER_LABELS}
    return value if value in allowed else ANSWER_BOTH


def color(key: str) -> str:
    defaults = {item[0]: item[2] for item in COLOR_KEYS}
    value = _get(key, defaults.get(key, "#333333"))
    if not value.startswith("#") or len(value) not in (4, 7):
        return defaults.get(key, "#333333")
    return value


def station() -> tuple[str, str]:
    s = settings()
    call = str(s.value("call", "") or "").strip().upper()
    grid = str(s.value("grid", "") or "").strip().upper()
    return call or "K1ABC", grid or "FN20"


def save_station(call: str, grid: str) -> None:
    s = settings()
    s.setValue("call", call.strip().upper())
    s.setValue("grid", grid.strip().upper())
    s.sync()


def dial_hz() -> int:
    raw = settings().value("dial_hz", DEFAULT_DIAL_HZ)
    try:
        hz = int(float(raw))
    except (TypeError, ValueError):
        parsed = parse_frequency(str(raw or ""))
        hz = parsed if parsed is not None else DEFAULT_DIAL_HZ
    if hz < 1_800_000 or hz > 29_700_000:
        return DEFAULT_DIAL_HZ
    return hz


def save_dial(hz: int) -> None:
    s = settings()
    s.setValue("dial_hz", int(hz))
    s.sync()


def save_prefs(
    *,
    cq: str,
    dx: str,
    repeat: str | None = None,
    answer: str | None = None,
    colors: dict[str, str] | None = None,
) -> None:
    s = settings()
    s.setValue("cq_pick", cq)
    s.setValue("dx_show", dx)
    if repeat is not None:
        s.setValue("repeat_qso", repeat)
    if answer is not None:
        s.setValue("answer_mode", answer)
    if colors:
        for key, value in colors.items():
            s.setValue(key, value)
    s.sync()
