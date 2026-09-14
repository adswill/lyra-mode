
from __future__ import annotations

import random
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, SimpleQueue

import numpy as np

from lyra.capture import AudioTap, default_input_index, list_inputs, load_sounddevice
from lyra.codec import decode_usb_all, heartbeat_bits
from lyra import codec as lyra_codec
from lyra.qso_auto import AutoAction, AutoQso, _is_grid, _is_rpt
from lyra.hamlib import LocalRigctld, bundled_hamlib_dir, find_rigctld, list_models, list_serial_ports
from lyra.rig import DummyRig, RigctldRig
from lyra.tx import TxSession, build_tx_audio, list_outputs, write_tx_wav
from lyra.bands import CUSTOM_LABEL, PRESETS, format_mhz, parse_frequency
from lyra.const import (
    BIT_RATE,
    CHIRP_S,
    CHANNELS,
    FRAME_BITS,
    SAMPLE_RATE,
    SPACING_HZ,
    TONE_A_HZ,
    TONE_B_HZ,
)
from lyra.pack import pack_cq

VIEW_LO = 300.0
VIEW_HI = 2_800.0
WF_NFFT = 8192
WF_ROWS = 360
WF_LEVELS = (-105.0, -55.0)
LIVE_DECODE_INTERVAL_S = 0.015
LIVE_AUDIO_S = 5.2
UI_RENDER_INTERVAL_S = 0.10
WF_INTERVAL_S = 0.125
CQ_PICK_MS = 5000
CQ_GAP_S = 6.0
CQ_SEEN_S = 5.4
ANSWER_CLEAR_S = 0.0
ANSWER_JITTER_S = 0.75
BUSY_HOLD_S = 8.0
CHANNEL_HOLD_S = 4.8
WATCH_ARM_S = 0.35
DECODE_GRACE_S = 0.45
ECHO_S = 5.5
CH_COLORS = ("#777777",) * 5
from lyra.modem import usb_spectrum


def _ensure_gui_deps() -> None:
    try:
        import PySide6  
        import pyqtgraph  
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "PySide6", "pyqtgraph"]
        )


_ensure_gui_deps()

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from lyra import geo
from lyra import prefs

CAND_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def _frame_s(mode: str) -> float:
    data = FRAME_BITS / BIT_RATE if str(mode).upper() == "L" else (FRAME_BITS / 2.0) / BIT_RATE
    return CHIRP_S + data


def _cq_start_time(now: float, audio_n: int, i0: object) -> float:
    if i0 is None or audio_n <= 0:
        return now
    return now + (int(i0) - audio_n) / SAMPLE_RATE


def _cq_end_time(now: float, audio_n: int, i0: object, mode: str) -> float:
    if i0 is None or audio_n <= 0:
        return now
    end_i = int(i0) + int(round(_frame_s(mode) * SAMPLE_RATE))
    return now + (end_i - audio_n) / SAMPLE_RATE


def _contrast_fg(bg: str) -> str:
    c = QColor(bg)
    if not c.isValid():
        return "#ffffff"
    lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#111111" if lum >= 148 else "#ffffff"


def _snr_db(audio: np.ndarray, fa: float | None = None, fb: float | None = None) -> float:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = min(len(x), 16384)
    if n < 512:
        return -99.0
    sl = x[-n:]
    spec = np.abs(np.fft.rfft(sl * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    band = (freqs >= VIEW_LO) & (freqs <= VIEW_HI)
    median_bin = float(np.median(spec[band]) + 1e-18) if np.any(band) else 1e-18
    
    
    
    noise_bin = median_bin / np.log(2.0)
    bin_hz = SAMPLE_RATE / n
    noise_2500 = noise_bin * (2500.0 / bin_hz)
    if fa is not None and fb is not None:
        hw = 18.0
        signal_mask = np.zeros(len(freqs), dtype=bool)
        for f0 in (fa, fb):
            signal_mask |= np.abs(freqs - f0) <= hw
        occupied = int(np.count_nonzero(signal_mask))
        sig = max(float(np.sum(spec[signal_mask])) - occupied * noise_bin, 1e-18)
        return float(10.0 * np.log10(sig / noise_2500 + 1e-12))
    peak = float(np.max(spec[band] if np.any(band) else spec) + 1e-18)
    return float(10.0 * np.log10(peak / noise_2500 + 1e-12))


def _jt_colormap() -> pg.ColorMap:
    stops = np.array([0.00, 0.14, 0.32, 0.50, 0.70, 0.86, 1.00])
    colors = np.array(
        [
            [0, 0, 48],
            [0, 50, 190],
            [0, 180, 230],
            [20, 210, 70],
            [255, 230, 0],
            [255, 150, 0],
            [255, 255, 255],
        ],
        dtype=np.float64,
    )
    return pg.ColorMap(stops, colors)


class DecodeWorker(QObject):
    decoded = Signal(dict)
    status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.block, _ = heartbeat_bits("K1ABC", "FN20")
        self.tap: AudioTap | None = None
        self.running = False
        self._thread: threading.Thread | None = None
        self.last = None
        self.last_t = 0.0
        self._seen_rows: dict[tuple, float] = {}
        self.decode_enabled = True
        self.mute_channel: int | None = None
        self.out_q: SimpleQueue = SimpleQueue()
        self._last_decode = 0.0
        self._last_seq = -1

    def start(self, device: int) -> str:
        self.stop()
        self._seen_rows.clear()
        self.tap = AudioTap(device, seconds=12.0)
        self.tap.start()
        self._last_seq = -1
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.tap.name

    def stop(self) -> None:
        self.running = False
        from lyra.rx import LOCK

        LOCK.reset()
        if self.tap is not None:
            self.tap.stop()
            self.tap = None

    def _push_rows(self, audio: np.ndarray, rows: list) -> None:
        mute = self.mute_channel
        mute_mid = None
        if mute is not None and 1 <= mute <= len(CHANNELS):
            fa, fb = CHANNELS[mute - 1]
            mute_mid = 0.5 * (fa + fb)
        for rec in rows:
            got = rec["row"]
            fa = float(rec.get("fa") or 0.0)
            fb = float(rec.get("fb") or 0.0)
            if mute_mid is not None and fa > 0.0 and fb > 0.0:
                if abs(0.5 * (fa + fb) - mute_mid) < 110.0:
                    continue
            mode = str(rec.get("mode") or "")
            now = time.monotonic()
            key = (got, round(fa), round(fb), mode)
            prev_t = self._seen_rows.get(key, -1e9)
            if now - prev_t < CQ_SEEN_S:
                continue
            self._seen_rows[key] = now
            cq_start = _cq_start_time(now, len(audio), rec.get("i0"))
            cq_end = _cq_end_time(now, len(audio), rec.get("i0"), mode)
            if len(self._seen_rows) > 512:
                self._seen_rows = {
                    k: t for k, t in self._seen_rows.items() if now - t < 30.0
                }
            self.last = key
            self.last_t = now
            kind, a, b = got
            msg = " ".join(p for p in (kind, a, b) if p)
            payload = {
                "utc": datetime.now(timezone.utc).strftime("%H%M%S"),
                "db": round(_snr_db(audio, fa or None, fb or None), 0),
                "fa": fa,
                "fb": fb,
                "msg": msg,
                "decoded": got,
                "mode": mode,
                "cq_start": cq_start,
                "cq_end": cq_end,
            }
            self.decoded.emit(payload)

    def decode_wav(self, path: Path) -> None:
        threading.Thread(target=self._decode_wav, args=(path,), daemon=True).start()

    def _decode_wav(self, path: Path) -> None:
        from lyra.listen import _read_wav_for_decode
        from lyra.rx import LOCK

        try:
            self.out_q.put(("status", f"Decoding {path.name}…"))
            LOCK.reset()
            audio = _read_wav_for_decode(path)
            rows = decode_usb_all(audio, self.block)
            self.out_q.put(("status", lyra_codec.LAST_STATUS or "no CRC"))
            if not rows:
                self.out_q.put(("status", lyra_codec.LAST_STATUS or "no decode"))
                return
            self.last = None
            self._push_rows(audio, rows)
        except Exception as e:
            self.out_q.put(("status", f"decode error: {type(e).__name__}: {e}"))

    def _loop(self) -> None:
        min_new = int(0.012 * SAMPLE_RATE)
        while self.running and self.tap is not None:
            if not self.decode_enabled:
                time.sleep(0.005)
                continue
            seq = self.tap.captured()
            if seq - self._last_seq < min_new:
                time.sleep(0.001)
                continue
            self._last_seq = seq
            audio = self.tap.latest(int(LIVE_AUDIO_S * SAMPLE_RATE))
            if audio is not None and len(audio) >= int(0.80 * SAMPLE_RATE):
                rows: list = []
                try:
                    rows = decode_usb_all(audio, self.block, live=True)
                except Exception as e:
                    self.out_q.put(("status", f"decode error: {type(e).__name__}"))
                    rows = []
                self._last_decode = time.monotonic()
                self._push_rows(audio, rows)
            else:
                time.sleep(0.001)


class TxBridge(QObject):
    status = Signal(str, bool)


class LyraWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("lyra")
        self.resize(1520, 940)
        self.worker = DecodeWorker()
        self.worker.decoded.connect(self._on_decode)
        self.tx_session = TxSession()
        self.tx_bridge = TxBridge()
        self.tx_bridge.status.connect(self._on_tx_status)
        self.rig = DummyRig()
        self._wf = None
        self._echo_until = 0.0
        self._echo_msg = ""
        self._echo_msgs: list[tuple[float, str]] = []
        self._started = False
        self._syncing = False
        self._ch_spec: list = []
        self._ch_wf: list = []
        self._mids: list[float] = [0.5 * (a + b) for a, b in CHANNELS[:3]]
        self._decode_count = 0
        self._channel_counts: dict[int, int] = {}
        self._channel_rows: dict[int, int] = {}
        self._channel_tones: dict[int, str | None] = {}
        self._channel_mode: dict[int, str] = {}
        self._channel_heard_at: dict[int, float] = {}
        self._pending_tx_log: tuple[AutoAction, int] | None = None
        self._last_render = 0.0
        self._last_wf = 0.0
        self._cq_pool: dict[str, dict] = {}
        self._worked: dict[str, float] = {}
        self._pending_answer: dict | None = None
        self._answer_armed_at = 0.0
        self._recent_rx: dict[tuple, float] = {}
        self._radio_hz: int | None = None
        self._radio_mode: str | None = None
        self._cq_pick_timer = QTimer(self)
        self._cq_pick_timer.setSingleShot(True)
        self._cq_pick_timer.timeout.connect(self._flush_cq_pool)
        self._cq_repeat_timer = QTimer(self)
        self._cq_repeat_timer.setSingleShot(True)
        self._cq_repeat_timer.timeout.connect(self._on_cq_repeat)
        self._cq_repeat_generation = 0
        self._watch_channel: int | None = None
        self._watch_arm_at = 0.0
        self._watch_hold_until = 0.0
        self._qso_call = ""
        self._build_menu()
        self._build()
        self._sync_channel_overview(self._channel_grid())
        self._load_devices()
        self._load_outputs()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick_clock)
        self.clock.start(250)
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._fast_watch)
        self._watch_timer.start(20)
        QTimer.singleShot(400, self._autostart)
        QTimer.singleShot(500, self._startup_notice)

    def _build_menu(self) -> None:
        file_m = self.menuBar().addMenu("&File")
        act_open = QAction("Open &WAV…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_wav)
        file_m.addAction(act_open)
        act_prefs = QAction("&Preferences…", self)
        act_prefs.setShortcut("Ctrl+,")
        act_prefs.triggered.connect(self._open_prefs)
        file_m.addAction(act_prefs)
        act_quit = QAction("E&xit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_m.addAction(act_quit)
        help_m = self.menuBar().addMenu("&Help")
        act_about = QAction("&About Lyra", self)
        act_about.triggered.connect(self._about)
        help_m.addAction(act_about)

    def _open_wav(self) -> None:
        path, _ok = QFileDialog.getOpenFileName(
            self,
            "Decode Lyra WAV",
            str(Path(__file__).resolve().parents[1] / "iq"),
            "WAV (*.wav);;All files (*)",
        )
        if not path:
            return
        self.decode_on.setChecked(True)
        self.worker.decode_enabled = True
        self.worker.decode_wav(Path(path))
        self.dev_lab.setText(f"File  {Path(path).name}")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About Lyra",
            f"Lyra — chirp + dual-rail GMSK on {format_mhz(self._dial_hz())} USB\n"
            "Lyra F (fast): slash high→low (right→left), unique bits per rail, ~2.3 s\n"
            "Lyra L (long): slash low→high (left→right), same bits both rails, ~4.3 s\n"
            f"{BIT_RATE:g} baud per rail, r=1/2 K=7 + CRC-16.\n\n"
            "Channels is a maximum. Lyra places 80 Hz bands on a 210 Hz USB grid inside a normal voice filter.",
        )

    def _dial_hz(self) -> int:
        return prefs.dial_hz()

    def _fill_dial_combo(self, combo: QComboBox) -> None:
        hz = self._dial_hz()
        combo.blockSignals(True)
        combo.clear()
        for label, preset in PRESETS:
            combo.addItem(label, preset)
        combo.addItem(CUSTOM_LABEL, None)
        idx = combo.findData(hz)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(combo.findData(None))
            combo.setEditText(format_mhz(hz))
        combo.blockSignals(False)

    def _hz_from_combo(self, combo: QComboBox) -> int | None:
        text = combo.currentText().strip()
        data = combo.currentData()
        idx = combo.currentIndex()
        if (
            isinstance(data, int)
            and data > 0
            and idx >= 0
            and text == combo.itemText(idx)
        ):
            return int(data)
        return parse_frequency(text)

    def _on_dial_chosen(self, _index: int = 0) -> None:
        if self.dial.currentData() is None and self.dial.currentText().strip() in ("", CUSTOM_LABEL):
            self.dial.setEditText(format_mhz(self._dial_hz()))
            return
        hz = self._hz_from_combo(self.dial)
        if hz is None:
            self._fill_dial_combo(self.dial)
            return
        self._apply_dial(hz)

    def _on_dial_typed(self) -> None:
        hz = parse_frequency(self.dial.currentText())
        if hz is None:
            self._fill_dial_combo(self.dial)
            return
        self._apply_dial(hz)

    def _apply_dial(self, hz: int) -> None:
        prefs.save_dial(hz)
        self._fill_dial_combo(self.dial)
        self._radio_hz = None
        if getattr(self.rig, "connected", False):
            try:
                self._prepare_radio()
            except Exception as exc:
                self.rig_status.setText(str(exc))

    def _make_dock(self, name: str, title: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        return dock

    def _build(self) -> None:
        pg.setConfigOptions(antialias=False, background="#000000", foreground="#777777")
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )
        shell = QWidget()
        shell.setObjectName("root")
        shell.setMaximumSize(0, 0)
        self.setCentralWidget(shell)

        listen = QWidget()
        top = QHBoxLayout(listen)
        top.setContentsMargins(8, 6, 8, 6)
        top.setSpacing(8)
        self.monitor = QCheckBox("Monitor")
        self.monitor.setChecked(True)
        self.monitor.toggled.connect(self._on_monitor)
        top.addWidget(self.monitor)

        self.decode_on = QCheckBox("Decode")
        self.decode_on.setChecked(True)
        self.decode_on.toggled.connect(self._on_decode_toggle)
        top.addWidget(self.decode_on)

        top.addWidget(self._vline())
        top.addWidget(QLabel("max ch"))
        self.n_ch = QSpinBox()
        self.n_ch.setRange(1, 10)
        self.n_ch.setValue(10)
        self.n_ch.setFixedWidth(56)
        self.n_ch.valueChanged.connect(self._on_n_channels)
        top.addWidget(self.n_ch)

        top.addWidget(self._vline())
        self.dial = QComboBox()
        self.dial.setEditable(True)
        self.dial.setMinimumWidth(168)
        self.dial.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._fill_dial_combo(self.dial)
        self.dial.activated.connect(self._on_dial_chosen)
        self.dial.lineEdit().editingFinished.connect(self._on_dial_typed)
        top.addWidget(self.dial)
        top.addWidget(QLabel("USB"))
        top.addStretch(1)
        top.addWidget(QLabel("audio"))
        self.dev = QComboBox()
        self.dev.setMinimumWidth(220)
        self.dev.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dev.currentIndexChanged.connect(self._on_device)
        top.addWidget(self.dev, 1)
        ref = QPushButton("refresh")
        ref.setFixedWidth(72)
        ref.clicked.connect(self._load_devices)
        top.addWidget(ref)
        top.addWidget(self._vline())
        top.addWidget(QLabel("rx"))
        self.vu = QProgressBar()
        self.vu.setRange(0, 100)
        self.vu.setTextVisible(False)
        self.vu.setFixedHeight(14)
        self.vu.setFixedWidth(140)
        top.addWidget(self.vu)
        self.rx_db = QLabel("dB  —")
        self.rx_db.setFixedWidth(72)
        top.addWidget(self.rx_db)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(6, 4, 6, 4)
        left_l.setSpacing(2)
        traffic_head = QHBoxLayout()
        self.band_cap = QLabel("activity   0")
        traffic_head.addWidget(self.band_cap)
        traffic_head.addStretch(1)
        self.pause_feed = QCheckBox("Pause feed")
        traffic_head.addWidget(self.pause_feed)
        self.answer_sel = QPushButton("work selected")
        self.answer_sel.clicked.connect(self._work_selected)
        traffic_head.addWidget(self.answer_sel)
        clear = QPushButton("clear")
        clear.setFixedWidth(64)
        clear.clicked.connect(self._clear_activity)
        traffic_head.addWidget(clear)
        left_l.addLayout(traffic_head)
        self.band = QTableWidget(0, 5)
        self.band.setHorizontalHeaderLabels(["time", "snr", "ch", "dx", "message"])
        self.band.verticalHeader().setVisible(False)
        self.band.setShowGrid(False)
        self.band.setAlternatingRowColors(True)
        self.band.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.band.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.band.doubleClicked.connect(self._work_selected)
        hh = self.band.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.band.setColumnWidth(0, 72)
        self.band.setColumnWidth(1, 44)
        self.band.setColumnWidth(2, 110)
        self.band.setColumnWidth(3, 120)
        self.band.verticalHeader().setDefaultSectionSize(24)
        self.band.horizontalHeader().setMinimumHeight(26)
        self._apply_dx_column()
        left_l.addWidget(self.band)

        qso_box = QWidget()
        qso_l = QVBoxLayout(qso_box)
        qso_l.setContentsMargins(6, 4, 6, 4)
        qso_l.setSpacing(4)
        self.qso_cap = QLabel("qso")
        qso_l.addWidget(self.qso_cap)
        self.qso = QTableWidget(0, 4)
        self.qso.setHorizontalHeaderLabels(["time", "snr", "ch", "message"])
        self.qso.verticalHeader().setVisible(False)
        self.qso.setShowGrid(False)
        self.qso.setAlternatingRowColors(True)
        self.qso.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.qso.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        qh = self.qso.horizontalHeader()
        qh.setStretchLastSection(True)
        qh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        qh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        qh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        qh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.qso.setColumnWidth(0, 78)
        self.qso.setColumnWidth(1, 52)
        self.qso.setColumnWidth(2, 110)
        self.qso.verticalHeader().setDefaultSectionSize(24)
        self.qso.horizontalHeader().setMinimumHeight(26)
        qso_l.addWidget(self.qso, 1)

        ch_box = QWidget()
        ch_l = QVBoxLayout(ch_box)
        ch_l.setContentsMargins(6, 4, 6, 4)
        ch_l.setSpacing(4)
        self.rx_cap = QLabel("channels")
        ch_l.addWidget(self.rx_cap)
        self.rx_table = QTableWidget(0, 7)
        self.rx_table.setHorizontalHeaderLabels(["ch", "mode", "hz", "last", "snr", "count", "busy"])
        self.rx_table.verticalHeader().setVisible(False)
        self.rx_table.setShowGrid(False)
        self.rx_table.setAlternatingRowColors(True)
        self.rx_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rx_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rh = self.rx_table.horizontalHeader()
        rh.setStretchLastSection(True)
        rh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        rh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        rh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        rh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        rh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.rx_table.setColumnWidth(0, 48)
        self.rx_table.setColumnWidth(1, 56)
        self.rx_table.setColumnWidth(2, 110)
        self.rx_table.setColumnWidth(3, 78)
        self.rx_table.setColumnWidth(4, 56)
        self.rx_table.setColumnWidth(5, 56)
        self.rx_table.verticalHeader().setDefaultSectionSize(24)
        self.rx_table.horizontalHeader().setMinimumHeight(26)
        ch_l.addWidget(self.rx_table, 1)

        graph = QWidget()
        graph_l = QVBoxLayout(graph)
        graph_l.setContentsMargins(0, 0, 0, 0)
        graph_l.setSpacing(0)
        graph.setMinimumHeight(220)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#000000")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setLabel("left", "dBFS")
        self.plot.setXRange(VIEW_LO, VIEW_HI, padding=0)
        self.plot.setYRange(-100, -10, padding=0)
        self.plot.disableAutoRange()
        self.plot.getViewBox().setLimits(xMin=200, xMax=2800, yMin=-130, yMax=0)
        self.plot.getAxis("bottom").setStyle(showValues=False)
        self.plot.setFixedHeight(130)
        self.curve = self.plot.plot(pen=pg.mkPen("#ffffff", width=1), clipToView=True)
        self.curve.setDownsampling(auto=False)
        graph_l.addWidget(self.plot)

        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setBackground("#000000")
        self.wf_plot.setLabel("bottom", "Hz")
        self.wf_plot.hideAxis("left")
        self.wf_plot.setXRange(VIEW_LO, VIEW_HI, padding=0)
        self.wf_plot.setYRange(0, WF_ROWS, padding=0)
        self.wf_plot.disableAutoRange()
        self.wf_plot.setXLink(self.plot)
        self.wf_plot.setMinimumHeight(180)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.wf_plot.addItem(self.img)
        self.img.setColorMap(_jt_colormap())
        if hasattr(self.img, "setTransformationMode"):
            self.img.setTransformationMode(Qt.TransformationMode.FastTransformation)
        if hasattr(self.img, "setAutoDownsample"):
            self.img.setAutoDownsample(False)
        graph_l.addWidget(self.wf_plot, 1)
        self._rebuild_channels(3)
        self._arm_auto()
        self._apply_theme()

        listen_dock = self._make_dock("dock_listen", "listen", listen)
        tx_dock = self._make_dock("dock_tx", "tx", self._build_tx_panel())
        act_dock = self._make_dock("dock_activity", "activity", left)
        qso_dock = self._make_dock("dock_qso", "qso", qso_box)
        ch_dock = self._make_dock("dock_channels", "channels", ch_box)
        graph_dock = self._make_dock("dock_graph", "graph", graph)
        graph_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._docks = (listen_dock, tx_dock, act_dock, qso_dock, ch_dock, graph_dock)

        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, listen_dock)
        self.splitDockWidget(listen_dock, tx_dock, Qt.Orientation.Vertical)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, graph_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, act_dock)
        self.splitDockWidget(graph_dock, act_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(act_dock, qso_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(qso_dock, ch_dock, Qt.Orientation.Vertical)

        view_m = self.menuBar().addMenu("&View")
        for dock in self._docks:
            view_m.addAction(dock.toggleViewAction())
        reset_lay = QAction("Reset layout", self)
        reset_lay.triggered.connect(self._reset_layout)
        view_m.addSeparator()
        view_m.addAction(reset_lay)

        geo = prefs.settings().value("win_geo")
        st = prefs.settings().value("win_docks")
        if geo is not None:
            self.restoreGeometry(geo)
        if st is not None:
            self.restoreState(st)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.utc_lab = QLabel("")
        self.dev_lab = QLabel("idle")
        sb.addWidget(self.utc_lab)
        sb.addWidget(self.dev_lab, 1)

    def _reset_layout(self) -> None:
        prefs.settings().remove("win_docks")
        prefs.settings().remove("win_geo")
        prefs.settings().sync()
        listen, tx, act, qso, ch, graph = self._docks
        for dock in self._docks:
            dock.setFloating(False)
            dock.show()
            self.removeDockWidget(dock)
            self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
            dock.show()
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, listen)
        self.splitDockWidget(listen, tx, Qt.Orientation.Vertical)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, graph)
        self.splitDockWidget(graph, act, Qt.Orientation.Vertical)
        self.splitDockWidget(act, qso, Qt.Orientation.Horizontal)
        self.splitDockWidget(qso, ch, Qt.Orientation.Vertical)

    def _build_tx_panel(self) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 6, 8, 7)
        row.setSpacing(7)

        row.addWidget(QLabel("call"))
        saved_call, saved_grid = prefs.station()
        self.my_call = QLineEdit(saved_call)
        self.my_call.setFixedWidth(82)
        self.my_call.editingFinished.connect(self._save_station)
        row.addWidget(self.my_call)
        row.addWidget(QLabel("grid"))
        self.my_grid = QLineEdit(saved_grid)
        self.my_grid.setFixedWidth(58)
        self.my_grid.editingFinished.connect(self._save_station)
        row.addWidget(self.my_grid)
        row.addWidget(QLabel("mode"))
        self.tx_mode = QComboBox()
        self.tx_mode.addItems(["F", "L"])
        row.addWidget(self.tx_mode)
        row.addWidget(QLabel("channel"))
        self.tx_channel = QSpinBox()
        self.tx_channel.setRange(1, 10)
        self.tx_channel.setValue(3)
        self.tx_channel.setFixedWidth(52)
        self.tx_channel.valueChanged.connect(self._paint_channel_regions)
        row.addWidget(self.tx_channel)
        row.addWidget(QLabel("option"))
        self.tx_operation = QComboBox()
        self.tx_operation.addItem("Auto", AutoQso.CALL_CQ)
        self.tx_operation.addItem("Answer", AutoQso.ANSWER_CQ)
        self.tx_operation.addItem("Manual", AutoQso.MANUAL_CQ)
        self.tx_operation.setMinimumWidth(215)
        row.addWidget(self.tx_operation)
        row.addWidget(QLabel("power"))
        self.tx_level = QSlider(Qt.Orientation.Horizontal)
        self.tx_level.setRange(1, 100)
        self.tx_level.setValue(10)
        self.tx_level.setFixedWidth(110)
        row.addWidget(self.tx_level)
        self.tx_level_value = QLabel("10%")
        self.tx_level_value.setFixedWidth(38)
        self.tx_level.valueChanged.connect(
            lambda value: self.tx_level_value.setText(f"{value}%")
        )
        row.addWidget(self.tx_level_value)
        setup = QPushButton("radio")
        setup.clicked.connect(self._open_tx_setup)
        row.addWidget(setup)
        self.tx_button = QPushButton("start")
        self.tx_button.clicked.connect(self._start_tx)
        row.addWidget(self.tx_button)
        self.tx_operation.currentIndexChanged.connect(self._on_operation_changed)
        self.tx_stop = QPushButton("stop")
        self.tx_stop.setEnabled(False)
        self.tx_stop.clicked.connect(self._stop_tx)
        row.addWidget(self.tx_stop)
        self.tx_light = QFrame()
        self.tx_light.setFixedSize(10, 10)
        row.addWidget(self.tx_light)
        self.tx_status = QLabel("TX off")
        row.addWidget(self.tx_status, 1)
        self._set_tx_light(False)

        
        self.rig_kind = QComboBox()
        self.rig_kind.addItem("Test", "dummy")
        self.rig_kind.addItem("Hamlib", "hamlib_local")
        self.rig_kind.addItem("Hamlib network", "rigctld")
        self.rig_kind.currentIndexChanged.connect(self._on_rig_kind)
        self.rig_host = QLineEdit("127.0.0.1")
        self.rig_host.setFixedWidth(105)
        self.rig_port = QSpinBox()
        self.rig_port.setRange(1, 65535)
        self.rig_port.setValue(4532)
        self.rig_port.setFixedWidth(76)
        self.rig_hamlib_path = QLineEdit()
        bundled = bundled_hamlib_dir()
        if bundled is not None:
            self.rig_hamlib_path.setText(str(bundled))
        self.rig_hamlib_path.setPlaceholderText("hamlib folder or rigctld")
        self.rig_browse = QPushButton("browse")
        self.rig_browse.clicked.connect(self._browse_hamlib)
        self.rig_search = QLineEdit()
        self.rig_search.setPlaceholderText("search name or id")
        self.rig_search.textChanged.connect(self._filter_rig_models)
        self.rig_model = QComboBox()
        self.rig_model.setMinimumWidth(220)
        self.rig_device = QComboBox()
        self.rig_device.setEditable(True)
        self.rig_device.setMinimumWidth(110)
        self.rig_baud = QComboBox()
        for rate in ("1200", "4800", "9600", "19200", "38400", "57600", "115200"):
            self.rig_baud.addItem(rate)
        self.rig_baud.setCurrentText("19200")
        self.rig_pktusb = QCheckBox("packet USB")
        self.rig_pktusb.setChecked(True)
        self.rig_pktusb.toggled.connect(lambda _on: setattr(self, "_radio_mode", None))
        self.rig_connect = QPushButton("connect")
        self.rig_connect.clicked.connect(self._connect_rig)
        self.rig_status = QLabel("test ready")
        self.rig_status.setMinimumWidth(125)
        self._hamlib = LocalRigctld()
        self._rig_models: list[tuple[int, str]] = []
        self.tx_dev = QComboBox()
        self.tx_dev.setMinimumWidth(230)
        self._setup_dialog = None
        self.auto_qso: AutoQso | None = None
        self.auto_active = False
        self._tx_label = ""
        self._on_rig_kind()
        self._on_operation_changed()
        return box

    def _vline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _load_devices(self) -> None:
        self._loading_dev = True
        self.dev.clear()
        sd = load_sounddevice()
        devices = list_inputs(sd)
        prefer = default_input_index(sd)
        sel = 0
        for n, (i, name, ch) in enumerate(devices):
            self.dev.addItem(f"{name}   (#{i}, {ch} ch)", i)
            if prefer is not None and i == prefer:
                sel = n
        if devices:
            self.dev.setCurrentIndex(sel)
        self._loading_dev = False

    def _load_outputs(self) -> None:
        self.tx_dev.clear()
        self.tx_dev.addItem("Test", None)
        try:
            sd = load_sounddevice()
            for idx, name, channels in list_outputs(sd):
                self.tx_dev.addItem(f"{name}   (#{idx}, {channels} ch)", idx)
            self.tx_dev.setCurrentIndex(0)
        except Exception as exc:
            self.tx_status.setText(f"Audio output unavailable: {exc}")

    def _on_rig_kind(self, _index: int = 0) -> None:
        kind = self.rig_kind.currentData()
        local = kind == "hamlib_local"
        network = kind == "rigctld"
        self.rig_host.setVisible(network)
        self.rig_host.setEnabled(network)
        self.rig_port.setVisible(local or network)
        self.rig_port.setEnabled(local or network)
        for widget in (
            self.rig_hamlib_path,
            self.rig_browse,
            self.rig_search,
            self.rig_model,
            self.rig_device,
            self.rig_baud,
        ):
            widget.setVisible(local)
        self.rig_pktusb.setVisible(local or network)
        if local:
            self._reload_serial_ports()
            self._reload_hamlib_models()
            self.rig_status.setText("not connected")
        elif network:
            self.rig_status.setText("not connected")
        else:
            self.rig_status.setText("test ready")

    def _apply_radio_mode(self) -> None:
        want = "PKTUSB" if self.rig_pktusb.isChecked() else "USB"
        last = None
        for mode, width in ((want, 3000), (want, 6000), ("USB", 3000), ("USB", 6000)):
            try:
                self.rig.set_mode(mode, width)
                self._radio_mode = mode
                return
            except Exception as exc:
                last = exc
        if last is not None:
            raise last

    def _prepare_radio(self) -> None:
        hz = self._dial_hz()
        want = "PKTUSB" if self.rig_pktusb.isChecked() else "USB"
        if self._radio_hz != hz:
            self.rig.set_frequency(hz)
            self._radio_hz = hz
        if self._radio_mode != want:
            self._apply_radio_mode()

    def _connect_rig(self) -> bool:
        try:
            self.rig.disconnect()
        except Exception:
            pass
        kind = self.rig_kind.currentData()
        try:
            if kind == "hamlib_local":
                exe = find_rigctld(self.rig_hamlib_path.text())
                if not exe:
                    raise RuntimeError("rigctld not found. Use the bundled hamlib folder, or browse to one.")
                model = self.rig_model.currentData()
                if model is None:
                    typed = self.rig_search.text().strip()
                    if typed.isdigit():
                        model = int(typed)
                if model is None:
                    raise RuntimeError("Pick a radio from the list, or type its hamlib id.")
                device = self.rig_device.currentText().strip()
                if not device:
                    raise RuntimeError("Set the COM port or serial device.")
                conf = "ptt_type=RIG"
                if self.rig_pktusb.isChecked():
                    conf += ",dmode=PKTUSB,dmode_comp=1"
                self._hamlib.start(
                    exe=exe,
                    model=int(model),
                    device=device,
                    baud=int(self.rig_baud.currentText()),
                    port=self.rig_port.value(),
                    conf=conf,
                )
                self.rig = RigctldRig("127.0.0.1", self.rig_port.value())
            elif kind == "rigctld":
                self._hamlib.stop()
                self.rig = RigctldRig(self.rig_host.text().strip(), self.rig_port.value())
            else:
                self._hamlib.stop()
                self.rig = DummyRig()
            self.rig.connect()
            self._radio_hz = None
            self._radio_mode = None
            self._prepare_radio()
        except Exception as exc:
            self.rig_status.setText("Connection failed")
            QMessageBox.warning(self, "Lyra rig control", str(exc))
            return False
        self.rig_status.setText(f"{self.rig.name} connected")
        return True

    def _browse_hamlib(self) -> None:
        path, _ok = QFileDialog.getExistingDirectory(self, "hamlib folder")
        if not path:
            return
        self.rig_hamlib_path.setText(path)
        self._reload_hamlib_models()

    def _reload_serial_ports(self) -> None:
        current = self.rig_device.currentText()
        self.rig_device.blockSignals(True)
        self.rig_device.clear()
        ports = list_serial_ports()
        if not ports:
            ports = ["COM3"] if sys.platform == "win32" else ["/dev/ttyUSB0"]
        for port in ports:
            self.rig_device.addItem(port)
        if current:
            idx = self.rig_device.findText(current)
            if idx >= 0:
                self.rig_device.setCurrentIndex(idx)
            else:
                self.rig_device.setEditText(current)
        self.rig_device.blockSignals(False)

    def _reload_hamlib_models(self) -> None:
        exe = find_rigctld(self.rig_hamlib_path.text())
        if not exe:
            self._rig_models = []
            self._filter_rig_models()
            return
        try:
            self._rig_models = list_models(exe)
        except Exception as exc:
            self._rig_models = []
            self.rig_status.setText(str(exc))
        self._filter_rig_models()

    def _filter_rig_models(self, _text: str = "") -> None:
        q = self.rig_search.text().strip().lower()
        current = self.rig_model.currentData()
        self.rig_model.blockSignals(True)
        self.rig_model.clear()
        for mid, label in self._rig_models:
            if q and q not in label.lower() and q != str(mid):
                continue
            self.rig_model.addItem(label, mid)
        if current is not None:
            idx = self.rig_model.findData(current)
            if idx >= 0:
                self.rig_model.setCurrentIndex(idx)
        if q.isdigit():
            idx = self.rig_model.findData(int(q))
            if idx >= 0:
                self.rig_model.setCurrentIndex(idx)
            elif self.rig_model.count() == 0:
                self.rig_model.addItem(q, int(q))
        self.rig_model.blockSignals(False)

    def _on_operation_changed(self, _index: int = 0) -> None:
        manual = self.tx_operation.currentData() == AutoQso.MANUAL_CQ
        self.tx_button.setText("start one qso" if manual else "start")
        self._sync_tx_channel_enabled()

    def _sync_tx_channel_enabled(self) -> None:
        answering = self.tx_operation.currentData() == AutoQso.ANSWER_CQ
        self.tx_channel.setEnabled(not answering and not self.auto_active)

    def _open_tx_setup(self) -> None:
        if self._setup_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("radio")
            dialog.setModal(False)
            dialog.resize(640, 250)
            lay = QVBoxLayout(dialog)
            rig_row = QHBoxLayout()
            rig_row.addWidget(QLabel("rig"))
            rig_row.addWidget(self.rig_kind)
            rig_row.addWidget(self.rig_host)
            rig_row.addWidget(self.rig_port)
            rig_row.addWidget(self.rig_connect)
            lay.addLayout(rig_row)
            ham_path = QHBoxLayout()
            ham_path.addWidget(QLabel("hamlib"))
            ham_path.addWidget(self.rig_hamlib_path, 1)
            ham_path.addWidget(self.rig_browse)
            lay.addLayout(ham_path)
            ham_model = QHBoxLayout()
            ham_model.addWidget(QLabel("radio"))
            ham_model.addWidget(self.rig_search, 1)
            ham_model.addWidget(self.rig_model, 2)
            lay.addLayout(ham_model)
            ham_port = QHBoxLayout()
            ham_port.addWidget(QLabel("port"))
            ham_port.addWidget(self.rig_device, 1)
            ham_port.addWidget(QLabel("baud"))
            ham_port.addWidget(self.rig_baud)
            ham_port.addWidget(self.rig_pktusb)
            lay.addLayout(ham_port)
            lay.addWidget(self.rig_status)
            audio_row = QHBoxLayout()
            audio_row.addWidget(QLabel("audio"))
            audio_row.addWidget(self.tx_dev, 1)
            refresh = QPushButton("refresh")
            refresh.clicked.connect(self._load_outputs)
            audio_row.addWidget(refresh)
            export = QPushButton("save wav")
            export.clicked.connect(self._export_tx)
            audio_row.addWidget(export)
            lay.addLayout(audio_row)
            close = QPushButton("close")
            close.clicked.connect(dialog.close)
            lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
            self._setup_dialog = dialog
        self._setup_dialog.show()
        self._setup_dialog.raise_()
        self._setup_dialog.activateWindow()

    def _make_tx_audio(self, action: AutoAction) -> np.ndarray:
        return build_tx_audio(
            action.bits,
            mode=self.tx_mode.currentText(),
            channel=self.tx_channel.value(),
            level=self.tx_level.value() / 100.0,
        )

    def _start_tx(self) -> None:
        if self.auto_active or self.tx_session.busy:
            return
        try:
            controller = AutoQso(
                self.my_call.text(),
                self.my_grid.text(),
                str(self.tx_operation.currentData()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Lyra automatic TX", f"Check call and grid:\n{exc}")
            return
        if not getattr(self.rig, "connected", False) and not self._connect_rig():
            return
        if self.rig_kind.currentData() in ("rigctld", "hamlib_local"):
            manual = controller.operation == AutoQso.MANUAL_CQ
            answer = QMessageBox.question(
                self,
                "Start one QSO?" if manual else "Start automatic transmission?",
                "Lyra will call CQ and key the radio until one QSO is complete."
                if manual
                else "Lyra will key the connected radio automatically until Stop is pressed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.auto_qso = controller
        self.auto_active = True
        self._reset_qso_log()
        self._cq_pool.clear()
        self._cq_pick_timer.stop()
        self._cq_repeat_timer.stop()
        self._pending_answer = None
        self._answer_armed_at = (
            time.monotonic() if controller.operation == AutoQso.ANSWER_CQ else 0.0
        )
        self._clear_channel_watch()
        self._auto_generation = getattr(self, "_auto_generation", 0) + 1
        for control in (
            self.my_call,
            self.my_grid,
            self.tx_mode,
            self.tx_channel,
            self.tx_operation,
        ):
            control.setEnabled(False)
        self.tx_button.setEnabled(False)
        self.tx_stop.setEnabled(True)
        self.tx_status.setText("TX on")
        action = controller.start()
        if action is None:
            if not self.monitor.isChecked():
                self.monitor.setChecked(True)
            if prefs.cq_pick() == prefs.CQ_MANUAL:
                self.tx_status.setText("pick a CQ")
            else:
                self.tx_status.setText("wait CQ")
        else:
            self._send_action(action)
            if prefs.cq_pick() == prefs.CQ_MANUAL:
                self.tx_status.setText("pick a reply")

    def _send_action(self, action: AutoAction) -> None:
        if not self.auto_active or self.tx_session.busy:
            return
        try:
            self._prepare_radio()
            audio = self._make_tx_audio(action)
            self._tx_label = action.label
            self._echo_msg = str(action.label or "").strip().upper()
            self._echo_msgs.append((time.monotonic(), self._echo_msg))
            self._pending_tx_log = (action, self.tx_channel.value())
            self.worker.mute_channel = int(self.tx_channel.value())
            self.tx_status.setText("TX on")
            self._set_tx_light(True)
            self.tx_session.start(
                audio,
                output_device=self.tx_dev.currentData(),
                rig=self.rig,
                callback=lambda message, done: self.tx_bridge.status.emit(message, done),
            )
        except Exception as exc:
            self._stop_tx()
            QMessageBox.warning(self, "Lyra automatic TX", str(exc))

    def _stop_tx(self) -> None:
        self.auto_active = False
        self.auto_qso = None
        self._pending_tx_log = None
        self._cq_pool.clear()
        self._cq_pick_timer.stop()
        self._cq_repeat_timer.stop()
        self._pending_answer = None
        self._answer_armed_at = 0.0
        self._clear_channel_watch()
        self._auto_generation = getattr(self, "_auto_generation", 0) + 1
        self.tx_session.stop()
        self.worker.decode_enabled = self.decode_on.isChecked()
        self._unmute_tx_channel()
        self._set_tx_light(False)
        self.tx_status.setText("TX off")
        self.tx_button.setEnabled(True)
        self.tx_stop.setEnabled(False)
        for control in (
            self.my_call,
            self.my_grid,
            self.tx_mode,
            self.tx_channel,
            self.tx_operation,
        ):
            control.setEnabled(True)
        self._sync_tx_channel_enabled()

    def _on_tx_status(self, message: str, done: bool) -> None:
        if not done:
            self.worker.decode_enabled = self.decode_on.isChecked()
            self._set_tx_light(True)
            if self.auto_active:
                self.tx_status.setText("TX on")
            return
        self.worker.decode_enabled = self.decode_on.isChecked()
        self._set_tx_light(False)
        self._echo_until = time.monotonic() + ECHO_S
        if self.worker.tap is not None:
            self.worker.tap.clear()
        QTimer.singleShot(400, self._unmute_tx_channel)
        pending = self._pending_tx_log
        self._pending_tx_log = None
        if message == "Transmission complete" and pending is not None:
            action, channel = pending
            self._echo_msg = str(action.label or "").strip().upper()
            fa, fb = CHANNELS[channel - 1]
            self._on_decode(
                {
                    "utc": datetime.now(timezone.utc).strftime("%H%M%S"),
                    "fa": fa,
                    "fb": fb,
                    "msg": action.label,
                    "origin": "tx",
                    "snr_text": "TX",
                }
            )
        if not self.auto_active or self.auto_qso is None:
            self.tx_status.setText("TX off")
            return
        self.tx_status.setText("TX on")
        if self.auto_qso.state == "calling":
            if pending is not None:
                self._arm_channel_watch(int(pending[1]))
            self._schedule_cq_repeat()
        elif self.auto_qso.state == "complete":
            generation = self._auto_generation
            QTimer.singleShot(2500, lambda g=generation: self._auto_continue(g, True))
            self._mark_worked(self.auto_qso.target)

    def _auto_continue(self, generation: int, completed: bool) -> None:
        if (
            not self.auto_active
            or self.auto_qso is None
            or generation != self._auto_generation
            or self.tx_session.busy
        ):
            return
        if not completed and self._cq_pool:
            self._flush_cq_pool()
            if self.tx_session.busy or (self.auto_qso is not None and self.auto_qso.target):
                return
        if not completed:
            if self.auto_qso is not None and self.auto_qso.target:
                return
            self._clear_channel_watch()
        action = self.auto_qso.resume() if completed else self.auto_qso.repeat_cq()
        if action is not None:
            self._send_action(action)
        else:
            self.tx_status.setText("TX on")

    def _reply_window_s(self) -> float:
        return CQ_GAP_S

    def _schedule_cq_repeat(self, delay_s: float | None = None) -> None:
        if not self.auto_active or self.auto_qso is None or self.auto_qso.target:
            self._cq_repeat_timer.stop()
            return
        self._cq_repeat_generation = self._auto_generation
        wait = self._reply_window_s() if delay_s is None else float(delay_s)
        self._cq_repeat_timer.start(max(200, int(wait * 1000)))

    def _on_cq_repeat(self) -> None:
        if not self.auto_active or self.auto_qso is None:
            return
        if self.auto_qso.target or self.tx_session.busy:
            self._schedule_cq_repeat(0.25)
            return
        self._auto_continue(self._cq_repeat_generation, False)

    def _clear_channel_watch(self) -> None:
        self._watch_channel = None
        self._watch_arm_at = 0.0
        self._watch_hold_until = 0.0

    def _arm_channel_watch(self, channel: int) -> None:
        self._watch_channel = max(1, min(len(CHANNELS), int(channel)))
        self._watch_arm_at = time.monotonic() + WATCH_ARM_S
        self._watch_hold_until = 0.0

    def _channel_index(self, fa: float, fb: float) -> int | None:
        if fa <= 0.0 or fb <= 0.0:
            return None
        mid = 0.5 * (fa + fb)
        idx = min(
            range(len(CHANNELS)),
            key=lambda i: abs(mid - 0.5 * (CHANNELS[i][0] + CHANNELS[i][1])),
        )
        return idx + 1

    def _channel_energy_busy(self, freqs: np.ndarray, mag: np.ndarray, fa: float, fb: float) -> bool:
        if freqs is None or mag is None or len(freqs) < 8:
            return False
        near = (freqs >= min(fa, fb) - 160.0) & (freqs <= max(fa, fb) + 160.0)
        rails = (np.abs(freqs - fa) <= 22.0) | (np.abs(freqs - fb) <= 22.0)
        noise_bins = mag[near & ~rails]
        if len(noise_bins) < 4:
            noise_bins = mag
        noise = float(np.median(noise_bins))

        def peak(f0: float) -> float:
            mask = np.abs(freqs - f0) <= 18.0
            if not np.any(mask):
                return -120.0
            return float(np.max(mag[mask]))

        pa, pb = peak(fa), peak(fb)
        return pa >= noise + 9.0 and pb >= noise + 9.0 and abs(pa - pb) <= 12.0

    def _channel_still_busy(self) -> bool:
        if self._watch_channel is None or self.worker.tap is None:
            return False
        sl = self.worker.tap.latest(4096)
        if sl is None or len(sl) < 1024:
            return False
        nfft = 1 << int(np.ceil(np.log2(max(1024, min(len(sl), 4096)))))
        freqs, mag = usb_spectrum(sl, nfft=nfft)
        band = (freqs >= VIEW_LO) & (freqs <= VIEW_HI)
        freqs = freqs[band]
        mag = np.asarray(mag[band], dtype=np.float32)
        fa, fb = CHANNELS[self._watch_channel - 1]
        return self._channel_energy_busy(freqs, mag, fa, fb)

    def _fast_watch(self) -> None:
        self._try_pending_answer()
        tap = self.worker.tap
        if tap is None or self._watch_channel is None:
            return
        sl = tap.latest(4096)
        if sl is None or len(sl) < 1024:
            return
        nfft = 1 << int(np.ceil(np.log2(max(1024, min(len(sl), 4096)))))
        freqs, mag = usb_spectrum(sl, nfft=nfft)
        band = (freqs >= VIEW_LO) & (freqs <= VIEW_HI)
        self._watch_cq_channel(freqs[band], np.asarray(mag[band], dtype=np.float32))

    def _watch_cq_channel(self, freqs: np.ndarray, mag: np.ndarray) -> None:
        return

    def _watch_other_traffic(self, decoded: tuple[str, str, str], fa: float, fb: float) -> None:
        if (
            not self.auto_active
            or self.auto_qso is None
            or self.auto_qso.state != "calling"
            or self.auto_qso.target
            or self.tx_session.busy
            or self._watch_channel is None
        ):
            return
        ch = self._channel_index(fa, fb)
        if ch != self._watch_channel:
            return
        first, second, _field = (str(x).strip().upper() for x in decoded)
        mine = self.auto_qso.my_call
        if second == mine or first == mine or first == "CQ":
            return
        return

    def _auto_hear(
        self,
        decoded: tuple[str, str, str],
        snr: int,
        fa: float = 0.0,
        fb: float = 0.0,
        mode: str = "",
        cq_end: float | None = None,
        cq_start: float | None = None,
    ) -> None:
        if not self.auto_active or self.auto_qso is None or self.tx_session.busy:
            return
        first, second, field = (str(x).strip().upper() for x in decoded)
        candidate = self._candidate_from_decode(
            decoded, snr, fa, fb, mode, cq_end, cq_start
        )
        if candidate is None and self._ignore_repeat(decoded):
            return
        if candidate is None and self._reject_answer_mode(decoded, mode):
            return
        if candidate is not None:
            self._note_candidate(candidate)
            pick = prefs.cq_pick()
            if pick == prefs.CQ_MANUAL:
                self.tx_status.setText(
                    "pick a CQ" if candidate["kind"] == "cq" else "pick a reply"
                )
                return
            self._work_candidate(candidate)
            if self.auto_qso is not None and self.auto_qso.target:
                self._cq_repeat_timer.stop()
                self._clear_channel_watch()
            return
        previous_state = self.auto_qso.state
        action = self.auto_qso.hear(decoded, snr)
        if self.auto_qso.state == "complete":
            self._mark_worked(self.auto_qso.target)
        if action is not None:
            self.tx_status.setText("TX on")
            self._cq_repeat_timer.stop()
            self._send_if_current(
                action, self.auto_qso, self.auto_qso.state, self.auto_qso.target
            )
        elif self.auto_qso.state == "complete":
            target = self.auto_qso.target
            self._mark_worked(target)
            if self.auto_qso.operation == AutoQso.MANUAL_CQ:
                self._finish_manual_qso(target)
            else:
                generation = self._auto_generation
                self.tx_status.setText("TX on")
                QTimer.singleShot(2500, lambda g=generation: self._auto_continue(g, True))
        elif previous_state != self.auto_qso.state and self.auto_qso.state == "listening":
            self.tx_status.setText("TX on")

    def _candidate_from_decode(
        self,
        decoded: tuple[str, str, str],
        snr: int,
        fa: float,
        fb: float,
        mode: str = "",
        cq_end: float | None = None,
        cq_start: float | None = None,
    ) -> dict | None:
        if self.auto_qso is None or self.auto_qso.target:
            return None
        first, second, field = (str(x).strip().upper() for x in decoded)
        if (
            self.auto_qso.operation == AutoQso.ANSWER_CQ
            and self.auto_qso.state == "listening"
            and first == "CQ"
            and second
            and second != self.auto_qso.my_call
        ):
            if self._is_worked(second) and not self._repeat_ok():
                return None
            if not self._answer_mode_ok(mode):
                return None
            ended = float(cq_end) if cq_end is not None else time.monotonic()
            started = float(cq_start) if cq_start is not None else ended - _frame_s(mode)
            return {
                "kind": "cq",
                "call": second,
                "grid": field,
                "snr": int(snr),
                "fa": float(fa or 0.0),
                "fb": float(fb or 0.0),
                "mode": str(mode or "").upper(),
                "t": time.monotonic(),
                "cq_start": started,
                "cq_end": ended,
            }
        if (
            self.auto_qso.state == "calling"
            and first == self.auto_qso.my_call
            and second
            and second != self.auto_qso.my_call
            and (_is_grid(field) or _is_rpt(field))
        ):
            if self._is_worked(second) and not self._repeat_ok():
                return None
            return {
                "kind": "reply",
                "call": second,
                "grid": field if _is_grid(field) else "",
                "roger": _is_rpt(field),
                "snr": int(snr),
                "fa": float(fa or 0.0),
                "fb": float(fb or 0.0),
                "mode": str(mode or "").upper(),
                "t": time.monotonic(),
            }
        return None

    def _note_candidate(self, candidate: dict) -> None:
        key = f"{candidate['kind']}:{candidate['call']}"
        prev = self._cq_pool.get(key)
        if prev is None or candidate["t"] >= prev["t"]:
            self._cq_pool[key] = candidate

    def _flush_cq_pool(self) -> None:
        if not self.auto_active or self.auto_qso is None or self.auto_qso.target:
            self._cq_pool.clear()
            return
        if not self._cq_pool:
            return
        pick = prefs.cq_pick()
        my_grid = self.my_grid.text()
        pool = [
            item
            for item in self._cq_pool.values()
            if (self._repeat_ok() or not self._is_worked(item.get("call") or ""))
            and self._answer_mode_ok(item.get("mode") or "")
        ]
        self._cq_pool.clear()
        if not pool:
            return

        def score(item: dict) -> float:
            dist = geo.distance_km(my_grid, item.get("grid") or "")
            if pick == prefs.CQ_QUICKEST:
                return -float(item["t"])
            if pick == prefs.CQ_FARTHEST:
                return float(dist) if dist is not None else -1.0
            if pick == prefs.CQ_NEAREST:
                return -float(dist) if dist is not None else -1e18
            if pick == prefs.CQ_HIGHEST_SNR:
                return float(item["snr"])
            return -float(item["snr"])

        self._work_candidate(max(pool, key=score))

    def _cq_heard_after_arm(self, candidate: dict) -> bool:
        armed = float(self._answer_armed_at or 0.0)
        if armed <= 0.0:
            return False
        start = float(candidate.get("cq_start") or 0.0)
        if start <= 0.0:
            mode = str(candidate.get("mode") or "F")
            start = float(candidate.get("cq_end") or candidate.get("t") or 0.0) - _frame_s(mode)
        return start >= armed

    def _work_candidate(self, candidate: dict, *, forced: bool = False) -> None:
        if not self.auto_active or self.auto_qso is None or self.tx_session.busy:
            return
        call = str(candidate.get("call") or "").strip().upper()
        if call and self._is_worked(call) and not self._repeat_ok():
            return
        if candidate.get("kind") == "cq":
            if not forced and not self._cq_heard_after_arm(candidate):
                return
            ch = self._channel_index(
                float(candidate.get("fa") or 0.0),
                float(candidate.get("fb") or 0.0),
            )
            if ch is not None:
                self.tx_channel.setValue(ch)
            mode = str(candidate.get("mode") or "F").upper()
            if mode not in ("F", "L"):
                mode = "F"
            self.tx_mode.setCurrentText(mode)
            self._cq_pool.clear()
            self._cq_pick_timer.stop()
            now = time.monotonic()
            jitter = random.uniform(0.0, ANSWER_JITTER_S)
            self._pending_answer = {
                "candidate": candidate,
                "jitter_s": jitter,
                "tx_at": now + jitter,
            }
            self._try_pending_answer()
            return
        action = self.auto_qso.begin_accept(
            candidate["call"], candidate["snr"], roger=bool(candidate.get("roger"))
        )
        if action is None:
            return
        self._bind_qso(candidate["call"])
        self._cq_pool.clear()
        self._cq_pick_timer.stop()
        self._cq_repeat_timer.stop()
        self._pending_answer = None
        self.tx_status.setText("TX on")
        self._send_if_current(
            action, self.auto_qso, self.auto_qso.state, self.auto_qso.target
        )

    def _work_selected(self, *_args) -> None:
        row = self.band.currentRow()
        if row < 0:
            return
        item = self.band.item(row, 0)
        candidate = item.data(CAND_ROLE) if item is not None else None
        if not isinstance(candidate, dict):
            return
        if not self.auto_active or self.auto_qso is None:
            QMessageBox.information(
                self,
                "Lyra",
                "Start Auto, Answer, or Manual first, then select the station to work.",
            )
            return
        self._work_candidate(candidate, forced=True)

    def _repeat_ok(self) -> bool:
        return (
            prefs.repeat_qso() == prefs.REPEAT_AGAIN
            or prefs.cq_pick() == prefs.CQ_MANUAL
        )

    def _is_worked(self, call: str) -> bool:
        return str(call or "").strip().upper() in self._worked

    def _mark_worked(self, call: str) -> None:
        call = str(call or "").strip().upper()
        if call:
            self._worked[call] = time.monotonic()

    def _ignore_repeat(self, decoded: tuple[str, str, str]) -> bool:
        if self.auto_qso is None or self.auto_qso.target or self._repeat_ok():
            return False
        first, second, field = (str(x).strip().upper() for x in decoded)
        call = ""
        if first == "CQ":
            call = second
        elif (
            first == self.auto_qso.my_call
            and len(field) == 4
            and field[:2].isalpha()
            and field[2:].isdigit()
        ):
            call = second
        return bool(call) and self._is_worked(call)

    def _answer_mode_ok(self, mode: str) -> bool:
        if prefs.cq_pick() == prefs.CQ_MANUAL:
            return True
        want = prefs.answer_mode()
        if want == prefs.ANSWER_BOTH:
            return True
        return str(mode or "").upper() == want

    def _reject_answer_mode(self, decoded: tuple[str, str, str], mode: str) -> bool:
        if self.auto_qso is None or self.auto_qso.operation != AutoQso.ANSWER_CQ:
            return False
        if self.auto_qso.target:
            return False
        first, second, _field = (str(x).strip().upper() for x in decoded)
        if first != "CQ" or not second:
            return False
        return not self._answer_mode_ok(mode)

    def _poll_answer(self, freqs: np.ndarray, mag: np.ndarray) -> None:
        self._try_pending_answer()

    def _try_pending_answer(self) -> None:
        pending = self._pending_answer
        if pending is None:
            return
        if (
            not self.auto_active
            or self.auto_qso is None
            or self.tx_session.busy
            or self.auto_qso.target
        ):
            if not self.auto_active or self.auto_qso is None or self.auto_qso.target:
                self._pending_answer = None
            return
        now = time.monotonic()
        tx_at = float(pending.get("tx_at") or pending.get("cq_end") or now)
        if now < tx_at:
            self.tx_status.setText("wait")
            return
        candidate = pending["candidate"]
        self._pending_answer = None
        action = self.auto_qso.begin_answer(candidate["call"], candidate["snr"])
        if action is None:
            return
        self._bind_qso(candidate["call"])
        self.tx_status.setText("TX on")
        self._send_if_current(
            action,
            self.auto_qso,
            self.auto_qso.state,
            self.auto_qso.target,
        )

    def _open_prefs(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("preferences")
        dialog.setModal(True)
        dialog.resize(420, 380)
        lay = QVBoxLayout(dialog)
        form = QFormLayout()
        cq = QComboBox()
        for key, label in prefs.CQ_LABELS:
            cq.addItem(label, key)
        idx = cq.findData(prefs.cq_pick())
        if idx >= 0:
            cq.setCurrentIndex(idx)
        form.addRow("priority", cq)
        repeat = QComboBox()
        for key, label in prefs.REPEAT_LABELS:
            repeat.addItem(label, key)
        idx = repeat.findData(prefs.repeat_qso())
        if idx >= 0:
            repeat.setCurrentIndex(idx)
        form.addRow("already worked", repeat)
        answer = QComboBox()
        for key, label in prefs.ANSWER_LABELS:
            answer.addItem(label, key)
        idx = answer.findData(prefs.answer_mode())
        if idx >= 0:
            answer.setCurrentIndex(idx)
        form.addRow("answer", answer)
        dx = QComboBox()
        for key, label in prefs.DX_LABELS:
            dx.addItem(label, key)
        idx = dx.findData(prefs.dx_show())
        if idx >= 0:
            dx.setCurrentIndex(idx)
        form.addRow("show from grid", dx)
        colors: dict[str, QPushButton] = {}
        for key, label, default in prefs.COLOR_KEYS:
            btn = QPushButton(prefs.color(key) or default)
            btn.setFixedWidth(92)
            self._paint_color_btn(btn)
            btn.clicked.connect(lambda _=False, b=btn: self._pick_color(b))
            form.addRow(label, btn)
            colors[key] = btn
        lay.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        prefs.save_prefs(
            cq=str(cq.currentData()),
            dx=str(dx.currentData()),
            repeat=str(repeat.currentData()),
            answer=str(answer.currentData()),
            colors={key: btn.text().strip() for key, btn in colors.items()},
        )
        self._apply_dx_column()
        self._restyle_tables()

    def _apply_dx_column(self) -> None:
        hidden = prefs.dx_show() == prefs.DX_HIDDEN
        self.band.setColumnHidden(3, hidden)

    def _paint_color_btn(self, btn: QPushButton) -> None:
        bg = btn.text().strip() or "#333333"
        btn.setStyleSheet(
            f"background: {bg}; color: {_contrast_fg(bg)}; border: 1px solid #3a3a3a;"
        )

    def _pick_color(self, btn: QPushButton) -> None:
        chosen = QColorDialog.getColor(QColor(btn.text().strip() or "#333333"), self, "highlight")
        if not chosen.isValid():
            return
        btn.setText(chosen.name())
        self._paint_color_btn(btn)

    def _restyle_tables(self) -> None:
        for table in (self.band, self.qso, self.rx_table):
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    item = table.item(r, c)
                    if item is not None:
                        self._style_activity_item(item, item.data(Qt.ItemDataRole.UserRole))

    def _reset_qso_log(self) -> None:
        self._qso_call = ""
        self.qso.setRowCount(0)
        self.qso_cap.setText("qso")

    def _bind_qso(self, call: str) -> None:
        call = str(call or "").strip().upper()
        if not call:
            return
        mine = self.my_call.text().strip().upper()
        if call != self._qso_call:
            if self.qso.rowCount() == 0:
                self._qso_call = call
                self._backfill_qso(call)
            else:
                self._qso_call = call
        self.qso_cap.setText(f"qso   {mine}  {call}" if mine else f"qso   {call}")

    def _backfill_qso(self, call: str) -> None:
        call = call.strip().upper()
        for r in range(self.band.rowCount()):
            msg_item = self.band.item(r, 4)
            if msg_item is None:
                continue
            msg = msg_item.text().upper()
            if call not in msg:
                continue
            vals = tuple(
                (self.band.item(r, c).text() if self.band.item(r, c) else "")
                for c in (0, 1, 2, 4)
            )
            tone = msg_item.data(Qt.ItemDataRole.UserRole)
            self._add_row(self.qso, vals, keep=80, newest=False, tone=tone)
        while self.qso.rowCount() > 40:
            self.qso.removeRow(0)

    def _log_qso_row(self, vals: tuple[str, ...], decoded, origin: str, tone: str | None) -> None:
        if not self._is_qso_traffic(decoded, origin, vals[-1] if vals else ""):
            return
        qso_vals = (vals[0], vals[1], vals[2], vals[-1])
        self._add_row(self.qso, qso_vals, keep=80, newest=True, tone=tone)
        if self._qso_call:
            mine = self.my_call.text().strip().upper()
            self.qso_cap.setText(f"qso   {mine}  {self._qso_call}")

    def _is_qso_traffic(self, decoded, origin: str, msg: str) -> bool:
        mine = self.my_call.text().strip().upper()
        target = ""
        if self.auto_qso is not None and self.auto_qso.target:
            target = self.auto_qso.target
        elif self._qso_call:
            target = self._qso_call
        if origin == "tx" and (self.auto_active or target):
            return True
        parts: list[str] = []
        if isinstance(decoded, tuple) and len(decoded) == 3:
            parts = [str(x).strip().upper() for x in decoded]
        else:
            parts = msg.upper().split()
        if not parts:
            return False
        if parts[0] == "CQ" and target and len(parts) > 1 and parts[1] == target:
            return True
        if target and mine:
            return mine in parts and target in parts
        return False

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            apply_app_theme(app)
        if getattr(self, "plot", None) is not None:
            self.plot.setBackground("#000000")
            self.curve.setPen(pg.mkPen("#ffffff", width=1))
        if getattr(self, "wf_plot", None) is not None:
            self.wf_plot.setBackground("#000000")

    def _send_if_current(
        self,
        action: AutoAction,
        controller: AutoQso,
        expected_state: str,
        expected_target: str,
    ) -> None:
        if (
            self.auto_active
            and self.auto_qso is controller
            and controller.state == expected_state
            and controller.target == expected_target
        ):
            self._send_action(action)

    def _finish_manual_qso(self, target: str) -> None:
        self._mark_worked(target)
        self.auto_active = False
        self.auto_qso = None
        self._auto_generation += 1
        self.tx_button.setEnabled(True)
        self.tx_stop.setEnabled(False)
        for control in (
            self.my_call,
            self.my_grid,
            self.tx_mode,
            self.tx_channel,
            self.tx_operation,
        ):
            control.setEnabled(True)
        self._sync_tx_channel_enabled()
        self.tx_status.setText("TX off")

    def _export_tx(self) -> None:
        try:
            action = AutoAction(
                pack_cq(self.my_call.text().strip().upper(), self.my_grid.text().strip().upper()),
                "CQ test",
            )
            audio = self._make_tx_audio(action)
        except Exception as exc:
            QMessageBox.warning(self, "Lyra TX", f"Cannot build message:\n{exc}")
            return
        default = (
            Path(__file__).resolve().parents[1]
            / "iq"
            / f"lyra_tx_{self.tx_mode.currentText()}_ch{self.tx_channel.value()}.wav"
        )
        path, _ok = QFileDialog.getSaveFileName(
            self, "Export Lyra transmission", str(default), "WAV (*.wav)"
        )
        if not path:
            return
        write_tx_wav(Path(path), audio)
        self.tx_status.setText(f"Exported {Path(path).name}")

    def _on_device(self, _idx: int) -> None:
        if getattr(self, "_loading_dev", False):
            return
        if self.monitor.isChecked():
            self._start_rx()

    def _startup_notice(self) -> None:
        QMessageBox.information(
            self,
            "Lyra",
            "Hey, this project is constantly being updated, please make sure to check for new updates regularly!",
        )

    def _autostart(self) -> None:
        if self._started or self.worker.running:
            return
        if self.dev.currentData() is None:
            self.monitor.setChecked(False)
            return
        self._started = True
        if self.monitor.isChecked() and not self.worker.running:
            self._start_rx()

    def _on_decode_toggle(self, on: bool) -> None:
        if not self.tx_session.busy:
            self.worker.decode_enabled = bool(on)

    def _set_tx_light(self, on: bool) -> None:
        color = "#3ddc84" if on else "#888888"
        self.tx_light.setStyleSheet(
            f"background:{color}; border-radius:5px; border:none;"
        )
        self._paint_channel_regions(txing=on)

    def _unmute_tx_channel(self) -> None:
        if self.tx_session.busy:
            return
        self.worker.mute_channel = None

    def _loopback(self, row: dict) -> bool:
        if str(row.get("origin", "rx")) == "tx":
            return False
        if self.tx_session.busy:
            return True
        mine = self.my_call.text().strip().upper()
        msg = str(row.get("msg") or "").strip().upper()
        now = time.monotonic()
        self._echo_msgs = [(t, m) for t, m in self._echo_msgs if now - t < 16.0]
        if mine and (msg == f"CQ {mine}" or msg.startswith(f"CQ {mine} ")):
            return True
        if any(msg == m for _t, m in self._echo_msgs):
            return True
        decoded = row.get("decoded")
        if isinstance(decoded, tuple) and len(decoded) == 3:
            first, second, _field = str(decoded[0]).upper(), str(decoded[1]).upper(), decoded[2]
            if mine and first == "CQ" and second == mine:
                return True
            if now < self._echo_until and mine and second == mine:
                return True
        return False

    def _on_monitor(self, on: bool) -> None:
        if on:
            if not self.worker.running:
                self._start_rx()
        elif self.worker.running:
            self.worker.stop()
            self.vu.setValue(0)
            self.dev_lab.setText("Idle")

    def _start_rx(self) -> None:
        data = self.dev.currentData()
        if data is None:
            self.monitor.setChecked(False)
            self.dev_lab.setText("No input device")
            return
        self.worker._last_decode = 0.0
        self._arm_auto()
        try:
            name = self.worker.start(int(data))
        except Exception as e:
            self.monitor.setChecked(False)
            QMessageBox.warning(self, "Lyra", f"Could not open input:\n{e}")
            return
        tap = self.worker.tap
        rate = tap.capture_rate if tap is not None else SAMPLE_RATE
        self.dev_lab.setText(f"{name}   USB {rate} Hz → {SAMPLE_RATE} Hz")

    def _tick_clock(self) -> None:
        utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        self.utc_lab.setText(utc)
        if self.auto_active and self.auto_qso is not None and not self.tx_session.busy:
            state = self.auto_qso.expire()
            if state == "calling":
                self._schedule_cq_repeat(0.25)
            elif state == "listening":
                self.tx_status.setText("TX on")

    def _region_color(self, mid: float, txing: bool | None = None) -> str:
        if txing is None:
            txing = bool(self.tx_session.busy)
        sel = int(self.tx_channel.value()) if hasattr(self, "tx_channel") else 0
        idx = min(
            range(len(CHANNELS)),
            key=lambda j: abs(mid - 0.5 * (CHANNELS[j][0] + CHANNELS[j][1])),
        )
        if (idx + 1) == sel and txing:
            return "#3ddc84"
        if (idx + 1) == sel:
            return "#ffffff"
        return "#777777"

    def _make_region(self, plot, fa: float, fb: float, color: str):
        fill = QColor(color)
        fill.setAlpha(55)
        hover = QColor(color)
        hover.setAlpha(95)
        none = pg.mkPen(None)
        reg = pg.LinearRegionItem(
            values=(fa, fb),
            orientation="vertical",
            brush=fill,
            pen=none,
            hoverBrush=hover,
            hoverPen=none,
            movable=False,
        )
        reg.setBounds((VIEW_LO, VIEW_HI))
        reg.setZValue(30 if color in ("#ffffff", "#3ddc84") else 20)
        plot.addItem(reg)
        return reg

    def _paint_channel_regions(self, *_args, txing: bool | None = None) -> None:
        if getattr(self, "_painting_regions", False):
            return
        if not self._mids or not hasattr(self, "plot"):
            return
        self._painting_regions = True
        try:
            self._rebuild_channels(len(self._mids), txing=txing)
        finally:
            self._painting_regions = False

    def _clear_regions(self) -> None:
        for reg in self._ch_spec + self._ch_wf:
            for plot in (self.plot, self.wf_plot):
                try:
                    plot.removeItem(reg)
                except Exception:
                    pass
        self._ch_spec.clear()
        self._ch_wf.clear()

    def _plan_pairs(self) -> list[tuple[float, float]]:
        half = 0.5 * SPACING_HZ
        return [(m - half, m + half) for m in self._mids]

    def _arm_auto(self) -> None:
        from lyra.rx import set_manual_plan, set_max_channels

        set_manual_plan(None)
        set_max_channels(int(self.n_ch.value()))

    def _follow_plan(self, plan: list[tuple[float, float]]) -> None:
        key = tuple((int(round(a / 8.0) * 8), int(round(b / 8.0) * 8)) for a, b in plan)
        if key == getattr(self, "_shown_plan", None) and len(self._ch_wf) == len(plan):
            return
        self._shown_plan = key
        self._mids = [0.5 * (a + b) for a, b in plan]
        self._rebuild_channels(len(plan))

    def _rebuild_channels(self, n: int, txing: bool | None = None) -> None:
        n = max(0, min(len(CHANNELS), int(n)))
        self._syncing = True
        self._clear_regions()
        if n < 1:
            self._mids = []
            self._syncing = False
            return
        defaults = [0.5 * (a + b) for a, b in CHANNELS]
        mids = list(self._mids)
        if len(mids) > n:
            mids = mids[:n]
        while len(mids) < n:
            mids.append(defaults[len(mids)])
        self._mids = mids
        half = 0.5 * SPACING_HZ
        for mid in mids:
            fa, fb = mid - half, mid + half
            col = self._region_color(mid, txing)
            self._ch_spec.append(self._make_region(self.plot, fa, fb, col))
            self._ch_wf.append(self._make_region(self.wf_plot, fa, fb, col))
        self._syncing = False

    def _on_n_channels(self, n: int) -> None:
        self._arm_auto()
        self._overview_plan = None
        self._sync_channel_overview(self._channel_grid())

    def _tick(self) -> None:
        self._drain_q()
        now = time.monotonic()
        if now - self._last_render < UI_RENDER_INTERVAL_S:
            return
        self._last_render = now
        tap = self.worker.tap
        if tap is None:
            return
        sl = tap.latest(WF_NFFT)
        if sl is None:
            return
        rms = float(np.sqrt(np.mean(sl**2)))
        self.vu.setValue(int(min(100, rms * 800)))
        freqs, mag = usb_spectrum(sl, nfft=WF_NFFT)
        band = (freqs >= VIEW_LO) & (freqs <= VIEW_HI)
        freqs = freqs[band]
        mag_i = np.asarray(mag[band], dtype=np.float32)
        self.curve.setData(freqs, mag_i)
        self._poll_answer(freqs, mag_i)
        from lyra.rx import LAST_PLAN

        plan = list(LAST_PLAN) if LAST_PLAN else self._plan_pairs()
        if LAST_PLAN:
            self._follow_plan(list(LAST_PLAN))
        self._sync_channel_overview(self._channel_grid())
        self._refresh_channel_busy(freqs, mag_i)
        if plan:
            fa0, fb0 = plan[0]
            self.rx_db.setText(f"{_snr_db(sl, fa0, fb0):+.0f} dB")
        else:
            self.rx_db.setText("dB  —")
        if now - self._last_wf < WF_INTERVAL_S and self._wf is not None:
            return
        self._last_wf = now
        if self._wf is None or self._wf.shape[1] != len(mag_i):
            self._wf = np.repeat(mag_i[np.newaxis, :], WF_ROWS, axis=0)
        else:
            self._wf = np.roll(self._wf, -1, axis=0)
            self._wf[-1] = mag_i
        self.img.setImage(self._wf, autoLevels=False, levels=WF_LEVELS)
        self.img.setRect(
            pg.QtCore.QRectF(VIEW_LO, 0.0, VIEW_HI - VIEW_LO, float(WF_ROWS))
        )

    def _drain_q(self) -> None:
        n = 0
        while n < 64:
            try:
                kind, payload = self.worker.out_q.get_nowait()
            except Empty:
                break
            n += 1
            if kind == "row":
                self._on_decode(payload)

    def _on_decode(self, row: dict) -> None:
        origin = str(row.get("origin", "rx"))
        if origin != "tx" and self._loopback(row):
            return
        if origin != "tx" and self.worker.mute_channel is not None:
            ch = self._channel_index(float(row.get("fa") or 0.0), float(row.get("fb") or 0.0))
            if ch == self.worker.mute_channel:
                return
        if origin != "tx" and not self.decode_on.isChecked():
            return
        if origin != "tx":
            msg_u = str(row.get("msg") or "").strip().upper()
            if not msg_u.startswith("CQ "):
                rx_key = (
                    msg_u,
                    round(float(row.get("fa") or 0.0)),
                    round(float(row.get("fb") or 0.0)),
                )
                now_rx = time.monotonic()
                prev_rx = self._recent_rx.get(rx_key)
                if prev_rx is not None and now_rx - prev_rx < 6.5:
                    return
                self._recent_rx[rx_key] = now_rx
        decoded = row.get("decoded")
        if origin != "tx" and isinstance(decoded, tuple) and len(decoded) == 3:
            self._auto_hear(
                decoded,
                int(row.get("db", -8)),
                float(row.get("fa") or 0.0),
                float(row.get("fb") or 0.0),
                str(row.get("mode") or ""),
                float(row["cq_end"]) if row.get("cq_end") is not None else None,
                float(row["cq_start"]) if row.get("cq_start") is not None else None,
            )
            self._watch_other_traffic(
                decoded,
                float(row.get("fa") or 0.0),
                float(row.get("fb") or 0.0),
            )
        tone = "tx" if origin == "tx" else self._incoming_activity_tone(decoded)
        self._decode_count += 1
        self.band_cap.setText(f"activity   {self._decode_count}")
        hz = ""
        if row.get("fa") and row.get("fb"):
            hz = f"{row['fa']:.0f}+{row['fb']:.0f}"
        elif row.get("fa"):
            hz = f"{row['fa']:.0f}"
        snr_text = str(row.get("snr_text") or f"{int(row.get('db', 0)):+d}")
        dx = ""
        candidate = None
        if isinstance(decoded, tuple) and len(decoded) == 3:
            first, second, field = (str(x).strip().upper() for x in decoded)
            grid = ""
            if first == "CQ" or (
                len(field) == 4 and field[:2].isalpha() and field[2:].isdigit()
            ):
                grid = field
            dx = geo.dx_text(grid, prefs.dx_show()) if grid else ""
            snr_i = int(row.get("db", -8))
            fa = float(row.get("fa") or 0.0)
            fb = float(row.get("fb") or 0.0)
            heard_mode = str(row.get("mode") or "").upper()
            if first == "CQ" and second:
                candidate = {
                    "kind": "cq",
                    "call": second,
                    "grid": grid,
                    "snr": snr_i,
                    "fa": fa,
                    "fb": fb,
                    "mode": heard_mode,
                    "t": time.monotonic(),
                    "cq_start": float(row["cq_start"]) if row.get("cq_start") is not None else 0.0,
                    "cq_end": float(row["cq_end"]) if row.get("cq_end") is not None else time.monotonic(),
                }
            elif (
                second
                and first == self.my_call.text().strip().upper()
                and len(field) == 4
                and field[:2].isalpha()
                and field[2:].isdigit()
            ):
                candidate = {
                    "kind": "reply",
                    "call": second,
                    "grid": grid,
                    "snr": snr_i,
                    "fa": fa,
                    "fb": fb,
                    "mode": heard_mode,
                    "t": time.monotonic(),
                }
        vals = (
            str(row.get("utc", "")),
            snr_text,
            hz,
            dx,
            str(row.get("msg", "")),
        )
        if not self.pause_feed.isChecked():
            self._add_row(
                self.band, vals, keep=250, newest=True, tone=tone, extra=candidate
            )
        self._log_qso_row(vals, decoded, origin, tone)

        from lyra.rx import LAST_PLAN

        plan = list(LAST_PLAN)
        if row.get("fa") and row.get("fb"):
            row_pair = (float(row["fa"]), float(row["fb"]))
            mid = 0.5 * sum(row_pair)
            idx = min(
                range(len(CHANNELS)),
                key=lambda i: abs(mid - 0.5 * (CHANNELS[i][0] + CHANNELS[i][1])),
            )
            if not any(abs(mid - 0.5 * (a + b)) < 0.5 * SPACING_HZ for a, b in plan):
                plan.append(row_pair)
                plan.sort(key=lambda pair: 0.5 * sum(pair))
            self._channel_counts[idx] = self._channel_counts.get(idx, 0) + 1
            if origin != "tx":
                self._channel_heard_at[idx] = time.monotonic()
            self._channel_tones[idx] = tone
            heard = str(row.get("mode") or "").upper()
            if heard in ("F", "L"):
                self._channel_mode[idx] = heard
            self._sync_channel_overview(self._channel_grid())
            r = self._channel_rows.get(idx)
            if r is not None:
                self.rx_table.item(r, 1).setText(self._channel_mode.get(idx, ""))
                self.rx_table.item(r, 3).setText(str(row.get("utc", "")))
                self.rx_table.item(r, 4).setText(snr_text)
                self.rx_table.item(r, 5).setText(str(self._channel_counts[idx]))
                self._set_channel_busy_cell(r, idx)
                for c in range(6):
                    self._style_activity_item(self.rx_table.item(r, c), tone)

    def _incoming_activity_tone(self, decoded) -> str | None:
        if not isinstance(decoded, tuple) or len(decoded) != 3:
            return "rx"
        first, second, _field = (str(x).upper() for x in decoded)
        mine = self.my_call.text().strip().upper()
        target = ""
        if self.auto_qso is not None:
            target = self.auto_qso.target
        if not target:
            target = self._qso_call
        if first == "CQ" and target and second == target:
            return "qso"
        if target and mine and mine in (first, second) and target in (first, second):
            return "qso"
        return "rx"

    @staticmethod
    def _style_activity_item(item: QTableWidgetItem | None, tone: str | None) -> None:
        if item is None:
            return
        item.setData(Qt.ItemDataRole.UserRole, tone)
        key = {"tx": "color_tx", "qso": "color_qso", "reply": "color_qso", "rx": "color_rx"}.get(
            str(tone or "")
        )
        if not key:
            return
        bg = prefs.color(key)
        item.setBackground(QColor(bg))
        item.setForeground(QColor(_contrast_fg(bg)))

    def _channel_grid(self) -> list[tuple[float, float]]:
        n = max(1, min(len(CHANNELS), int(self.n_ch.value())))
        return list(CHANNELS[:n])

    def _channel_marked_busy(self, idx: int, freqs=None, mag=None) -> bool:
        if time.monotonic() - self._channel_heard_at.get(idx, 0.0) < BUSY_HOLD_S:
            return True
        if freqs is None or mag is None or idx < 0 or idx >= len(CHANNELS):
            return False
        fa, fb = CHANNELS[idx]
        return self._channel_energy_busy(freqs, mag, fa, fb)

    def _set_channel_busy_cell(self, row: int, idx: int, freqs=None, mag=None) -> None:
        busy = self._channel_marked_busy(idx, freqs, mag)
        item = self.rx_table.item(row, 6)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rx_table.setItem(row, 6, item)
        item.setText("busy" if busy else "ok")
        if busy:
            item.setBackground(QColor("#6a3a3a"))
            item.setForeground(QColor("#f0d0d0"))
        else:
            item.setBackground(QColor("#2f4a38"))
            item.setForeground(QColor("#d8f0dc"))

    def _refresh_channel_busy(self, freqs, mag) -> None:
        for idx, r in self._channel_rows.items():
            self._set_channel_busy_cell(r, idx, freqs, mag)

    def _sync_channel_overview(self, plan: list[tuple[float, float]]) -> None:
        signature = tuple((round(a), round(b)) for a, b in plan)
        if signature == getattr(self, "_overview_plan", None):
            return
        self._overview_plan = signature
        old: dict[int, tuple[str, str]] = {}
        for idx, r in self._channel_rows.items():
            if r < self.rx_table.rowCount():
                old[idx] = (
                    self.rx_table.item(r, 3).text() if self.rx_table.item(r, 3) else "—",
                    self.rx_table.item(r, 4).text() if self.rx_table.item(r, 4) else "—",
                )
        self.rx_table.setRowCount(0)
        self._channel_rows.clear()
        for fa, fb in plan:
            mid = 0.5 * (fa + fb)
            idx = min(
                range(len(CHANNELS)),
                key=lambda i: abs(mid - 0.5 * (CHANNELS[i][0] + CHANNELS[i][1])),
            )
            r = self.rx_table.rowCount()
            self.rx_table.insertRow(r)
            last, snr = old.get(idx, ("—", "—"))
            vals = (
                str(idx + 1),
                self._channel_mode.get(idx, ""),
                f"{fa:.0f}+{fb:.0f}",
                last,
                snr,
                str(self._channel_counts.get(idx, 0)),
                "",
            )
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    tint = QColor(CH_COLORS[idx % len(CH_COLORS)])
                    tint.setAlpha(55)
                    item.setBackground(tint)
                if c < 6:
                    self._style_activity_item(item, self._channel_tones.get(idx))
                self.rx_table.setItem(r, c, item)
            self._channel_rows[idx] = r
            self._set_channel_busy_cell(r, idx)
        self.rx_cap.setText(f"channels   {len(plan)}")

    def _clear_activity(self) -> None:
        self.band.setRowCount(0)
        self._decode_count = 0
        self._channel_counts.clear()
        self._channel_tones.clear()
        self._channel_mode.clear()
        self._channel_heard_at.clear()
        self.worker._seen_rows.clear()
        self._overview_plan = None
        self.band_cap.setText("activity   0")
        self._reset_qso_log()
        self._sync_channel_overview(self._channel_grid())

    def _add_row(
        self,
        table: QTableWidget,
        vals: tuple[str, ...],
        keep: int,
        *,
        newest: bool = False,
        tone: str | None = None,
        extra=None,
    ) -> None:
        r = 0 if newest else table.rowCount()
        table.insertRow(r)
        for c, text in enumerate(vals):
            item = QTableWidgetItem(text)
            if c < 4:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if extra is not None and c == 0:
                item.setData(CAND_ROLE, extra)
            self._style_activity_item(item, tone)
            table.setItem(r, c, item)
        while table.rowCount() > keep:
            table.removeRow(table.rowCount() - 1 if newest else 0)

    def _save_station(self) -> None:
        prefs.save_station(self.my_call.text(), self.my_grid.text())

    def closeEvent(self, event) -> None:
        self._save_station()
        s = prefs.settings()
        s.setValue("win_geo", self.saveGeometry())
        s.setValue("win_docks", self.saveState())
        s.sync()
        self.tx_session.stop()
        try:
            self.rig.disconnect()
        except Exception:
            pass
        try:
            self._hamlib.stop()
        except Exception:
            pass
        self.worker.stop()
        super().closeEvent(event)


def _checkbox_x_url(color: str = "#ffffff") -> str:
    size = 13
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    pen = QPen(QColor(color))
    pen.setWidth(2)
    painter.setPen(pen)
    painter.drawLine(2, 2, size - 3, size - 3)
    painter.drawLine(size - 3, 2, 2, size - 3)
    painter.end()
    path = Path(tempfile.gettempdir()) / f"lyra_checkbox_x_{color.replace('#', '')}.png"
    img.save(str(path))
    return path.resolve().as_posix()


_THEME = {
    "dark": {
        "bg": "#000000",
        "fg": "#d0d0d0",
        "muted": "#808080",
        "border": "#3a3a3a",
        "input": "#050505",
        "hover": "#1a1a1a",
        "alt": "#050505",
        "disabled": "#555555",
        "strong": "#ffffff",
        "groove": "#242424",
        "sel": "#3a3a3a",
        "seltext": "#ffffff",
        "x": "#ffffff",
    },
}

_APP_QSS = """
QMainWindow, QWidget#root {
    background: @bg;
    color: @fg;
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 12px;
}
QDockWidget {
    color: @fg;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background: @input;
    color: @fg;
    padding: 4px 8px;
    border: 1px solid @border;
}
QMenuBar {
    background: @bg;
    color: @fg;
    border-bottom: 1px solid @border;
}
QMenuBar::item:selected, QMenu::item:selected { background: @sel; color: @seltext; }
QMenu {
    background: @bg;
    color: @fg;
    border: 1px solid @border;
}
QComboBox, QPushButton, QSpinBox, QLineEdit {
    background: @input;
    color: @fg;
    border: 1px solid @border;
    border-radius: 2px;
    padding: 3px 8px;
    min-height: 20px;
    selection-background-color: @sel;
}
QPushButton:hover, QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    background: @hover;
    border-color: @strong;
}
QPushButton:pressed { background: @sel; }
QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
    background: @bg;
    color: @disabled;
    border-color: @groove;
}
QComboBox QAbstractItemView {
    background: @input;
    color: @fg;
    selection-background-color: @sel;
    border: 1px solid @border;
}
QCheckBox, QRadioButton { spacing: 6px; color: @fg; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 13px;
    height: 13px;
    background: @bg;
    border: 1px solid @border;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: @bg;
    border-color: @strong;
    image: url("__CHECK_X__");
}
QCheckBox::indicator:checked:hover, QRadioButton::indicator:checked:hover {
    border-color: @strong;
    image: url("__CHECK_X__");
}
QGroupBox {
    color: @strong;
    border: 1px solid @border;
    margin-top: 8px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    background: @bg;
}
QLabel { color: @fg; }
QHeaderView::section {
    background: @input;
    color: @muted;
    border: 0;
    border-right: 1px solid @groove;
    border-bottom: 1px solid @border;
    padding: 3px 6px;
}
QTableWidget {
    background: @bg;
    alternate-background-color: @alt;
    color: @fg;
    gridline-color: @groove;
    selection-background-color: @sel;
    selection-color: @seltext;
    border: 1px solid @border;
}
QProgressBar {
    background: @bg;
    border: 1px solid @border;
    text-align: center;
}
QProgressBar::chunk { background: @strong; }
QStatusBar {
    background: @bg;
    color: @muted;
    border-top: 1px solid @border;
}
QSplitter::handle { background: @border; }
QToolTip {
    background: @bg;
    color: @fg;
    border: 1px solid @strong;
}
QSlider::groove:horizontal {
    height: 4px;
    background: @groove;
}
QSlider::sub-page:horizontal {
    background: @strong;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    background: @strong;
    border: 1px solid @strong;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: @bg;
    width: 11px;
    height: 11px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: @border;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QDialog {
    background: @bg;
    color: @fg;
}
"""


def app_stylesheet(theme: str | None = None) -> str:
    palette = _THEME["dark"]
    qss = _APP_QSS
    for key, value in sorted(palette.items(), key=lambda kv: -len(kv[0])):
        qss = qss.replace("@" + key, value)
    return qss.replace("__CHECK_X__", _checkbox_x_url(palette["x"]))


def apply_app_theme(app: QApplication, theme: str | None = None) -> None:
    p = _THEME["dark"]
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p["bg"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p["fg"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(p["input"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p["alt"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(p["fg"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(p["input"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p["fg"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p["strong"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p["bg"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(p["muted"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p["bg"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p["fg"]))
    app.setPalette(pal)
    app.setStyleSheet(app_stylesheet())


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_app_theme(app)
    win = LyraWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
