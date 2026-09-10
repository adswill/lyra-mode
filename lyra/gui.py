
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, SimpleQueue

import numpy as np

from lyra.capture import AudioTap, default_input_index, list_inputs, load_sounddevice
from lyra.codec import decode_usb_all, heartbeat_bits
from lyra import codec as lyra_codec
from lyra.qso_auto import AutoAction, AutoQso, contention_plan
from lyra.rig import DummyRig, RigctldRig
from lyra.tx import TxSession, build_tx_audio, list_outputs, write_tx_wav
from lyra.const import (
    BIT_RATE,
    CHANNEL_MODES,
    CHANNELS,
    RF_DIAL_HZ,
    SAMPLE_RATE,
    SPACING_HZ,
    TONE_A_HZ,
    TONE_B_HZ,
)
from lyra.pack import pack_cq

VIEW_LO = 400.0
VIEW_HI = 5_500.0
WF_NFFT = 8192
WF_ROWS = 360
WF_LEVELS = (-105.0, -55.0)
LIVE_DECODE_INTERVAL_S = 0.15
UI_RENDER_INTERVAL_S = 0.10
CQ_LISTEN_MS = {"F": 4500, "L": 6500}
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
from PySide6.QtGui import QAction, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
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
        self.out_q: SimpleQueue = SimpleQueue()
        self._last_decode = 0.0

    def start(self, device: int) -> str:
        self.stop()
        self._seen_rows.clear()
        self.tap = AudioTap(device, seconds=12.0)
        self.tap.start()
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
        for rec in rows:
            got = rec["row"]
            fa = float(rec.get("fa") or 0.0)
            fb = float(rec.get("fb") or 0.0)
            now = time.monotonic()
            key = (got, round(fa), round(fb))
            if now - self._seen_rows.get(key, -1e9) < 8.0:
                continue
            self._seen_rows[key] = now
            if len(self._seen_rows) > 512:
                self._seen_rows = {
                    k: t for k, t in self._seen_rows.items() if now - t < 30.0
                }
            self.last = key
            self.last_t = now
            kind, a, b = got
            msg = " ".join(p for p in (kind, a, b) if p)
            self.out_q.put(
                (
                    "row",
                    {
                        "utc": datetime.now(timezone.utc).strftime("%H%M%S"),
                        "db": round(_snr_db(audio, fa or None, fb or None), 0),
                        "fa": fa,
                        "fb": fb,
                        "msg": msg,
                        "decoded": got,
                    },
                )
            )

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
        while self.running and self.tap is not None:
            now = time.monotonic()
            if not self.decode_enabled or (now - self._last_decode) < LIVE_DECODE_INTERVAL_S:
                time.sleep(0.02)
                continue
            
            
            audio = self.tap.latest(6 * SAMPLE_RATE)
            if audio is not None and len(audio) >= int(0.80 * SAMPLE_RATE):
                rows: list = []
                try:
                    rows = decode_usb_all(audio, self.block)
                    self.out_q.put(("status", lyra_codec.LAST_STATUS or ""))
                except Exception as e:
                    self.out_q.put(("status", f"decode error: {type(e).__name__}"))
                    rows = []
                self._last_decode = time.monotonic()
                self._push_rows(audio, rows)
            time.sleep(0.02)


class TxBridge(QObject):
    status = Signal(str, bool)


class LyraWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("lyra")
        self.resize(1400, 860)
        self.worker = DecodeWorker()
        self.tx_session = TxSession()
        self.tx_bridge = TxBridge()
        self.tx_bridge.status.connect(self._on_tx_status)
        self.rig = DummyRig()
        self._resume_monitor = False
        self._wf = None  
        self._started = False
        self._last_lock = ""
        self._syncing = False
        self._ch_spec: list = []
        self._ch_wf: list = []
        self._mids: list[float] = [0.5 * (a + b) for a, b in CHANNELS[:3]]
        self._decode_count = 0
        self._channel_counts: dict[int, int] = {}
        self._channel_rows: dict[int, int] = {}
        self._channel_tones: dict[int, str | None] = {}
        self._pending_tx_log: tuple[AutoAction, int] | None = None
        self._last_render = 0.0
        self._build_menu()
        self._build()
        self._load_devices()
        self._load_outputs()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick_clock)
        self.clock.start(250)
        QTimer.singleShot(400, self._autostart)

    def _build_menu(self) -> None:
        file_m = self.menuBar().addMenu("&File")
        act_open = QAction("Open &WAV…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_wav)
        file_m.addAction(act_open)
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
            f"Lyra — chirp + dual-rail GMSK on {RF_DIAL_HZ / 1e6:.4f} USB\n"
            "Lyra F (fast): slash high→low (right→left), unique bits per rail, ~2.3 s\n"
            "Lyra L (long): slash low→high (left→right), same bits both rails, ~4.3 s\n"
            f"{BIT_RATE:g} baud per rail, r=1/2 K=7 + CRC-16.\n\n"
            "Channels is a maximum. Lyra places 80 Hz bands on the 500 Hz USB grid automatically.",
        )

    def _build(self) -> None:
        pg.setConfigOptions(antialias=False, background="#000000", foreground="#777777")
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(8, 6, 8, 4)
        lay.setSpacing(6)

        top = QHBoxLayout()
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
        dial = QLabel(f"{RF_DIAL_HZ / 1e3:.1f} kHz   USB")
        top.addWidget(dial)
        top.addStretch(1)
        top.addWidget(QLabel("audio"))
        self.dev = QComboBox()
        self.dev.setMinimumWidth(280)
        self.dev.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dev.currentIndexChanged.connect(self._on_device)
        top.addWidget(self.dev, 1)
        ref = QPushButton("refresh")
        ref.setFixedWidth(72)
        ref.clicked.connect(self._load_devices)
        top.addWidget(ref)
        lay.addLayout(top)

        meters = QHBoxLayout()
        meters.setSpacing(8)
        meters.addWidget(QLabel("rx"))
        self.vu = QProgressBar()
        self.vu.setRange(0, 100)
        self.vu.setTextVisible(False)
        self.vu.setFixedHeight(14)
        self.vu.setFixedWidth(140)
        meters.addWidget(self.vu)
        self.rx_db = QLabel("dB  —")
        self.rx_db.setFixedWidth(72)
        meters.addWidget(self.rx_db)
        self.lock_lab = QLabel("no lock")
        meters.addWidget(self.lock_lab, 1)
        lay.addLayout(meters)
        lay.addWidget(self._build_tx_panel())

        lists = QSplitter(Qt.Orientation.Horizontal)
        lists.setChildrenCollapsible(False)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(2)
        traffic_head = QHBoxLayout()
        self.band_cap = QLabel("activity   0")
        traffic_head.addWidget(self.band_cap)
        traffic_head.addStretch(1)
        self.pause_feed = QCheckBox("Pause feed")
        traffic_head.addWidget(self.pause_feed)
        clear = QPushButton("clear")
        clear.setFixedWidth(64)
        clear.clicked.connect(self._clear_activity)
        traffic_head.addWidget(clear)
        left_l.addLayout(traffic_head)
        self.band = QTableWidget(0, 4)
        self.band.setHorizontalHeaderLabels(["time", "snr", "ch", "message"])
        self.band.verticalHeader().setVisible(False)
        self.band.setShowGrid(False)
        self.band.setAlternatingRowColors(True)
        self.band.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.band.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        hh = self.band.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.band.setColumnWidth(0, 72)
        self.band.setColumnWidth(1, 44)
        self.band.setColumnWidth(2, 110)
        self.band.verticalHeader().setDefaultSectionSize(20)
        left_l.addWidget(self.band)
        lists.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(2)
        self.rx_cap = QLabel("channels")
        right_l.addWidget(self.rx_cap)
        self.rx_table = QTableWidget(0, 6)
        self.rx_table.setHorizontalHeaderLabels(["ch", "mode", "hz", "last", "snr", "count"])
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
        rh.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.rx_table.setColumnWidth(0, 42)
        self.rx_table.setColumnWidth(1, 48)
        self.rx_table.setColumnWidth(2, 100)
        self.rx_table.setColumnWidth(3, 70)
        self.rx_table.setColumnWidth(4, 54)
        self.rx_table.verticalHeader().setDefaultSectionSize(20)
        right_l.addWidget(self.rx_table)
        lists.addWidget(right)
        lists.setSizes([760, 320])

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(lists)

        graph = QWidget()
        graph_l = QVBoxLayout(graph)
        graph_l.setContentsMargins(0, 0, 0, 0)
        graph_l.setSpacing(0)
        gcap = QLabel("graph")
        graph_l.addWidget(gcap)
        graph.setMinimumHeight(300)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#000000")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setLabel("left", "dBFS")
        self.plot.setXRange(VIEW_LO, VIEW_HI, padding=0)
        self.plot.setYRange(-100, -10, padding=0)
        self.plot.disableAutoRange()
        self.plot.getViewBox().setLimits(xMin=300, xMax=5500, yMin=-130, yMax=0)
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
        self.wf_plot.setMinimumHeight(220)
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
        split.addWidget(graph)
        split.setSizes([280, 380])
        lay.addWidget(split, 1)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.utc_lab = QLabel("")
        self.dev_lab = QLabel("idle")
        sb.addWidget(self.utc_lab)
        sb.addWidget(self.dev_lab, 1)

    def _build_tx_panel(self) -> QGroupBox:
        box = QGroupBox("tx")
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 6, 8, 7)
        row.setSpacing(7)

        row.addWidget(QLabel("call"))
        self.my_call = QLineEdit("K1ABC")
        self.my_call.setFixedWidth(82)
        row.addWidget(self.my_call)
        row.addWidget(QLabel("grid"))
        self.my_grid = QLineEdit("FN20")
        self.my_grid.setFixedWidth(58)
        row.addWidget(self.my_grid)
        row.addWidget(QLabel("mode"))
        self.tx_mode = QComboBox()
        self.tx_mode.addItems(["F", "L"])
        self.tx_mode.currentIndexChanged.connect(self._on_tx_mode)
        row.addWidget(self.tx_mode)
        row.addWidget(QLabel("channel"))
        self.tx_channel = QSpinBox()
        self.tx_channel.setRange(1, 10)
        self.tx_channel.setValue(3)
        self.tx_channel.setFixedWidth(52)
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
        self.tx_status = QLabel("TX off")
        row.addWidget(self.tx_status, 1)

        
        self.rig_kind = QComboBox()
        self.rig_kind.addItem("Test", "dummy")
        self.rig_kind.addItem("Hamlib rigctld", "rigctld")
        self.rig_kind.currentIndexChanged.connect(self._on_rig_kind)
        self.rig_host = QLineEdit("127.0.0.1")
        self.rig_host.setFixedWidth(105)
        self.rig_port = QSpinBox()
        self.rig_port.setRange(1, 65535)
        self.rig_port.setValue(4532)
        self.rig_port.setFixedWidth(76)
        self.rig_connect = QPushButton("connect")
        self.rig_connect.clicked.connect(self._connect_rig)
        self.rig_status = QLabel("test ready")
        self.rig_status.setMinimumWidth(125)
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
        network = self.rig_kind.currentData() == "rigctld"
        self.rig_host.setEnabled(network)
        self.rig_port.setEnabled(network)
        self.rig_status.setText("not connected" if network else "test ready")

    def _connect_rig(self) -> bool:
        try:
            self.rig.disconnect()
        except Exception:
            pass
        try:
            if self.rig_kind.currentData() == "rigctld":
                self.rig = RigctldRig(self.rig_host.text().strip(), self.rig_port.value())
            else:
                self.rig = DummyRig()
            self.rig.connect()
            self.rig.set_frequency(RF_DIAL_HZ)
            self.rig.set_mode("USB", 6000)
        except Exception as exc:
            self.rig_status.setText("Connection failed")
            QMessageBox.warning(self, "Lyra rig control", str(exc))
            return False
        self.rig_status.setText(f"{self.rig.name} connected")
        return True

    def _on_tx_mode(self, _index: int = 0) -> None:
        mode = self.tx_mode.currentText()
        ch = self.tx_channel.value()
        if mode == "F" and ch > 5:
            self.tx_channel.setValue(3)
        elif mode == "L" and ch < 6:
            self.tx_channel.setValue(6)

    def _on_operation_changed(self, _index: int = 0) -> None:
        manual = self.tx_operation.currentData() == AutoQso.MANUAL_CQ
        self.tx_button.setText("start one qso" if manual else "start")

    def _open_tx_setup(self) -> None:
        if self._setup_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("radio")
            dialog.setModal(False)
            dialog.resize(560, 155)
            lay = QVBoxLayout(dialog)
            rig_row = QHBoxLayout()
            rig_row.addWidget(QLabel("rig"))
            rig_row.addWidget(self.rig_kind)
            rig_row.addWidget(self.rig_host)
            rig_row.addWidget(self.rig_port)
            rig_row.addWidget(self.rig_connect)
            lay.addLayout(rig_row)
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
        if self.rig_kind.currentData() == "rigctld":
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
        else:
            self._send_action(action)

    def _send_action(self, action: AutoAction) -> None:
        if not self.auto_active or self.tx_session.busy:
            return
        try:
            self.rig.set_frequency(RF_DIAL_HZ)
            self.rig.set_mode("USB", 6000)
            audio = self._make_tx_audio(action)
            self._resume_monitor = self.monitor.isChecked()
            if self._resume_monitor:
                self.monitor.setChecked(False)
            self._tx_label = action.label
            self._pending_tx_log = (action, self.tx_channel.value())
            self.tx_status.setText("TX on")
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
        self._auto_generation = getattr(self, "_auto_generation", 0) + 1
        self.tx_session.stop()
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

    def _on_tx_status(self, message: str, done: bool) -> None:
        if not done:
            if self.auto_active:
                self.tx_status.setText("TX on")
            return
        pending = self._pending_tx_log
        self._pending_tx_log = None
        if message == "Transmission complete" and pending is not None:
            action, channel = pending
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
        if self._resume_monitor:
            self._resume_monitor = False
            self.monitor.setChecked(True)
        if not self.auto_active or self.auto_qso is None:
            self.tx_status.setText("TX off")
            return
        self.tx_status.setText("TX on")
        generation = self._auto_generation
        if self.auto_qso.state == "calling":
            delay = CQ_LISTEN_MS[self.tx_mode.currentText()]
            QTimer.singleShot(delay, lambda: self._auto_continue(generation, False))
        elif self.auto_qso.state == "complete":
            QTimer.singleShot(2500, lambda: self._auto_continue(generation, True))

    def _auto_continue(self, generation: int, completed: bool) -> None:
        if (
            not self.auto_active
            or self.auto_qso is None
            or generation != self._auto_generation
            or self.tx_session.busy
        ):
            return
        action = self.auto_qso.resume() if completed else self.auto_qso.repeat_cq()
        if action is not None:
            self._send_action(action)
        else:
            self.tx_status.setText("TX on")

    def _auto_hear(
        self,
        decoded: tuple[str, str, str],
        snr: int,
        fa: float = 0.0,
        fb: float = 0.0,
    ) -> None:
        if not self.auto_active or self.auto_qso is None or self.tx_session.busy:
            return
        delay_ms = 150
        previous_state = self.auto_qso.state
        if (
            self.auto_qso.operation == AutoQso.ANSWER_CQ
            and self.auto_qso.state == "listening"
            and decoded[0] == "CQ"
            and fa
            and fb
        ):
            mid = 0.5 * (fa + fb)
            idx = min(
                range(len(CHANNELS)),
                key=lambda i: abs(mid - 0.5 * (CHANNELS[i][0] + CHANNELS[i][1])),
            )
            mode = CHANNEL_MODES[idx]
            plan = contention_plan(
                decoded[1],
                self.auto_qso.my_call,
                mode,
                idx + 1,
            )
            self.tx_mode.setCurrentText(mode)
            self.tx_channel.setValue(plan.channel)
            delay_ms = plan.delay_ms
        action = self.auto_qso.hear(decoded, snr)
        if action is not None:
            self.tx_status.setText("TX on")
            controller = self.auto_qso
            expected_state = controller.state
            expected_target = controller.target
            QTimer.singleShot(
                delay_ms,
                lambda: self._send_if_current(
                    action, controller, expected_state, expected_target
                ),
            )
        elif self.auto_qso.state == "complete":
            target = self.auto_qso.target
            if self.auto_qso.operation == AutoQso.MANUAL_CQ:
                self._finish_manual_qso(target)
            else:
                generation = self._auto_generation
                self.tx_status.setText("TX on")
                QTimer.singleShot(2500, lambda: self._auto_continue(generation, True))
        elif previous_state != self.auto_qso.state and self.auto_qso.state == "listening":
            self.tx_status.setText("TX on")

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

    def _autostart(self) -> None:
        if self._started or self.worker.running:
            return
        if self.dev.currentData() is None:
            self.lock_lab.setText("No input device")
            self.monitor.setChecked(False)
            return
        self._started = True
        if self.monitor.isChecked() and not self.worker.running:
            self._start_rx()

    def _on_decode_toggle(self, on: bool) -> None:
        self.worker.decode_enabled = bool(on)

    def _on_monitor(self, on: bool) -> None:
        if on:
            if not self.worker.running:
                self._start_rx()
        elif self.worker.running:
            self.worker.stop()
            self.lock_lab.setText("Monitor off")
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
                action = self.auto_qso.repeat_cq()
                if action is not None:
                    self.tx_status.setText("TX on")
                    self._send_action(action)
            elif state == "listening":
                self.tx_status.setText("TX on")

    def _make_region(self, plot, fa: float, fb: float, color: str):
        fill = QColor(color)
        fill.setAlpha(55)
        hover = QColor(color)
        hover.setAlpha(95)
        reg = pg.LinearRegionItem(
            values=(fa, fb),
            orientation="vertical",
            brush=fill,
            pen=pg.mkPen(color, width=2),
            hoverBrush=hover,
            movable=False,
        )
        reg.setBounds((VIEW_LO, VIEW_HI))
        reg.setZValue(20)
        plot.addItem(reg)
        return reg

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

    def _rebuild_channels(self, n: int) -> None:
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
        for i, mid in enumerate(mids):
            fa, fb = mid - half, mid + half
            col = CH_COLORS[i % len(CH_COLORS)]
            self._ch_spec.append(self._make_region(self.plot, fa, fb, col))
            self._ch_wf.append(self._make_region(self.wf_plot, fa, fb, col))
        self._syncing = False

    def _on_n_channels(self, n: int) -> None:
        self._arm_auto()

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
        from lyra.rx import LAST_PLAN

        plan = list(LAST_PLAN) if LAST_PLAN else self._plan_pairs()
        if LAST_PLAN:
            self._follow_plan(list(LAST_PLAN))
        self._sync_channel_overview(plan)
        if plan:
            fa0, fb0 = plan[0]
            self.rx_db.setText(f"{_snr_db(sl, fa0, fb0):+.0f} dB")
            pk = "  ".join(f"{a:.0f}+{b:.0f}" for a, b in plan)
        else:
            self.rx_db.setText("dB  —")
            pk = "no channels"
        extra = ""
        if rms < 1.5e-4:
            extra = "    silence on cable"
        if tap.overruns:
            extra += f"    audio drop {tap.overruns}"
        if self._wf is None or self._wf.shape[1] != len(mag_i):
            self._wf = np.repeat(mag_i[np.newaxis, :], WF_ROWS, axis=0)
        else:
            self._wf = np.roll(self._wf, -1, axis=0)
            self._wf[-1] = mag_i
        self.img.setImage(self._wf, autoLevels=False, levels=WF_LEVELS)
        self.img.setRect(
            pg.QtCore.QRectF(VIEW_LO, 0.0, VIEW_HI - VIEW_LO, float(WF_ROWS))
        )
        self.lock_lab.setText((self._last_lock or pk) + extra)

    def _drain_q(self) -> None:
        n = 0
        while n < 64:
            try:
                kind, payload = self.worker.out_q.get_nowait()
            except Empty:
                break
            n += 1
            if kind == "status":
                self._on_lock(str(payload))
            elif kind == "row":
                self._on_decode(payload)

    def _on_decode(self, row: dict) -> None:
        origin = str(row.get("origin", "rx"))
        if origin != "tx" and not self.decode_on.isChecked():
            return
        decoded = row.get("decoded")
        if origin != "tx" and isinstance(decoded, tuple) and len(decoded) == 3:
            self._auto_hear(
                decoded,
                int(row.get("db", -8)),
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
        vals = (
            str(row.get("utc", "")),
            snr_text,
            hz,
            str(row.get("msg", "")),
        )
        if not self.pause_feed.isChecked():
            self._add_row(self.band, vals, keep=250, newest=True, tone=tone)

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
            self._channel_tones[idx] = tone
            self._sync_channel_overview(plan or list(CHANNELS))
            r = self._channel_rows.get(idx)
            if r is not None:
                self.rx_table.item(r, 3).setText(str(row.get("utc", "")))
                self.rx_table.item(r, 4).setText(snr_text)
                self.rx_table.item(r, 5).setText(str(self._channel_counts[idx]))
                for c in range(self.rx_table.columnCount()):
                    self._style_activity_item(self.rx_table.item(r, c), tone)

    def _incoming_activity_tone(self, decoded) -> str | None:
        if (
            not self.auto_active
            or self.auto_qso is None
            or not isinstance(decoded, tuple)
            or len(decoded) != 3
        ):
            return None
        first, second, _field = (str(x).upper() for x in decoded)
        target = self.auto_qso.target
        if first == "CQ":
            if self.auto_qso.operation == AutoQso.ANSWER_CQ and target == second:
                return "reply"
            return None
        if target and self.auto_qso.my_call in (first, second) and target in (first, second):
            return "reply"
        return None

    @staticmethod
    def _style_activity_item(item: QTableWidgetItem | None, tone: str | None) -> None:
        if item is None:
            return
        item.setData(Qt.ItemDataRole.UserRole, tone)
        if tone == "reply":
            item.setBackground(QColor("#333333"))
            item.setForeground(QColor("#ffffff"))
        elif tone == "tx":
            item.setBackground(QColor("#dddddd"))
            item.setForeground(QColor("#000000"))

    def _sync_channel_overview(self, plan: list[tuple[float, float]]) -> None:
        signature = tuple((round(a), round(b)) for a, b in plan)
        if signature == getattr(self, "_overview_plan", None):
            return
        self._overview_plan = signature
        old: dict[int, tuple[str, str]] = {}
        for idx, r in self._channel_rows.items():
            if r < self.rx_table.rowCount():
                old[idx] = (
                    self.rx_table.item(r, 3).text(),
                    self.rx_table.item(r, 4).text(),
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
                CHANNEL_MODES[idx],
                f"{fa:.0f}+{fb:.0f}",
                last,
                snr,
                str(self._channel_counts.get(idx, 0)),
            )
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    tint = QColor(CH_COLORS[idx % len(CH_COLORS)])
                    tint.setAlpha(55)
                    item.setBackground(tint)
                self._style_activity_item(item, self._channel_tones.get(idx))
                self.rx_table.setItem(r, c, item)
            self._channel_rows[idx] = r
        self.rx_cap.setText(f"channels   {len(plan)}")

    def _clear_activity(self) -> None:
        self.band.setRowCount(0)
        self._decode_count = 0
        self._channel_counts.clear()
        self._channel_tones.clear()
        self.worker._seen_rows.clear()
        self._overview_plan = None
        self.band_cap.setText("activity   0")
        from lyra.rx import LAST_PLAN

        self._sync_channel_overview(list(LAST_PLAN))

    def _add_row(
        self,
        table: QTableWidget,
        vals: tuple[str, ...],
        keep: int,
        *,
        newest: bool = False,
        tone: str | None = None,
    ) -> None:
        r = 0 if newest else table.rowCount()
        table.insertRow(r)
        for c, text in enumerate(vals):
            item = QTableWidgetItem(text)
            if c < 3:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_activity_item(item, tone)
            table.setItem(r, c, item)
        while table.rowCount() > keep:
            table.removeRow(table.rowCount() - 1 if newest else 0)

    def _on_lock(self, text: str) -> None:
        if "IQ IS " in text:
            self._last_lock = text
        elif " CRC" in text:
            self._last_lock = text[: text.index(" CRC") + 4]
        elif "costas" in text:
            self._last_lock = text[text.index("costas") :]
        else:
            self._last_lock = text

    def closeEvent(self, event) -> None:
        self.tx_session.stop()
        try:
            self.rig.disconnect()
        except Exception:
            pass
        self.worker.stop()
        super().closeEvent(event)


_APP_QSS = """
QMainWindow, QWidget#root {
    background: #000000;
    color: #d0d0d0;
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 12px;
}
QMenuBar {
    background: #000000;
    color: #d0d0d0;
    border-bottom: 1px solid #3a3a3a;
}
QMenuBar::item:selected, QMenu::item:selected { background: #3a3a3a; color: #ffffff; }
QMenu {
    background: #000000;
    color: #d0d0d0;
    border: 1px solid #3a3a3a;
}
QComboBox, QPushButton, QSpinBox, QLineEdit {
    background: #050505;
    color: #d0d0d0;
    border: 1px solid #3a3a3a;
    border-radius: 2px;
    padding: 3px 8px;
    min-height: 20px;
    selection-background-color: #666666;
}
QPushButton:hover, QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    background: #1a1a1a;
    border-color: #ffffff;
}
QPushButton:pressed { background: #3a3a3a; }
QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
    background: #000000;
    color: #555555;
    border-color: #242424;
}
QComboBox QAbstractItemView {
    background: #050505;
    color: #d0d0d0;
    selection-background-color: #3a3a3a;
    border: 1px solid #3a3a3a;
}
QCheckBox { spacing: 6px; color: #d0d0d0; }
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    background: #000000;
    border: 1px solid #3a3a3a;
}
QCheckBox::indicator:checked {
    background: #ffffff;
    border-color: #ffffff;
}
QGroupBox {
    color: #ffffff;
    border: 1px solid #3a3a3a;
    margin-top: 8px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    background: #000000;
}
QLabel { color: #d0d0d0; }
QHeaderView::section {
    background: #050505;
    color: #808080;
    border: 0;
    border-right: 1px solid #242424;
    border-bottom: 1px solid #3a3a3a;
    padding: 3px 6px;
}
QTableWidget {
    background: #000000;
    alternate-background-color: #050505;
    color: #d0d0d0;
    gridline-color: #242424;
    selection-background-color: #3a3a3a;
    selection-color: #ffffff;
    border: 1px solid #3a3a3a;
}
QProgressBar {
    background: #000000;
    border: 1px solid #3a3a3a;
    text-align: center;
}
QProgressBar::chunk { background: #ffffff; }
QStatusBar {
    background: #000000;
    color: #808080;
    border-top: 1px solid #3a3a3a;
}
QSplitter::handle { background: #3a3a3a; }
QToolTip {
    background: #000000;
    color: #d0d0d0;
    border: 1px solid #ffffff;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #242424;
}
QSlider::sub-page:horizontal {
    background: #ffffff;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    background: #ffffff;
    border: 1px solid #ffffff;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #000000;
    width: 11px;
    height: 11px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #3a3a3a;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#d0d0d0"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#050505"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#d0d0d0"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#050505"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#d0d0d0"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#666666"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#d0d0d0"))
    app.setPalette(pal)
    app.setStyleSheet(_APP_QSS)
    win = LyraWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
