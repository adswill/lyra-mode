
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from lyra import gui as lyra_gui
from lyra.gui import (
    CHANNELS,
    LIVE_DECODE_INTERVAL_S,
    SAMPLE_RATE,
    SPACING_HZ,
    UI_RENDER_INTERVAL_S,
    VIEW_HI,
    VIEW_LO,
    WF_LEVELS,
    WF_NFFT,
    WF_ROWS,
    DecodeWorker as _DecodeWorker,
    LyraWindow as _LyraWindow,
    _snr_db,
)
from lyra import long_demo as lyra_codec
from lyra.long_demo import build_tx_audio, decode_usb_all, read_wav
from lyra.modem import usb_spectrum
from lyra.qso_auto import AutoAction

from PySide6.QtGui import QAction, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
)
import pyqtgraph as pg


class DecodeWorker(_DecodeWorker):
    def stop(self) -> None:
        self.running = False
        if self.tap is not None:
            self.tap.stop()
            self.tap = None

    def _decode_wav(self, path: Path) -> None:
        try:
            self.out_q.put(("status", f"Decoding {path.name}…"))
            audio = read_wav(path)
            rows = decode_usb_all(audio)
            self.out_q.put(("status", lyra_codec.LAST_STATUS or "no CRC"))
            if not rows:
                self.out_q.put(("status", lyra_codec.LAST_STATUS or "no decode"))
                return
            self.last = None
            view = np.real(np.asarray(audio))
            self._push_rows(view, rows)
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
                    rows = decode_usb_all(audio)
                    self.out_q.put(("status", lyra_codec.LAST_STATUS or ""))
                except Exception as e:
                    self.out_q.put(("status", f"decode error: {type(e).__name__}"))
                    rows = []
                self._last_decode = time.monotonic()
                self._push_rows(audio, rows)
            time.sleep(0.02)


class LyraWindow(_LyraWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("lyra long v2 demo")
        old = self.worker
        old.stop()
        self.worker = DecodeWorker()
        self.tx_mode.blockSignals(True)
        self.tx_mode.clear()
        self.tx_mode.addItems(["L"])
        self.tx_mode.blockSignals(False)
        self.tx_channel.setRange(6, 10)
        self.tx_channel.setValue(6)

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
        act_about = QAction("&About lyra long v2 demo", self)
        act_about.triggered.connect(self._about)
        help_m.addAction(act_about)

    def _open_wav(self) -> None:
        path, _ok = QFileDialog.getOpenFileName(
            self,
            "decode lyra long v2 demo wav",
            str(Path(__file__).resolve().parents[1] / "example qso wavs"),
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
            "about lyra long v2 demo",
            "lyra long v2 demo decoder. x-chirp, 16-bit uw, rate-1/4 fec.\n"
            "this window does not use the production f/l decoder.\n"
            "open the example qso wav or monitor usb audio on channels 6–10.",
        )

    def _on_tx_mode(self, _index: int = 0) -> None:
        if self.tx_channel.value() < 6:
            self.tx_channel.setValue(6)

    def _make_tx_audio(self, action: AutoAction) -> np.ndarray:
        return build_tx_audio(
            action.bits,
            mode="L",
            channel=self.tx_channel.value(),
            level=self.tx_level.value() / 100.0,
        )

    def _arm_auto(self) -> None:
        return

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
        plan = list(lyra_codec.LAST_PLAN) if lyra_codec.LAST_PLAN else self._plan_pairs()
        if lyra_codec.LAST_PLAN:
            self._follow_plan(list(lyra_codec.LAST_PLAN))
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
        plan = list(lyra_codec.LAST_PLAN)
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

    def _clear_activity(self) -> None:
        self.band.setRowCount(0)
        self._decode_count = 0
        self._channel_counts.clear()
        self._channel_tones.clear()
        self.worker._seen_rows.clear()
        self._overview_plan = None
        self.band_cap.setText("activity   0")
        self._sync_channel_overview(list(lyra_codec.LAST_PLAN))


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
    app.setStyleSheet(lyra_gui._APP_QSS)
    win = LyraWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
