
from __future__ import annotations

from dataclasses import dataclass
import time
import zlib

from lyra.pack import pack_73, pack_cq, pack_reply, pack_rpt, pack_rr73


@dataclass
class AutoAction:
    bits: list[int]
    label: str


@dataclass(frozen=True)
class ReplyPlan:
    channel: int
    delay_ms: int


def contention_plan(
    cq_call: str,
    responder_call: str,
    mode: str,
    cq_channel: int,
    *,
    epoch_s: float | None = None,
) -> ReplyPlan:
    first = 1 if mode.upper() == "F" else 6
    allowed = list(range(first, first + 5))
    if int(cq_channel) in allowed:
        allowed.remove(int(cq_channel))
    bucket = int((time.time() if epoch_s is None else epoch_s) // 15)
    key = f"{cq_call.upper()}|{responder_call.upper()}|{bucket}".encode("ascii", "ignore")
    value = zlib.crc32(key)
    return ReplyPlan(
        channel=allowed[value % len(allowed)],
        delay_ms=100 + ((value >> 8) % 501),
    )


class AutoQso:
    CALL_CQ = "Call CQ"
    ANSWER_CQ = "Answer CQs"
    MANUAL_CQ = "Manual CQ"
    WAIT_TIMEOUT_S = 12.0

    def __init__(self, my_call: str, grid: str, operation: str) -> None:
        self.my_call = my_call.strip().upper()
        self.grid = grid.strip().upper()
        self.operation = operation
        self.target = ""
        self.remote_snr = -8
        self.state = "idle"
        self.deadline = 0.0
        
        pack_cq(self.my_call, self.grid)

    def start(self) -> AutoAction | None:
        self.target = ""
        self.deadline = 0.0
        if self.operation in (self.CALL_CQ, self.MANUAL_CQ):
            self.state = "calling"
            return AutoAction(pack_cq(self.my_call, self.grid), f"CQ {self.my_call} {self.grid}")
        self.state = "listening"
        return None

    def repeat_cq(self) -> AutoAction | None:
        if self.operation in (self.CALL_CQ, self.MANUAL_CQ) and self.state == "calling":
            return AutoAction(pack_cq(self.my_call, self.grid), f"CQ {self.my_call} {self.grid}")
        return None

    def _wait(self, state: str, now: float) -> None:
        self.state = state
        self.deadline = now + self.WAIT_TIMEOUT_S

    def hear(
        self,
        row: tuple[str, str, str],
        snr: int,
        *,
        now: float | None = None,
    ) -> AutoAction | None:
        now = time.monotonic() if now is None else float(now)
        first, second, field = (str(x).strip().upper() for x in row)
        report = max(-35, min(28, int(round(snr))))

        if self.operation == self.ANSWER_CQ:
            if self.state == "listening" and first == "CQ" and second != self.my_call:
                self.target = second
                self.remote_snr = report
                self._wait("wait_report", now)
                return AutoAction(
                    pack_reply(self.target, self.my_call, self.grid),
                    f"{self.target} {self.my_call} {self.grid}",
                )
            if self.state == "wait_report" and first == self.my_call and second == self.target:
                if field.startswith(("+", "-")):
                    self._wait("wait_rr73", now)
                    tag = f"R{self.remote_snr:+03d}"
                    return AutoAction(
                        pack_rpt(self.target, self.my_call, self.remote_snr, True),
                        f"{self.target} {self.my_call} {tag}",
                    )
            
            
            if self.state == "wait_report" and second == self.target and first != self.my_call:
                if field.startswith(("+", "-")):
                    self.target = ""
                    self.state = "listening"
                    self.deadline = 0.0
            if self.state == "wait_rr73" and first == self.my_call and second == self.target:
                if field in ("RR73", "RRR"):
                    self.state = "complete"
                    self.deadline = 0.0
                    return AutoAction(
                        pack_73(self.target, self.my_call),
                        f"{self.target} {self.my_call} 73",
                    )
            return None

        if self.state == "calling" and first == self.my_call and second != self.my_call:
            if len(field) == 4 and field[:2].isalpha() and field[2:].isdigit():
                self.target = second
                self._wait("wait_r_report", now)
                return AutoAction(
                    pack_rpt(self.target, self.my_call, report, False),
                    f"{self.target} {self.my_call} {report:+03d}",
                )
        if self.state == "wait_r_report" and first == self.my_call and second == self.target:
            if field.startswith("R+") or field.startswith("R-"):
                self._wait("wait_73", now)
                return AutoAction(
                    pack_rr73(self.target, self.my_call),
                    f"{self.target} {self.my_call} RR73",
                )
        if self.state == "wait_73" and first == self.my_call and second == self.target:
            if field == "73":
                self.state = "complete"
                self.deadline = 0.0
        return None

    def expire(self, *, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else float(now)
        if not self.deadline or now < self.deadline:
            return None
        self.target = ""
        self.deadline = 0.0
        self.state = (
            "calling"
            if self.operation in (self.CALL_CQ, self.MANUAL_CQ)
            else "listening"
        )
        return self.state

    def resume(self) -> AutoAction | None:
        if self.operation == self.MANUAL_CQ:
            return None
        return self.start()

