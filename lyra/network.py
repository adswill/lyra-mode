from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

# Filled after the Worker is deployed.
DEFAULT_URL = "https://lyra-presence.kristijonasvr.workers.dev"
TIMEOUT_S = 10
LEAVE_TIMEOUT_S = 3
HEARTBEAT_MS = 30_000
HELP = (
    "Connecting shows how many operators are online and adds you to the count."
)


class NetworkError(Exception):
    pass


def new_session_id() -> str:
    return str(uuid4())


def heartbeat(base_url: str, session_id: str, user_id: str = "") -> int:
    payload = {"id": session_id}
    if user_id:
        payload["user"] = user_id
    data = _request("POST", f"{base_url.rstrip('/')}/v1/heartbeat", payload)
    return _int_field(data, "online")


def count(base_url: str) -> int:
    data = _request("GET", f"{base_url.rstrip('/')}/v1/count")
    return _int_field(data, "online")


def unique(base_url: str, day: str | None = None) -> tuple[str, int]:
    stamp = parse_day(day)
    query = urllib.parse.urlencode({"day": stamp})
    data = _request("GET", f"{base_url.rstrip('/')}/v1/unique?{query}")
    got = str(data.get("day") or stamp)
    return got, _int_field(data, "unique")


def parse_day(value: str | None = None) -> str:
    raw = (value or "today").strip().lower()
    today = datetime.now(timezone.utc).date()
    if raw in ("", "today"):
        return today.isoformat()
    if raw in ("yesterday", "yday"):
        return (today - timedelta(days=1)).isoformat()
    if raw in ("tomorrow", "tmrw"):
        return (today + timedelta(days=1)).isoformat()
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise NetworkError("day must be YYYY-MM-DD, today, yesterday, or tomorrow") from exc


def leave(base_url: str, session_id: str) -> None:
    try:
        _request(
            "POST",
            f"{base_url.rstrip('/')}/v1/leave",
            {"id": session_id},
            timeout=LEAVE_TIMEOUT_S,
        )
    except Exception:
        return


def _int_field(data: dict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < 0:
        raise NetworkError("bad response")
    return value


def _request(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = TIMEOUT_S,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "Lyra/1"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise NetworkError(f"HTTP {exc.code}") from exc
    except Exception as exc:
        raise NetworkError(str(exc) or "network error") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NetworkError("bad response") from exc
    if not isinstance(data, dict):
        raise NetworkError("bad response")
    return data
