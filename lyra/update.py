from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REPO = "adswill/lyra-mode"
BRANCH = "main"
TIMEOUT_S = 8
ROOT = Path(__file__).resolve().parents[1]


class UpdateError(Exception):
    pass


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=5,
    )


def local_sha() -> str | None:
    try:
        proc = _git("rev-parse", "HEAD")
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip().lower()
    return sha if len(sha) >= 7 else None


def _has_commit(sha: str) -> bool:
    try:
        proc = _git("cat-file", "-e", sha + "^{commit}")
    except Exception:
        return False
    return proc.returncode == 0


def _is_ancestor(maybe_old: str, maybe_new: str) -> bool:
    try:
        proc = _git("merge-base", "--is-ancestor", maybe_old, maybe_new)
    except Exception:
        return False
    return proc.returncode == 0


def remote_sha() -> str:
    url = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Lyra/1",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise UpdateError(f"HTTP {exc.code}") from exc
    except Exception as exc:
        raise UpdateError(str(exc) or "network error") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpdateError("bad response") from exc
    sha = str(data.get("sha") or "").strip().lower()
    if len(sha) < 7:
        raise UpdateError("bad response")
    return sha


def check() -> str:
    latest = remote_sha()
    here = local_sha()
    short = latest[:7]
    if here is None:
        return (
            f"Couldn't read this copy's version. Latest on GitHub is {short}."
        )
    if here == latest:
        return "This copy is up to date."
    if _has_commit(latest) and _is_ancestor(latest, here):
        return "This copy is up to date."
    return (
        "A new update is on GitHub.\n\n"
        f"This copy: {here[:7]}\n"
        f"GitHub:    {short}\n\n"
        "In the Lyra folder run:\n\ngit pull"
    )
