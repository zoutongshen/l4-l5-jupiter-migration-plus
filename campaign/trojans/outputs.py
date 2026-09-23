"""Writers for manifests and provenance. Callers pass the directory to write into."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from trojans.paths import REPO_ROOT


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def write_parquet(path: Path, frame) -> None:
    """A table written whole or not at all, the way write_json writes JSON.

    Long runs checkpoint into these every few results. A crash, a full disk or an
    unplugged external drive part-way through a plain ``to_parquet`` leaves a file
    that no longer reads, costing the whole run rather than the last few results.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def provenance() -> dict:
    import numpy
    import rebound

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "rebound": rebound.__version__,
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
        "host": platform.node(),
        "command": sys.argv,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_provenance(directory: Path) -> None:
    write_json(directory / "provenance.json", provenance())


class Manifest:
    """manifest.json for a raw download directory: one entry per file."""

    def __init__(self, directory: Path):
        self.path = directory / "manifest.json"
        self.entries = json.loads(self.path.read_text())["files"] if self.path.exists() else {}

    def has(self, filename: str) -> bool:
        return filename in self.entries

    def save_file(self, directory: Path, filename: str, download, note: str = "", flush: bool = True) -> None:
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(download.content)
        self.entries[filename] = {
            "url": download.url,
            "params": download.params,
            "fetched_utc": download.fetched_utc,
            "bytes": len(download.content),
            "sha256": sha256(download.content),
            **({"note": note} if note else {}),
            **download.extra,
        }
        if flush:
            self.flush()

    def flush(self) -> None:
        write_json(self.path, {"files": self.entries})
