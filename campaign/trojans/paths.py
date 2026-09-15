"""Every file path the campaign uses is built here, and only here.

Heavy data lives under the data root, which is ``$TROJAN_DATA`` if set (ALICE) or
the repo-root ``data`` symlink to the external SSD. Small tracked outputs go to
``campaign/reports/<run_id>/``.

Functions that return a path never create anything. Only
:func:`make_output_dir` creates directories, and only scripts call it.
"""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = REPO_ROOT / "campaign"
DATA_LINK = REPO_ROOT / "data"
SSD_MOUNT = Path("/Volumes/Crucial-X10-Pro")

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DataRootMissing(RuntimeError):
    """The data root (normally the SSD) is not available."""


class OutputExists(FileExistsError):
    """An output directory already exists and resuming was not requested."""


def _check_id(value: str, kind: str) -> str:
    if not _ID_PATTERN.match(value):
        raise ValueError(f"invalid {kind} {value!r}: use letters, digits, '.', '_' or '-'")
    return value


def data_root() -> Path:
    """Resolved data root. Raises instead of falling back to a local folder."""
    env = os.environ.get("TROJAN_DATA")
    if env:
        root = Path(env).expanduser()
        if not root.is_dir():
            raise DataRootMissing(f"TROJAN_DATA={env} is not a directory")
        return root.resolve()
    if not DATA_LINK.is_symlink():
        raise DataRootMissing(f"{DATA_LINK} is not a symlink to the data root")
    root = DATA_LINK.resolve()
    if not root.is_dir():
        raise DataRootMissing(
            f"SSD not mounted: {SSD_MOUNT} ({DATA_LINK} -> {os.readlink(DATA_LINK)})"
        )
    return root


# --- Data root (heavy, not in git) -----------------------------------------

def raw_dir(source: str, fetch_date: str) -> Path:
    """Immutable downloads from one source on one date (YYYY-MM-DD)."""
    return data_root() / "raw" / _check_id(source, "source") / _check_id(fetch_date, "date")


def catalogue_dir(catalogue_id: str) -> Path:
    return data_root() / "catalogue" / _check_id(catalogue_id, "catalogue id")


def initial_dir(ic_id: str) -> Path:
    return data_root() / "initial" / _check_id(ic_id, "initial-condition id")


def run_dir(run_id: str) -> Path:
    return data_root() / "runs" / _check_id(run_id, "run id")


def chunk_dir(run_id: str, chunk: int) -> Path:
    return run_dir(run_id) / "chunks" / f"c{chunk:02d}"


def merged_dir(run_id: str) -> Path:
    return run_dir(run_id) / "merged"


# --- Repo (small, tracked) -------------------------------------------------

def config_path(name: str) -> Path:
    return CAMPAIGN_ROOT / "configs" / f"{_check_id(name, 'config name')}.toml"


def report_dir(run_id: str) -> Path:
    return CAMPAIGN_ROOT / "reports" / _check_id(run_id, "run id")


# --- Creation (scripts only) -----------------------------------------------

def make_output_dir(path: Path, resume: bool = False) -> Path:
    """Create an output directory, refusing to reuse one unless resuming."""
    if path.exists():
        if not resume:
            raise OutputExists(f"{path} already exists; pass --resume to continue it")
        return path
    if resume:
        raise FileNotFoundError(f"cannot resume: {path} does not exist")
    path.mkdir(parents=True)
    return path
