"""The repo keeps the structure described in README.md.

A new top-level entry means updating the allowlist here and the map in README.md
in the same commit.
"""

from trojans.paths import CAMPAIGN_ROOT, DATA_LINK, REPO_ROOT

# Created by tools and ignored by git; allowed anywhere.
TOOL_DEBRIS = {".DS_Store", "__pycache__", ".pytest_cache"}

REPO_ROOT_ALLOWED = {
    ".git",
    ".gitignore",
    "README.md",
    "environment.yml",
    "L4-L5-Jupiter-Migration-Plus.code-workspace",
    "data",
    "docs",
    "literature",
    "migration",
    "campaign",
}

CAMPAIGN_ROOT_ALLOWED = {
    "README.md",
    "pyproject.toml",
    "configs",
    "trojans",
    "scripts",
    "slurm",
    "tests",
    "reports",
}


def _unexpected(directory, allowed):
    return sorted(
        p.name for p in directory.iterdir()
        if p.name not in allowed and p.name not in TOOL_DEBRIS
    )


def test_repo_root_contains_only_allowlisted_entries():
    assert _unexpected(REPO_ROOT, REPO_ROOT_ALLOWED) == []


def test_campaign_root_contains_only_allowlisted_entries():
    assert _unexpected(CAMPAIGN_ROOT, CAMPAIGN_ROOT_ALLOWED) == []


def test_data_is_a_symlink_not_a_folder():
    assert DATA_LINK.is_symlink()


def test_docs_hold_only_markdown():
    docs = REPO_ROOT / "docs"
    assert _unexpected(docs, {p.name for p in docs.glob("*.md")}) == []


def test_reports_hold_only_run_directories():
    reports = CAMPAIGN_ROOT / "reports"
    if reports.exists():
        assert [p.name for p in reports.iterdir() if p.is_file() and p.name not in TOOL_DEBRIS] == []


def test_no_pickles_in_campaign():
    assert sorted(str(p) for p in CAMPAIGN_ROOT.rglob("*.pkl")) == []
