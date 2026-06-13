import fnmatch
import os
from pathlib import Path

from .models import FileRecord


LANGUAGE_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
}

_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache",
              ".ruff_cache", ".pytest_cache", "dist", "build", ".tox"}


def _load_gitignore(directory: Path) -> list[str]:
    gi = directory / ".gitignore"
    if not gi.exists():
        return []
    patterns = []
    for line in gi.read_text(errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _is_ignored(rel: str, patterns: list[str]) -> bool:
    name = os.path.basename(rel)
    for p in patterns:
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p):
            return True
    return False


def si_dir_walker(root: Path) -> list[dict]:
    """Walk root depth-first, apply .gitignore rules. Returns [{absolute_path, relative_path}]."""
    root = Path(root)
    patterns = _load_gitignore(root)
    result = []

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        dirpath = Path(dirpath_str)
        rel_dir = dirpath.relative_to(root)

        # Prune dirs in-place so os.walk skips them entirely
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not d.startswith(".")
            and not _is_ignored(str(rel_dir / d), patterns)
        ]

        # Pick up any nested .gitignore
        nested_gi = dirpath / ".gitignore"
        if nested_gi.exists() and dirpath != root:
            patterns.extend(_load_gitignore(dirpath))

        for name in filenames:
            abs_path = dirpath / name
            rel_path = str(abs_path.relative_to(root))
            if not _is_ignored(rel_path, patterns):
                result.append({"absolute_path": str(abs_path), "relative_path": rel_path})

    return result


def si_file_reader(file_paths: list[dict]) -> list[dict]:
    """Read each file; skip binaries. Returns [{relative_path, text}]."""
    result = []
    for fp in file_paths:
        try:
            with open(fp["absolute_path"], "rb") as f:
                raw = f.read()
            if b"\x00" in raw[:8192]:
                continue  # binary
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            result.append({"relative_path": fp["relative_path"], "text": text})
        except OSError:
            pass
    return result


def si_language_classifier(raw_contents: list[dict]) -> list[FileRecord]:
    """Map extension → language tag; assemble FileManifest."""
    manifest = []
    for rc in raw_contents:
        ext = Path(rc["relative_path"]).suffix.lower()
        language = LANGUAGE_MAP.get(ext, "other")
        manifest.append(FileRecord(
            relative_path=rc["relative_path"],
            language=language,
            text=rc["text"],
        ))
    return manifest


def ingest(root: str) -> list[FileRecord]:
    paths = si_dir_walker(Path(root))
    contents = si_file_reader(paths)
    return si_language_classifier(contents)
