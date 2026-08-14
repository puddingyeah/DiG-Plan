#!/usr/bin/env python3
"""Fail closed on files that should not enter the public DiG-Plan repository."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_ROOT_FILES = {
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements-analysis.txt",
    "requirements-inference.txt",
    "requirements.txt",
}
ALLOWED_TOP_LEVEL_DIRS = {
    ".github",
    "artifacts",
    "assets",
    "data",
    "docs",
    "licenses",
    "scripts",
    "tests",
}
ALLOWED_PICKLES = {"artifacts/value_function/plan_scorer_combo07_toolset.pkl"}
EXPECTED_SHA256 = {
    "data/taskbench/taskbench_hf_improved_flattened.jsonl": "113fda5637517d4bfc5bb2d64c6fa754e479116f3aa4ba0d6e419f07032cd23a",
    "data/ids_500.txt": "084f973b75cbca5bfb139ebbd820827876b0066913fd259147bd16510ec643f3",
    "artifacts/candidate_pools/taskbench_dream_k5_train670.json": "f688be0ebe3a54893ba93b8e324ab915c672eb3ef2d854a8f27caa7de6473189",
    "artifacts/candidate_pools/taskbench_dream_k5_test334.json": "0e40ec04df89e411043f4e605233a99e788684d107a7da9c0c18b8b8b66779fa",
    "artifacts/value_function/plan_scorer_combo07_toolset.pkl": "5d058a0ca3f7f0f3eb60773007da33ccbb4efd13618d04a0de654a12ae40cdc4",
    "artifacts/results/taskbench_selection_eval.json": "bb97d6817053fb8f95df554b4ca03b2f37af8b035b149d51dd159043bf6311a2",
}
REQUIRED_FILES = {
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/DATA.md",
    "docs/PUBLIC_RELEASE.md",
    "docs/REPRODUCIBILITY.md",
    "data/taskbench/taskbench_hf_improved_flattened.jsonl",
    "artifacts/candidate_pools/taskbench_dream_k5_test334.json",
    "artifacts/candidate_pools/taskbench_dream_k5_train670.json",
    "artifacts/results/taskbench_selection_eval.json",
    "artifacts/value_function/plan_scorer_combo07_toolset.pkl",
}

FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".key",
    ".log",
    ".p12",
    ".pem",
    ".pfx",
    ".pid",
    ".pt",
    ".pth",
    ".safetensors",
    ".sentinel",
    ".tar",
    ".tgz",
    ".zip",
}
FORBIDDEN_PATH_MARKERS = {
    "migration_manifest",
    "migration_packages",
    "copyright_transfer_agreement",
    "rebuttal",
    "nohup",
}

# The script itself contains detection expressions, so it is excluded from its
# own content scan. It is still checked for path, size, and symlink safety.
SELF = "scripts/audit_public_release.py"
CONTENT_PATTERNS = {
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,255}"),
    "OpenAI-style key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "private key block": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "assigned secret": re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[\"']?[^\s\"']{8,}"
    ),
    "local data-home path": re.compile(rb"/data[0-9]+/(?:home|users)/"),
    "local home path": re.compile(rb"/home/[A-Za-z0-9._-]+/"),
}


def release_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(x.decode("utf-8") for x in proc.stdout.split(b"\0") if x)


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    )
    files = release_files(root)
    errors: list[str] = []

    missing = sorted(REQUIRED_FILES - set(files))
    if missing:
        errors.extend(f"missing required file: {path}" for path in missing)

    for rel in files:
        path = root / rel
        lower = rel.lower()
        parts = Path(rel).parts

        if len(parts) == 1:
            if rel not in ALLOWED_ROOT_FILES:
                errors.append(f"unapproved root file: {rel}")
        elif parts[0] not in ALLOWED_TOP_LEVEL_DIRS:
            errors.append(f"unapproved top-level directory: {rel}")

        if any(marker in lower for marker in FORBIDDEN_PATH_MARKERS):
            errors.append(f"forbidden path marker: {rel}")

        suffixes = {suffix.lower() for suffix in path.suffixes}
        if suffixes & FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file type: {rel}")
        if path.suffix.lower() == ".pkl" and rel not in ALLOWED_PICKLES:
            errors.append(f"unapproved pickle: {rel}")

        if path.is_symlink():
            target = path.readlink()
            if target.is_absolute():
                errors.append(f"absolute symlink: {rel} -> {target}")
            continue
        if not path.is_file():
            errors.append(f"not a regular file: {rel}")
            continue

        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            errors.append(f"file exceeds 20 MiB: {rel} ({size} bytes)")

        expected_hash = EXPECTED_SHA256.get(rel)
        if expected_hash:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                errors.append(
                    f"checksum mismatch: {rel} (expected {expected_hash}, got {actual_hash})"
                )

        if rel == SELF or path.suffix.lower() in {".pdf", ".pkl"}:
            continue
        content = path.read_bytes()
        if b"\0" in content[:8192]:
            continue
        for label, pattern in CONTENT_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label} detected in: {rel}")

    if errors:
        print("PUBLIC RELEASE AUDIT: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    total = sum((root / rel).stat().st_size for rel in files if not (root / rel).is_symlink())
    print(f"PUBLIC RELEASE AUDIT: PASS ({len(files)} files, {total / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
