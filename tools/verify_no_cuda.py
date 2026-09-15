#!/usr/bin/env python3
"""Fail when legacy CUDA build/runtime dependencies re-enter the Ascend fork."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN_ROOTS = [
    ROOT / "include",
    ROOT / "src",
    ROOT / "python",
    ROOT / "tests",
    ROOT / "benchmarks",
    ROOT / "env",
]
SCAN_FILES = [
    ROOT / "CMakeLists.txt",
    ROOT / "pyproject.toml",
    ROOT / "environment.yml",
    ROOT / "justfile",
    ROOT / "conanfile.py",
]

TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".asc", ".py", ".pyi", ".cmake", ".toml", ".yml", ".yaml", ".sh",
}

# These patterns target real compile/runtime dependencies and API names. Plain
# prose containing the historical word "CUDA" is intentionally allowed.
FORBIDDEN_CONTENT = [
    re.compile(r"#\s*include\s*[<\"]cuda(?:_runtime)?\.h[>\"]", re.IGNORECASE),
    re.compile(r"\bCUDA::[A-Za-z0-9_]+"),
    re.compile(r"\bCMAKE_CUDA_[A-Za-z0-9_]+"),
    re.compile(r"\bCUDACXX\b"),
    re.compile(r"\bCUDA_HOME\b"),
    re.compile(r"\bNVCC(?:_PREPEND_FLAGS)?\b"),
    re.compile(r"\bcuda(?:Malloc|Free|Memcpy|Memset|GetDevice|SetDevice|GetDeviceCount|GetLastError|Stream|Event|Func|Device)[A-Za-z0-9_]*\b"),
    re.compile(r"\bcudaError_t\b"),
    re.compile(r"\bcudaStream_t\b"),
    re.compile(r"\bcudaMemcpy(?:HostToDevice|DeviceToHost|DeviceToDevice)\b"),
    re.compile(r"\b__shfl_(?:down_)?sync\b"),
    re.compile(r"\bcub::"),
]

# Source/build assets whose names encode the old backend are not allowed.
FORBIDDEN_PATH_PART = re.compile(r"(?:^|[_./-])cuda(?:[_./-]|$)", re.IGNORECASE)

# Source-contract tests deliberately contain CUDA-only tokens as *forbidden
# strings*. They are allowed to mention those strings but may not compile/link
# CUDA headers or APIs.
SOURCE_CONTRACT_SUFFIX = "_kernel_contract.py"


def iter_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(path for path in SCAN_FILES if path.exists())
    return sorted(set(files))


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix()

        if path.suffix.lower() in {".cu", ".cuh"}:
            failures.append(f"legacy CUDA source extension: {relative_text}")
        if FORBIDDEN_PATH_PART.search(relative_text):
            failures.append(f"legacy CUDA asset path: {relative_text}")

        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "CMakeLists.txt", "justfile"
        }:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name.endswith(SOURCE_CONTRACT_SUFFIX):
            # These files quote forbidden tokens to assert that Ascend kernel
            # sources do not contain them. Still reject true CUDA includes and
            # build-system dependencies in the contract tests themselves.
            patterns = FORBIDDEN_CONTENT[:6]
        else:
            patterns = FORBIDDEN_CONTENT

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{relative_text}:{line}: forbidden CUDA dependency token "
                    f"{match.group(0)!r}"
                )

    if failures:
        print("CUDA dependency gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("CUDA dependency gate PASS: no legacy CUDA build/runtime dependencies found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
