#!/usr/bin/env python3
"""Verify a local dataset copy against the published SHA-256 manifests.

The manifests pin the exact archive copy the paper audited. Split-audit results
are properties of a specific copy, so a reader must confirm their download
matches before applying the published indices.

Usage:
    python3 verify_manifests.py manifest <dataset_root> --out manifests/x.sha256
    python3 verify_manifests.py verify   <dataset_root> --manifest manifests/x.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def walk_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def cmd_manifest(root: Path, out: Path) -> int:
    lines = [
        f"{sha256_of(p)}  {p.relative_to(root).as_posix()}" for p in walk_files(root)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {out}")
    return 0


def read_manifest(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            entries[rel] = digest
    return entries


def cmd_verify(root: Path, manifest: Path) -> int:
    expected = read_manifest(manifest)
    actual = {p.relative_to(root).as_posix(): p for p in walk_files(root)}

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = [
        rel
        for rel in sorted(set(expected) & set(actual))
        if sha256_of(actual[rel]) != expected[rel]
    ]

    for rel in missing:
        print(f"MISSING  {rel}", file=sys.stderr)
    for rel in extra:
        print(f"EXTRA    {rel}", file=sys.stderr)
    for rel in changed:
        print(f"CHANGED  {rel}", file=sys.stderr)

    if missing or extra or changed:
        print(
            f"\n{len(missing)} missing, {len(extra)} extra, {len(changed)} changed "
            f"against {manifest.name}. This copy is NOT the audited one, so the "
            f"published indices may not apply to it.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(expected)} files match {manifest.name} exactly.")
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="write a SHA-256 manifest for a dataset copy")
    m.add_argument("root", type=Path)
    m.add_argument("--out", type=Path, required=True)

    v = sub.add_parser("verify", help="check a dataset copy against a manifest")
    v.add_argument("root", type=Path)
    v.add_argument("--manifest", type=Path, required=True)

    a = ap.parse_args(argv[1:])
    return cmd_manifest(a.root, a.out) if a.cmd == "manifest" else cmd_verify(a.root, a.manifest)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
