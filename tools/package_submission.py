#!/usr/bin/env python3
"""Build and verify the reviewer ZIP; preserve original manifest bytes."""

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".venv", "artifacts", "downloads", "dist", ".pytest_cache"}


def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and not any(part in EXCLUDED or part.startswith(".venv") for part in p.relative_to(root).parts)
                  and p.name not in {".DS_Store", "SHA256SUMS"} and not p.name.endswith((".pyc", ".kaggle-partial")))


def build(output, root=ROOT, archive_root="soup-hermes-t4-takehome"):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Write the ZIP outside the repository to avoid recursive inclusion")
    members = files(root)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in members}
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    members.append(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = archive_root + "/"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in members:
            info = zipfile.ZipInfo(prefix + str(path.relative_to(root)), date_time=(2026, 11, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        if len(archive.namelist()) != len(members):
            raise ValueError("ZIP member count differs")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != expected:
                raise ValueError("ZIP hash mismatch: " + name)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    return {"file": str(output), "files": len(members), "bytes": output.stat().st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(f"Verified ZIP: {result['files']} files, {result['bytes']/2**20:.2f} MiB\nSHA-256: {result['sha256']}")


if __name__ == "__main__":
    main()
