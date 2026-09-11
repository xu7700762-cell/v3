from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "reproducibility" / "protocols" / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the v27 FEMBA checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "local_assets" / "pretrained_femba_v27.ckpt",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = manifest["pretrained_femba"]
    output = args.output.expanduser().resolve()
    expected = str(metadata["sha256"])
    if output.is_file():
        actual = sha256_file(output)
        if actual == expected:
            print(f"Already verified: {output}")
            return 0
        raise RuntimeError(
            f"Refusing to overwrite checkpoint with SHA-256 {actual}: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        print(f"Downloading {metadata['download_url']}")
        urllib.request.urlretrieve(str(metadata["download_url"]), temporary)
        actual = sha256_file(temporary)
        if actual != expected:
            raise RuntimeError(
                f"Downloaded checkpoint SHA-256 mismatch: expected {expected}, found {actual}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
