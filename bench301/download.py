"""Explicit download of official archives into ignored NB301 cache."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

ASSETS = {
    "data": (
        "https://ndownloader.figshare.com/files/25509518",
        "nb301_data.zip",
        "4d71f86400a4d9bfdfcd0235a899c975",
    ),
    "full-data": (
        "https://ndownloader.figshare.com/files/25595333",
        "nasbench301_full_data.zip",
        "c909b2a049f8b2f5117f626b12c0c824",
    ),
    "models": (
        "https://ndownloader.figshare.com/files/24992018",
        "nasbench301_models_v1.0.zip",
        "2cf9845823c67e323a5905a0321bfec8",
    ),
}


def download(asset, directory):
    url, filename, expected = ASSETS[asset]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.exists():
        raise FileExistsError(f"Asset already exists: {destination}; no automatic overwrite")
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Official archive MD5 mismatch; partial preserved at {partial}")
    partial.rename(destination)
    print(f"Verified {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=tuple(ASSETS))
    parser.add_argument("--directory", type=Path, default=Path("bench301/cache"))
    args = parser.parse_args()
    download(args.asset, args.directory)


if __name__ == "__main__":
    main()
