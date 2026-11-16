"""Download the real public dataset used to check generalisation.

    python -m backend.scripts.fetch_public_data

The SMS Spam Collection (UCI, CC BY 4.0) is fetched once and cached under
``data/datasets/public/``. If the network is unavailable the script exits cleanly with a
message and every evaluation script records the public-data evaluation as *skipped*
rather than silently dropping it from the report.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from ..app.datasets import PUBLIC_DIR, SMS_SPAM, public_dataset_path, readme_for_public

URLS = [
    SMS_SPAM["source_url"],
    # Mirrors, tried in order. The UCI static host is the canonical source.
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
]


def download(urls: list[str], destination: Path, timeout: int = 60) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()
            if len(payload) < 10_000:
                last_error = ValueError(f"response too small ({len(payload)} bytes) from {url}")
                continue
            destination.write_bytes(payload)
            print(f"downloaded {len(payload):,} bytes from {url}")
            return True
        except Exception as exc:
            last_error = exc
            print(f"  failed: {url} ({type(exc).__name__}: {exc})")
    print(f"could not download the public dataset: {last_error}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the real public risk dataset.")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    target = public_dataset_path()
    if target.exists():
        print(f"already present: {target}")
    else:
        ok = download(URLS, target, timeout=args.timeout)
        if not ok:
            print(
                "\nThe public dataset is optional. Evaluation will still run on the "
                "synthetic corpus and will report the public-data stage as skipped.\n"
                f"Place the file manually at {target} to enable it.\n"
                f"Licence: {SMS_SPAM['licence']} - see the README written alongside it."
            )
            return 0

    readme = target.parent / "README.md"
    readme.write_text(readme_for_public(SMS_SPAM), encoding="utf-8")
    print(f"wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
