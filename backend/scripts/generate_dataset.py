"""Regenerate the seeded synthetic dataset.

    python -m backend.scripts.generate_dataset --n 320 --seed 20260310
"""

from __future__ import annotations

import argparse
import json

from ..app.datasets import SYNTHETIC_DIR, ensure_synthetic
from ..app.ingest.generator import write_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the seeded synthetic dataset.")
    parser.add_argument("--n", type=int, default=320, help="Number of records (>= 300).")
    parser.add_argument("--seed", type=int, default=20260310, help="Random seed.")
    parser.add_argument("--out", type=str, default=str(SYNTHETIC_DIR))
    args = parser.parse_args()

    from pathlib import Path

    out = Path(args.out)
    written = write_dataset(out, n=args.n, seed=args.seed)
    meta = json.loads((out / "dataset_meta.json").read_text(encoding="utf-8"))

    print(f"Synthetic dataset written to {out}")
    print(json.dumps(meta, indent=2))
    for name, path in written.items():
        print(f"  {name:6s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
