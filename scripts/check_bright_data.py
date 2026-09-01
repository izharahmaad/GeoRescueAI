from __future__ import annotations

import argparse
from pathlib import Path

from georescue.bright_dataset import BrightDatasetError, discover_bright_samples


SPLITS = {
    "train": "train_set.txt",
    "holdout": "holdout_set.txt",
    "validation": "val_set.txt",
    "test": "test_set.txt",
}


def read_split_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check BRIGHT DFC25 data availability."
    )

    parser.add_argument(
        "--bright-root",
        required=True,
        help="Path to the BRIGHT dataset root.",
    )

    parser.add_argument(
        "--split-root",
        default="data/splits/dfc25",
        help="Path to the official DFC25 split directory.",
    )

    args = parser.parse_args()

    bright_root = Path(args.bright_root)
    split_root = Path(args.split_root)

    print("BRIGHT DFC25 data readiness")
    print("=" * 60)
    print(f"Dataset root: {bright_root}")
    print(f"Split root:   {split_root}")
    print()

    if not bright_root.is_dir():
        raise SystemExit(
            f"Dataset root does not exist: {bright_root}"
        )

    # Discover all physically available matched samples.
    try:
        samples = discover_bright_samples(
            bright_root
        )
    except BrightDatasetError as exc:
        raise SystemExit(
            f"Dataset discovery failed: {exc}"
        ) from exc

    available_ids = {
        sample.sample_id
        for sample in samples
    }

    print(
        f"Fully matched local samples: "
        f"{len(available_ids)}"
    )

    print()

    for split_name, filename in SPLITS.items():
        split_path = split_root / filename
        split_ids = read_split_ids(split_path)

        split_set = set(split_ids)
        available = split_set & available_ids
        missing = split_set - available_ids

        print(f"{split_name.upper()}")
        print("-" * 60)
        print(f"Official IDs:   {len(split_ids)}")
        print(f"Available:      {len(available)}")
        print(f"Missing:        {len(missing)}")

        if missing:
            print(
                "Missing examples:",
                sorted(missing)[:5],
            )

        print()

    print("Readiness check complete.")


if __name__ == "__main__":
    main()