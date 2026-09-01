from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


SPLITS = (
    "train_set.txt",
    "holdout_set.txt",
    "val_set.txt",
    "test_set.txt",
)

# Official DFC25 semantics:
#
# train_set + holdout_set:
#   visible labels used for training purposes
#
# val_set:
#   validation labels used during model development
#
# test_set:
#   final unseen-event test set
#
# Therefore, overlap between train and holdout is allowed by the
# official split definition. We only reject duplicate IDs inside
# the same split and overlap involving the final test set.


def read_ids(path: Path) -> list[str]:
    """Read non-empty sample IDs from a split file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    ids = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not ids:
        raise ValueError(
            f"Split file is empty: {path}"
        )

    counts = Counter(ids)

    duplicates = sorted(
        sample_id
        for sample_id, count in counts.items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            f"Duplicate IDs inside {path}: "
            f"{duplicates[:10]}"
        )

    return ids


def find_overlap(
    first: set[str],
    second: set[str],
) -> set[str]:
    """Return IDs shared by two splits."""

    return first.intersection(second)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate official BRIGHT DFC25 split files."
    )

    parser.add_argument(
        "--split-dir",
        default="data/splits/dfc25",
        help="Directory containing DFC25 split files.",
    )

    args = parser.parse_args()

    split_dir = Path(args.split_dir)

    split_data: dict[str, list[str]] = {}

    for filename in SPLITS:
        split_data[filename] = read_ids(
            split_dir / filename
        )

    print("BRIGHT DFC25 split validation")
    print("=" * 60)

    for filename, ids in split_data.items():
        print(f"{filename:<18} {len(ids):>5} IDs")

    train = set(split_data["train_set.txt"])
    holdout = set(split_data["holdout_set.txt"])
    val = set(split_data["val_set.txt"])
    test = set(split_data["test_set.txt"])

    print()
    print("Official split relationships")
    print("=" * 60)

    train_holdout = find_overlap(
        train,
        holdout,
    )

    train_val = find_overlap(
        train,
        val,
    )

    holdout_val = find_overlap(
        holdout,
        val,
    )

    print(
        "train ∩ holdout:",
        len(train_holdout),
        "(allowed by official DFC25 split)",
    )

    print(
        "train ∩ val:",
        len(train_val),
        "(reported by official files)",
    )

    print(
        "holdout ∩ val:",
        len(holdout_val),
        "(reported by official files)",
    )

    # Final test must remain isolated.
    test_overlap_sets = {
        "train": find_overlap(train, test),
        "holdout": find_overlap(holdout, test),
        "val": find_overlap(val, test),
    }

    print()
    print("Final test isolation")
    print("=" * 60)

    test_leakage = False

    for name, overlap in test_overlap_sets.items():
        print(
            f"{name} ∩ test:",
            len(overlap),
        )

        if overlap:
            test_leakage = True

            print(
                "  Example IDs:",
                sorted(overlap)[:10],
            )

    if test_leakage:
        raise SystemExit(
            "ERROR: Final test IDs overlap with a development split."
        )

    print()
    print(
        "PASS: Final test set is isolated from "
        "train, holdout, and validation."
    )

    # Verify unseen test events.
    print()
    print("Unseen-event verification")
    print("=" * 60)

    events = (
        "noto-earthquake",
        "marshall-wildfire",
    )

    for event in events:
        print(f"\nEvent: {event}")

        for filename, ids in split_data.items():
            matches = [
                sample_id
                for sample_id in ids
                if event in sample_id.lower()
            ]

            print(
                f"  {filename:<18} {len(matches):>3}"
            )

    test_events = {
        sample_id.split("_")[0]
        for sample_id in test
        if (
            "noto-earthquake" in sample_id.lower()
            or "marshall-wildfire" in sample_id.lower()
        )
    }

    if test_events != {
        "noto-earthquake",
        "marshall-wildfire",
    }:
        raise SystemExit(
            "ERROR: Expected both unseen DFC25 events "
            "in the final test set."
        )

    print()
    print(
        "PASS: Noto-Earthquake and Marshall-Wildfire "
        "are present in the final test set."
    )

    print()
    print(
        "DFC25 split validation completed successfully."
    )


if __name__ == "__main__":
    main()