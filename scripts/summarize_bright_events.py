from __future__ import annotations

from collections import Counter
from pathlib import Path


SPLITS = (
    "train_set.txt",
    "holdout_set.txt",
    "val_set.txt",
    "test_set.txt",
)


def read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def event_name(sample_id: str) -> str:
    """Extract the BRIGHT event name from a sample ID."""
    return sample_id.rsplit("_", 1)[0]


def main() -> None:
    split_root = Path("data/splits/dfc25")

    for filename in SPLITS:
        path = split_root / filename

        if not path.is_file():
            raise SystemExit(
                f"Missing split file: {path}"
            )

        ids = read_ids(path)
        counts = Counter(
            event_name(sample_id)
            for sample_id in ids
        )

        print(f"\n{filename}")
        print("=" * 60)
        print(f"Total samples: {len(ids)}")

        for event, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"{event:<40} {count:>4}")


if __name__ == "__main__":
    main()