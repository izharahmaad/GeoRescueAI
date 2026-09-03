from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


SUFFIXES = {
    "pre-event": "_pre_disaster.tif",
    "post-event": "_post_disaster.tif",
    "target": "_building_damage.tif",
}


def find_member(
    archive: zipfile.ZipFile,
    directory: str,
    sample_id: str,
    suffix: str,
) -> str:
    """Find one sample file inside an archive."""

    expected_name = (
        f"{sample_id}{suffix}"
    )

    matches = [
        name
        for name in archive.namelist()
        if (
            name.endswith(
                f"/{directory}/{expected_name}"
            )
            or name.endswith(
                f"\\{directory}\\{expected_name}"
            )
        )
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {directory}/{expected_name} "
            "inside the archive."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple matches for {expected_name}: "
            f"{matches}"
        )

    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one BRIGHT sample from the trainval archive."
    )

    parser.add_argument(
        "--archive",
        required=True,
        help="Path to dfc25_track2_trainval.zip",
    )

    parser.add_argument(
        "--sample-id",
        required=True,
        help="BRIGHT sample ID.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output BRIGHT dataset directory.",
    )

    args = parser.parse_args()

    archive_path = Path(
        args.archive
    )

    output_root = Path(
        args.output
    )

    if not archive_path.is_file():
        raise SystemExit(
            f"Archive not found: {archive_path}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted = []

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:

        for directory, suffix in SUFFIXES.items():

            member = find_member(
                archive,
                directory,
                args.sample_id,
                suffix,
            )

            destination_dir = (
                output_root / directory
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            archive.extract(
                member,
                path=output_root,
            )

            extracted.append(
                member
            )

            print(
                f"Found {directory}: {member}"
            )

    print()
    print(
        f"Extracted sample: {args.sample_id}"
    )

    for member in extracted:
        print(
            f"  {member}"
        )


if __name__ == "__main__":
    main()