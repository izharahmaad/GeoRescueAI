from __future__ import annotations

import argparse

from georescue.bright_dataset import (
    BrightDatasetError,
    discover_bright_samples,
    inspect_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the local BRIGHT dataset structure."
    )

    parser.add_argument(
        "--root",
        default="data/raw/BRIGHT",
        help="Path to the BRIGHT dataset root.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of samples to inspect.",
    )

    args = parser.parse_args()

    try:
        samples = discover_bright_samples(args.root)
    except BrightDatasetError as exc:
        raise SystemExit(
            f"BRIGHT dataset error: {exc}"
        ) from exc

    print(f"Matched samples: {len(samples)}")

    for sample in samples[: args.limit]:
        info = inspect_sample(sample)

        print("\n" + "=" * 72)
        print(f"Sample ID: {info['sample_id']}")

        for modality in ("optical", "sar", "target"):
            metadata = info[modality]

            print(f"\n{modality.upper()}")
            print(f"  path:    {metadata['path']}")
            print(
                f"  size:    "
                f"{metadata['width']} x {metadata['height']}"
            )
            print(f"  bands:   {metadata['count']}")
            print(f"  dtype:   {metadata['dtype']}")
            print(f"  CRS:     {metadata['crs']}")
            print(f"  nodata:  {metadata['nodata']}")


if __name__ == "__main__":
    main()