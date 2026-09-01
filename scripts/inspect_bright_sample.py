from __future__ import annotations

import argparse

import numpy as np

from georescue.bright_preprocessing import (
    BrightPreprocessingError,
    combine_modalities,
    load_bright_modalities,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one BRIGHT sample."
    )

    parser.add_argument(
        "--optical",
        required=True,
        help="Path to pre-event optical GeoTIFF.",
    )

    parser.add_argument(
        "--sar",
        required=True,
        help="Path to post-event SAR GeoTIFF.",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Path to building-damage target GeoTIFF.",
    )

    args = parser.parse_args()

    try:
        optical, sar, target = load_bright_modalities(
            optical_path=args.optical,
            sar_path=args.sar,
            target_path=args.target,
        )

        combined = combine_modalities(
            optical,
            sar,
        )

    except (
        BrightPreprocessingError,
        FileNotFoundError,
    ) as exc:
        raise SystemExit(
            f"BRIGHT validation failed: {exc}"
        ) from exc

    print("BRIGHT sample validation successful")
    print()
    print(f"Optical shape:  {optical.shape}")
    print(f"SAR shape:      {sar.shape}")
    print(f"Combined shape: {combined.shape}")
    print(f"Target shape:   {target.shape}")
    print()
    print(
        "Optical range:",
        float(optical.min()),
        "to",
        float(optical.max()),
    )
    print(
        "SAR range:",
        float(sar.min()),
        "to",
        float(sar.max()),
    )

    values, counts = np.unique(
        target,
        return_counts=True,
    )

    print(
        "Target values:",
        dict(
            zip(
                values.tolist(),
                counts.tolist(),
            )
        ),
    )


if __name__ == "__main__":
    main()