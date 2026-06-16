#!/usr/bin/env python
"""Apply per-point classification labels from a text file to a point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pdal
from numpy.lib import recfunctions as rfn


def default_output_path(point_cloud_path: Path) -> Path:
    return point_cloud_path.with_name(f"{point_cloud_path.stem}_classified.laz")


def read_point_cloud(point_cloud_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    pipeline = pdal.Pipeline(json.dumps([str(point_cloud_path)]))
    pipeline.execute()

    if len(pipeline.arrays) != 1:
        raise RuntimeError(f"Expected one point view, got {len(pipeline.arrays)}")

    return pipeline.arrays[0].copy(), pipeline.metadata


def read_classifications(classification_path: Path, point_count: int) -> np.ndarray:
    values = np.loadtxt(classification_path)

    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError("Classification text file must contain exactly one value per row")

    if values.shape[0] != point_count:
        raise ValueError(
            f"Classification row count ({values.shape[0]}) does not match point count ({point_count})"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Classification values must be finite")

    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("Classification values must be integers")

    if np.any((rounded < 0) | (rounded > 255)):
        raise ValueError("Classification values must be in the LAS uint8 range [0, 255]")

    return rounded.astype(np.uint8)


def set_classification(points: np.ndarray, classifications: np.ndarray) -> np.ndarray:
    if "Classification" in points.dtype.names:
        points["Classification"] = classifications
        return points

    return rfn.append_fields(
        points,
        "Classification",
        classifications,
        dtypes=np.uint8,
        usemask=False,
    )


def reader_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    stages = metadata.get("metadata", metadata)
    reader_entries = [value for key, value in stages.items() if key.startswith("readers.")]

    if len(reader_entries) != 1:
        raise RuntimeError(f"Expected metadata for one reader, got {len(reader_entries)}")

    return reader_entries[0]


def writer_options(output_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    reader = reader_metadata(metadata)
    options: dict[str, Any] = {
        "type": "writers.las",
        "filename": str(output_path),
        "forward": "all",
        "extra_dims": "all",
    }

    for name in ("scale_x", "scale_y", "scale_z", "offset_x", "offset_y", "offset_z"):
        if name in reader:
            options[name] = reader[name]

    for name in ("minor_version", "dataformat_id"):
        if name in reader:
            options[name] = reader[name]

    srs = reader.get("srs", {})
    wkt = srs.get("wkt") or reader.get("spatialreference")
    if wkt:
        options["a_srs"] = wkt

    return options


def write_point_cloud(points: np.ndarray, output_path: Path, metadata: dict[str, Any]) -> None:
    pipeline = pdal.Pipeline(
        json.dumps([writer_options(output_path, metadata)]),
        arrays=[points],
    )
    pipeline.execute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overwrite or create the LAS Classification dimension from a text file "
            "with one integer class value per input point."
        )
    )
    parser.add_argument("point_cloud", type=Path, help="Input LAS/LAZ or other PDAL-readable point cloud")
    parser.add_argument("classifications", type=Path, help="Text file with one classification value per point")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output LAZ path. Defaults to <input_stem>_classified.laz next to the input.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    point_cloud_path = args.point_cloud
    classification_path = args.classifications
    output_path = args.output or default_output_path(point_cloud_path)

    if not point_cloud_path.exists():
        raise FileNotFoundError(point_cloud_path)
    if not classification_path.exists():
        raise FileNotFoundError(classification_path)

    points, metadata = read_point_cloud(point_cloud_path)
    classifications = read_classifications(classification_path, len(points))
    classified_points = set_classification(points, classifications)
    write_point_cloud(classified_points, output_path, metadata)

    print(f"Wrote {len(classified_points)} points to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
