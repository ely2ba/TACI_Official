#!/usr/bin/env python3
import os
from pathlib import Path

# Ensure you’ve installed these packages:
#    pip install pydicom pillow numpy

import pydicom
import numpy as np
from PIL import Image

def dicom_to_png(src: Path, dst: Path):
    """Load a DICOM file, normalize its pixel data, and save as PNG."""
    ds = pydicom.dcmread(str(src))
    arr = ds.pixel_array.astype(float)
    # Normalize to 0–255
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(str(dst))
    print(f"Converted {src.name} → {dst}")

def main():
    # Paths to your downloaded DICOM files
    dicom_paths = [
        Path.home() / "Downloads" / "Training_P_00005_RIGHT_CC.dcm",
        Path.home() / "Downloads" / "Training_P_00007_LEFT_CC.dcm",
    ]

    # Output directory for PNGs
    output_dir = Path("assets/images/cbis-ddsm/classification")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dicom_path in dicom_paths:
        if not dicom_path.exists():
            print(f"❌ File not found: {dicom_path}")
            continue

        base_name = dicom_path.stem  # e.g. Calc-Training_P_00005_RIGHT_CC
        png_path = output_dir / f"{base_name}.png"
        dicom_to_png(dicom_path, png_path)

if __name__ == "__main__":
    main()
