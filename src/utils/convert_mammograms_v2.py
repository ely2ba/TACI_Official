#!/usr/bin/env python3
import os
from pathlib import Path
import pydicom
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import json

def dicom_to_png(src: Path, dst: Path):
    ds = pydicom.dcmread(str(src))
    arr = ds.pixel_array.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(str(dst))
    print(f"Converted {src.name} → {dst}")

def xml_rois_to_mask(xml_path, dicom_path, mask_out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ds = pydicom.dcmread(dicom_path)
    shape = ds.pixel_array.shape  # (height, width)
    mask = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    all_points = []
    for roi in root.iter('dict'):
        points_tag = None
        for idx, child in enumerate(list(roi)):
            if child.tag == 'key' and child.text == 'Point_px':
                points_tag = idx + 1
                break
        if points_tag:
            arr = roi[points_tag]
            coords = []
            for pt in arr.iter('string'):
                numbers = pt.text.strip("()").split(",")
                if len(numbers) >= 2:
                    x, y = float(numbers[0]), float(numbers[1])
                    coords.append((x, y))
                    all_points.append((x, y))
            if len(coords) == 1:
                x, y = coords[0]
                r = 3  # pixel radius for dot
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
            elif len(coords) > 1:
                draw.polygon(coords, outline=255, fill=255)
    mask.save(mask_out_path)
    print(f"Saved mask to {mask_out_path}")
    return all_points, mask

def compute_bbox(points):
    if not points:
        raise ValueError("No points found in ROI.")
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
    print(f"Ground truth bbox: {bbox}")
    return bbox

def overlay_mask_on_dicom(dicom_path: Path, mask_path: Path, dst: Path):
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    arr_uint8 = arr.astype(np.uint8)
    mask = np.array(Image.open(mask_path)).astype(bool)
    overlay = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
    overlay[mask, 0] = 255  # red channel
    overlay[mask, 1:] = 0
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(str(dst))
    print(f"Created overlay {dst}")

def save_bbox_and_labels(bbox, labels, dst):
    with open(dst, 'w') as f:
        json.dump({'labels': labels, 'bbox': bbox}, f, ensure_ascii=False, indent=2)
    print(f"Saved bbox and labels to {dst}")

def png_mask_to_bbox_and_labels(mask_path: Path, bbox_path: Path, labels):
    mask = np.array(Image.open(mask_path).convert("L"))
    y_idxs, x_idxs = np.where(mask > 0)
    if len(x_idxs) == 0 or len(y_idxs) == 0:
        raise ValueError(f"No abnormality found in mask {mask_path}")
    xmin, xmax = int(x_idxs.min()), int(x_idxs.max())
    ymin, ymax = int(y_idxs.min()), int(y_idxs.max())
    bbox = [xmin, ymin, xmax, ymax]
    with open(bbox_path, "w") as f:
        json.dump({"labels": labels, "bbox": bbox}, f, ensure_ascii=False, indent=2)
    print(f"Saved bbox and labels {bbox}, {labels} to {bbox_path}")

def main():
    # Update these for your actual chosen files:
    normal_dicom = Path.home() / "Downloads" / "22678622_61b13c59bcba149e_MG_R_CC_ANON.dcm"  # normal
    abnormal_dicom = Path.home() / "Downloads" / "53581460_b231a8ba4dd4214f_MG_R_CC_ANON.dcm"  # abnormal (nódulo)
    abnormal_xml = Path.home() / "Downloads" / "53581460.xml"  # annotation for abnormal

    # Output directories
    image_dir = Path("assets/images/chest_xray")
    gt_dir = image_dir / "ground_truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Convert DICOMs to PNG (images only)
    dicom_to_png(normal_dicom, image_dir / f"{normal_dicom.stem}.png")
    dicom_to_png(abnormal_dicom, image_dir / f"{abnormal_dicom.stem}.png")

    # Mammography (abnormal) labels
    mammogram_labels = ["nodule", "mass", "lesion", "abnormality", "tumor"]

    # Convert abnormal XML to mask and extract all ROI points (goes to ground_truth)
    mask_path = gt_dir / f"{abnormal_xml.stem}_mask.png"
    all_points, mask = xml_rois_to_mask(abnormal_xml, abnormal_dicom, mask_path)

    # Compute and save tightest bbox and labels (goes to ground_truth)
    bbox = compute_bbox(all_points)
    bbox_path = gt_dir / f"{abnormal_xml.stem}_bbox.json"
    save_bbox_and_labels(bbox, mammogram_labels, bbox_path)

    # Overlay mask on abnormal image for sanity check (goes to ground_truth)
    overlay_mask_on_dicom(abnormal_dicom, mask_path, gt_dir / f"{abnormal_dicom.stem}_overlay.png")

    # ===== Capsule/Parcel VISION TASK mask to bbox and labels logic =====
    parcel_mask = Path("assets/images/parcel_qc/ground_truth/000_mask.png")
    parcel_bbox = Path("assets/images/parcel_qc/ground_truth/000_bbox.json")
    parcel_labels = ["crack", "defect", "damage", "broken capsule", "anomaly"]
    if parcel_mask.exists():
        png_mask_to_bbox_and_labels(parcel_mask, parcel_bbox, parcel_labels)
    else:
        print(f"No parcel mask found at {parcel_mask} (skipped bbox/label generation)")

if __name__ == "__main__":
    main()
