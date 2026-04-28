"""
-------------------------------------------
Baseball Detection — Image Standardization + Train/Val/Test Split
-------------------------------------------
Step 4 in the pipeline, downstream of main.py.

What this script does
---------------------
    1. Loads annotations.csv (produced by main.py's Step 2)
    2. Normalizes the 'video' column so it matches the frame folders
       produced by Step 1 (strips .mov, .xml, etc.)
    3. Filters labels to the project target (moving baseballs only, by
       default) and drops rows with outside=1
    4. Assigns each video to a split (train/val/test) using either a
       manual mapping or a by-video ratio policy
    5. For every labeled frame, letterbox-resizes the source image so
       its longest side equals TARGET_MAX_DIM (aspect ratio preserved),
       rescales the bounding boxes to match, and writes the image to
       dataset/<split>/images/
    6. Writes a COCO-format annotations.json per split — drop-in
       compatible with torchvision detection models (Faster R-CNN,
       RetinaNet, etc.) via pycocotools
    7. Emits a plain-text summary of counts, splits, and warnings

Directory layout this script expects
    frames/<video>/frame_0000.jpg ...     (produced by main.py Step 1)
    annotations.csv                        (produced by main.py Step 2)

Directory layout this script produces
    dataset/
      train/
        images/<video>_frame_0026.jpg
        annotations.json      (COCO)
      val/
        images/
        annotations.json
      test/
        images/
        annotations.json
      summary.txt
      manifest.csv             (flat row-per-box for sanity checks)

Notes on the choice of defaults
    * TARGET_MAX_DIM=1333 matches torchvision's default Faster R-CNN
      GeneralizedRCNNTransform max_size. Standardizing on disk lets us
      skip the per-iteration resize and shaves training time.
    * Letterboxing (resize preserving aspect, no padding) is used here
      rather than square letterbox with gray padding, since torchvision
      detectors handle variable input sizes natively. This avoids
      introducing synthetic padding pixels that can confuse the network.

Requirements
    pip install opencv-python
    (no pycocotools dependency — we write the COCO JSON ourselves)
"""

import cv2
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------
#  CONFIGURATION — tweak these, then run `python prepare_dataset.py`
# -----------------------------------------------------------------

# Inputs (produced by main.py)
ANNOTATIONS_CSV = "annotations.csv"
FRAMES_DIR = "frames"

# Output root
DATASET_DIR = "dataset"

# Image standardization
TARGET_MAX_DIM = 1333            # longest side after resize
JPEG_QUALITY = 95

# Label filter: which annotations count as positives
#   "moving_only"  — only moving=True, outside=0   (project target)
#   "all_visible"  — every outside=0 row
#   "all"          — every row, even outside=1     (not recommended)
LABEL_FILTER = "moving_only"

# Split policy:
#   "manual"    — use MANUAL_SPLITS below (video_name -> split)
#   "by_video"  — deterministic split across videos using SPLIT_RATIOS
#
# Given the current data (nearly all labels in one video), by_video is
# degenerate — a manual override is usually the honest choice.
SPLIT_POLICY = "by_video"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Optional manual assignment. Keys are normalized video names (no .mov
# extension). Any video not listed here falls back to SPLIT_POLICY.
MANUAL_SPLITS: Dict[str, str] = {
    # "IMG_0036":          "train",
    # "IMG_0037":          "val",
    # "IMG_8223_amarnath": "test",
}

# Single class for now. If the class adds moving/stationary as
# separate classes later, extend this mapping.
CATEGORY_NAME = "baseball"
CATEGORY_ID = 1


# -----------------------------------------------------------------
#  VIDEO NAME NORMALIZATION
# -----------------------------------------------------------------
#
# main.py stores the video name from the CVAT <source> tag. When the
# source tag is missing, it falls back to the XML filename, giving
# names like 'IMG_8223_amarnath.xml'. Frame folders are named after
# the stripped video filename. We normalize both sides so they match.

_VIDEO_EXT_RE = re.compile(r"\.(mov|mp4|avi|mkv|xml)$", re.IGNORECASE)


def normalize_video_name(raw: str) -> str:
    """Strip any known extension so annotations match frame folders."""
    return _VIDEO_EXT_RE.sub("", raw.strip())


# -----------------------------------------------------------------
#  LOAD + FILTER ANNOTATIONS
# -----------------------------------------------------------------


def _row_is_positive(row: dict, mode: str) -> bool:
    """Return True if this CSV row should be kept given LABEL_FILTER."""
    if mode == "all":
        return True
    if row["outside"] != "0":
        return False
    if mode == "all_visible":
        return True
    if mode == "moving_only":
        return row.get("moving", "").strip().lower() == "true"
    raise ValueError(f"Unknown LABEL_FILTER: {mode}")


def load_annotations(csv_path: str, mode: str) -> List[dict]:
    """
    Read annotations.csv and return normalized rows. Every returned row
    has: video (normalized), frame (int), xtl/ytl/xbr/ybr (float).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")

    kept: List[dict] = []
    dropped = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not _row_is_positive(row, mode):
                dropped += 1
                continue
            kept.append({
                "video":   normalize_video_name(row["video"]),
                "frame":   int(row["frame"]),
                "xtl":     float(row["xtl"]),
                "ytl":     float(row["ytl"]),
                "xbr":     float(row["xbr"]),
                "ybr":     float(row["ybr"]),
                "moving":  row.get("moving", "").strip().lower() == "true",
            })

    print(f"Loaded {len(kept)} label rows (dropped {dropped}) from {csv_path}")
    print(f"  filter mode: {mode}")
    return kept


def group_by_frame(rows: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    """Group label rows by (video, frame) so we emit one image per frame."""
    groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["video"], r["frame"])].append(r)
    return groups


# -----------------------------------------------------------------
#  SPLIT ASSIGNMENT
# -----------------------------------------------------------------


def assign_splits(videos: Iterable[str]) -> Dict[str, str]:
    """
    Return {video_name: split}. Honors MANUAL_SPLITS first, then falls
    back to SPLIT_POLICY for any unassigned video.
    """
    videos = sorted(set(videos))
    assignment: Dict[str, str] = {}

    # 1. Manual overrides
    for v in videos:
        if v in MANUAL_SPLITS:
            assignment[v] = MANUAL_SPLITS[v]

    # 2. Policy for the rest
    remaining = [v for v in videos if v not in assignment]
    if SPLIT_POLICY == "manual":
        if remaining:
            raise ValueError(
                "SPLIT_POLICY='manual' but these videos are not in "
                f"MANUAL_SPLITS: {remaining}"
            )
    elif SPLIT_POLICY == "by_video":
        # Deterministic allocation using the ratio vector. We walk the
        # sorted video list and fill train, then val, then test so small
        # N gives a predictable result.
        n = len(remaining)
        target_counts = {
            "train": max(1, round(n * SPLIT_RATIOS["train"])),
            "val":   max(0, round(n * SPLIT_RATIOS["val"])),
            "test":  max(0, round(n * SPLIT_RATIOS["test"])),
        }
        # Adjust for rounding so counts sum to n
        diff = n - sum(target_counts.values())
        target_counts["train"] += diff

        i = 0
        for split in ("train", "val", "test"):
            for _ in range(target_counts[split]):
                if i >= n:
                    break
                assignment[remaining[i]] = split
                i += 1
    else:
        raise ValueError(f"Unknown SPLIT_POLICY: {SPLIT_POLICY}")

    return assignment


# -----------------------------------------------------------------
#  IMAGE STANDARDIZATION
# -----------------------------------------------------------------


def letterbox_resize(img, target_max: int) -> Tuple["cv2.Mat", float]:
    """
    Resize `img` so its longest side equals target_max, preserving
    aspect ratio. No padding — torchvision detectors accept variable
    input sizes. Returns (resized_image, scale_factor).
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest == target_max:
        return img, 1.0
    scale = target_max / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_box(box: Tuple[float, float, float, float], scale: float
              ) -> Tuple[float, float, float, float]:
    """Scale an xyxy bbox by a uniform factor."""
    xtl, ytl, xbr, ybr = box
    return (xtl * scale, ytl * scale, xbr * scale, ybr * scale)


def frame_source_path(video: str, frame: int) -> str:
    """Map (video, frame) back to the jpg produced by main.py Step 1."""
    return os.path.join(FRAMES_DIR, video, f"frame_{frame:04d}.jpg")


# -----------------------------------------------------------------
#  COCO JSON EMISSION
# -----------------------------------------------------------------


def build_coco_dict(
    split: str,
    image_records: List[dict],
    anno_records: List[dict],
) -> dict:
    """
    Assemble a COCO-format dict.

    image_records: list of {"id", "file_name", "width", "height"}
    anno_records:  list of {"id", "image_id", "bbox":[x,y,w,h], "area", ...}
    """
    return {
        "info": {
            "description": f"Baseball detection — {split} split",
            "version": "1.0",
            "contributor": "ECON 8310 team",
        },
        "licenses": [],
        "categories": [
            {"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": ""}
        ],
        "images": image_records,
        "annotations": anno_records,
    }


# -----------------------------------------------------------------
#  PER-SPLIT PROCESSING
# -----------------------------------------------------------------


def process_split(
    split: str,
    frames_for_split: Dict[Tuple[str, int], List[dict]],
    dataset_root: str,
    target_max: int,
) -> Tuple[dict, List[dict]]:
    """
    For every (video, frame) in this split, load the source frame,
    standardize it, rescale the boxes, and write the image. Collect
    COCO image + annotation records.

    Returns (coco_dict, manifest_rows).
    """
    split_dir = os.path.join(dataset_root, split)
    images_dir = os.path.join(split_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_records: List[dict] = []
    anno_records: List[dict] = []
    manifest_rows: List[dict] = []

    next_image_id = 1
    next_anno_id = 1
    missing_frames: List[str] = []

    for (video, frame), boxes in sorted(frames_for_split.items()):
        src_path = frame_source_path(video, frame)
        if not os.path.isfile(src_path):
            missing_frames.append(src_path)
            continue

        img = cv2.imread(src_path)
        if img is None:
            missing_frames.append(src_path + "  (unreadable)")
            continue

        resized, scale = letterbox_resize(img, target_max)
        h, w = resized.shape[:2]

        out_name = f"{video}_frame_{frame:04d}.jpg"
        out_path = os.path.join(images_dir, out_name)
        cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        image_records.append({
            "id": next_image_id,
            "file_name": out_name,
            "width": w,
            "height": h,
            "source_video": video,
            "source_frame": frame,
        })

        for b in boxes:
            xtl, ytl, xbr, ybr = scale_box(
                (b["xtl"], b["ytl"], b["xbr"], b["ybr"]), scale
            )
            bw = max(0.0, xbr - xtl)
            bh = max(0.0, ybr - ytl)
            anno_records.append({
                "id": next_anno_id,
                "image_id": next_image_id,
                "category_id": CATEGORY_ID,
                "bbox": [round(xtl, 2), round(ytl, 2),
                         round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "moving": bool(b["moving"]),
            })
            manifest_rows.append({
                "split": split,
                "image_file": out_name,
                "source_video": video,
                "source_frame": frame,
                "xtl": round(xtl, 2),
                "ytl": round(ytl, 2),
                "xbr": round(xbr, 2),
                "ybr": round(ybr, 2),
                "moving": bool(b["moving"]),
            })
            next_anno_id += 1

        next_image_id += 1

    coco_dict = build_coco_dict(split, image_records, anno_records)
    coco_path = os.path.join(split_dir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"  [{split}] wrote {len(image_records)} images, "
          f"{len(anno_records)} boxes -> {coco_path}")
    if missing_frames:
        print(f"  [{split}] WARNING: skipped {len(missing_frames)} frames "
              f"whose source .jpg was missing. Examples:")
        for p in missing_frames[:3]:
            print(f"      {p}")

    return coco_dict, manifest_rows


# -----------------------------------------------------------------
#  SUMMARY
# -----------------------------------------------------------------


def _box_size_stats(manifest_rows: List[dict]) -> List[str]:
    """Quick percentile report on box sizes (max-dim in post-resize pixels).
    Small boxes are the main training risk for this project, so surface them."""
    if not manifest_rows:
        return ["Box size stats: (no boxes)"]
    sizes = sorted(
        max(r["xbr"] - r["xtl"], r["ybr"] - r["ytl"]) for r in manifest_rows
    )
    def pct(p): return sizes[min(len(sizes) - 1, int(len(sizes) * p))]
    lines = [
        "Box size (longest edge, post-resize pixels)",
        f"  min   : {sizes[0]:.1f}",
        f"  p10   : {pct(0.10):.1f}",
        f"  median: {pct(0.50):.1f}",
        f"  p90   : {pct(0.90):.1f}",
        f"  max   : {sizes[-1]:.1f}",
    ]
    tiny = sum(1 for s in sizes if s < 16)
    if tiny:
        lines.append(
            f"  NOTE: {tiny}/{len(sizes)} boxes are <16 px on their longest "
            "edge. Default Faster R-CNN anchors start at 32 px — consider "
            "raising TARGET_MAX_DIM or customizing the anchor generator."
        )
    return lines


def write_summary(
    out_path: str,
    label_mode: str,
    split_assignment: Dict[str, str],
    per_split_counts: Dict[str, Tuple[int, int]],
    manifest_rows: Optional[List[dict]] = None,
) -> None:
    """Human-readable sanity-check report next to the dataset."""
    lines = []
    lines.append("Baseball Detection — Dataset Summary")
    lines.append("=" * 60)
    lines.append(f"Label filter: {label_mode}")
    lines.append(f"Target max dim: {TARGET_MAX_DIM} px")
    lines.append(f"Split policy: {SPLIT_POLICY}  ratios={SPLIT_RATIOS}")
    lines.append("")
    lines.append("Video -> split assignment")
    for v, s in sorted(split_assignment.items()):
        lines.append(f"  {v:40s}  -> {s}")
    lines.append("")
    lines.append("Per-split counts (images / boxes)")
    for split in ("train", "val", "test"):
        n_img, n_box = per_split_counts.get(split, (0, 0))
        lines.append(f"  {split:5s}: {n_img:5d} images, {n_box:5d} boxes")
    lines.append("")
    if manifest_rows is not None:
        lines.extend(_box_size_stats(manifest_rows))
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary -> {out_path}")


# -----------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------


def main(dataset_root: Optional[str] = None) -> None:
    print("\nBaseball Detection — Standardization + Split")
    print("=" * 60 + "\n")

    dataset_root = dataset_root or DATASET_DIR

    # 1. Load + filter annotations
    rows = load_annotations(ANNOTATIONS_CSV, LABEL_FILTER)
    if not rows:
        print("[ERROR] No rows passed the label filter — nothing to do.")
        sys.exit(1)

    # 2. Group by (video, frame)
    groups = group_by_frame(rows)
    videos = {v for (v, _) in groups}
    print(f"  {len(videos)} video(s), {len(groups)} labeled frame(s)\n")

    # 3. Split assignment
    assignment = assign_splits(videos)
    print("Split assignment:")
    for v in sorted(videos):
        print(f"  {v:40s}  -> {assignment[v]}")
    print()

    # 4. Bucket frames by split
    by_split: Dict[str, Dict[Tuple[str, int], List[dict]]] = {
        "train": {}, "val": {}, "test": {}
    }
    for (video, frame), boxes in groups.items():
        split = assignment[video]
        by_split.setdefault(split, {})[(video, frame)] = boxes

    # 5. Process each split
    os.makedirs(dataset_root, exist_ok=True)
    all_manifest: List[dict] = []
    per_split_counts: Dict[str, Tuple[int, int]] = {}
    for split in ("train", "val", "test"):
        frames_in_split = by_split.get(split, {})
        if not frames_in_split:
            print(f"  [{split}] empty — skipping")
            per_split_counts[split] = (0, 0)
            continue
        coco, manifest = process_split(
            split, frames_in_split, dataset_root, TARGET_MAX_DIM
        )
        per_split_counts[split] = (len(coco["images"]), len(coco["annotations"]))
        all_manifest.extend(manifest)
    print()

    # 6. Flat manifest for quick sanity checks
    manifest_path = os.path.join(dataset_root, "manifest.csv")
    if all_manifest:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_manifest[0].keys()))
            writer.writeheader()
            writer.writerows(all_manifest)
        print(f"Manifest  -> {manifest_path}")

    # 7. Summary
    summary_path = os.path.join(dataset_root, "summary.txt")
    write_summary(summary_path, LABEL_FILTER, assignment,
                  per_split_counts, manifest_rows=all_manifest)

    print("\nDone.")


if __name__ == "__main__":
    main()
