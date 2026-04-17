"""
-------------------------------------------
Baseball Detection Pre-Processing Pipeline
-------------------------------------------
Runs the full pre-processing pipeline in order:

    Step 1 — Frame Extraction
        Reads each .mov video file and saves every frame as a
        numbered .jpg image.

    Step 2 — Annotation Parsing
        Reads all CVAT .xml annotation files and compiles them
        into a single annotations.csv with one row per labeled frame.

    Step 3 — Frame Differencing
        Computes motion between consecutive frames using pixel-wise
        absolute difference. Outputs a motion_regions.csv and
        optional side-by-side debug visualizations.

Directory layout expected before running:
    videos/          ← your .mov files go here
    annotations/     ← your CVAT .xml files go here

Output after running:
    frames/                        ← extracted .jpg frames per video
    annotations.csv                ← parsed CVAT bounding box labels
    diff_output/                   ← debug visualizations (if enabled)
    motion_regions.csv             ← frame differencing detections

Requirements:
    pip install opencv-python
"""

import cv2
import os
import csv
import xml.etree.ElementTree as ET
import numpy as np


# -----------------------------------------------------------------
#  CONFIGURATION — update these paths and parameters before running
# -----------------------------------------------------------------

VIDEOS_DIR = "videos"
ANNOTATIONS_DIR = "annotations"
FRAMES_DIR = "frames"

# Step 1: Frame Extraction

JPEG_QUALITY = 95
ANNOTATIONS_CSV = "annotations.csv"

# Output folder for diff debug visualizations
DIFF_OUTPUT_DIR = "diff_output"

# Output CSV for frame differencing detections
MOTION_CSV = "motion_regions.csv"
BLUR_KERNEL = (5, 5)
DIFF_THRESHOLD = 25
MIN_CONTOUR_AREA = 50
SAVE_VISUALS = True



# -----------------------------------------------------------------
#  STEP 1: FRAME EXTRACTION
# -----------------------------------------------------------------


def extract_frames_from_video(video_path: str, output_folder: str) -> int:
    """
    Extract every frame from a single .mov file and save as numbered .jpg images.

    Parameters
    ----------
    video_path    : full path to the .mov file
    output_folder : directory where frames will be saved

    Returns
    -------
    Number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return 0

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        filename = f"frame_{frame_count:04d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_count += 1

    cap.release()
    return frame_count


def run_frame_extraction():
    """Step 1 entry point: extract frames from all .mov files in VIDEOS_DIR."""
    print("=" * 60)
    print("STEP 1: Frame Extraction")
    print("=" * 60)

    if not os.path.isdir(VIDEOS_DIR):
        print(f"[ERROR] Videos directory not found: '{VIDEOS_DIR}'")
        print("Create the folder and add your .mov files, then re-run.")
        return False

    video_files = sorted([f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(".mov")])

    if not video_files:
        print(f"[WARNING] No .mov files found in '{VIDEOS_DIR}'")
        return False

    print(f"Found {len(video_files)} video(s)\n")
    total_frames = 0

    for video_file in video_files:
        video_path = os.path.join(VIDEOS_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(FRAMES_DIR, video_name)

        print(f"  Processing: {video_file}")
        n = extract_frames_from_video(video_path, output_folder)
        total_frames += n
        print(f"  Extracted {n} frames → {output_folder}")

    print(f"\nTotal frames extracted: {total_frames}\n")
    return True


# -----------------------------------------------------------------
#  STEP 2: ANNOTATION PARSING
# -----------------------------------------------------------------


def parse_cvat_xml(xml_path: str) -> list[dict]:
    """
    Parse a single CVAT XML file into a list of bounding box records.

    Each record contains: video name, frame number, label, bounding box
    coordinates, outside flag, occluded flag, and moving attribute.

    Frames marked outside="1" mean the ball is not visible — these are
    included in the CSV but flagged so they can be filtered during training.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Source video name is stored in the task metadata
    source_elem = root.find("./meta/task/source")
    video_name = source_elem.text.strip() if source_elem is not None else os.path.basename(xml_path)

    rows = []
    for track in root.findall("track"):
        label = track.attrib.get("label", "unknown")

        for box in track.findall("box"):
            frame   = int(box.attrib["frame"])
            outside = int(box.attrib.get("outside", 0))   # 1 = ball not visible in frame
            occluded = int(box.attrib.get("occluded", 0))
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            # Read the optional "moving" attribute
            moving = None
            for attr in box.findall("attribute"):
                if attr.attrib.get("name") == "moving":
                    moving = attr.text.strip().lower() == "true"

            rows.append({
                "video":    video_name,
                "frame":    frame,
                "label":    label,
                "xtl":      xtl,
                "ytl":      ytl,
                "xbr":      xbr,
                "ybr":      ybr,
                "outside":  outside,
                "occluded": occluded,
                "moving":   moving,
            })

    return rows


def run_annotation_parsing():
    """Step 2 entry point: parse all CVAT .xml files into a single CSV."""
    print("=" * 60)
    print("STEP 2: Annotation Parsing")
    print("=" * 60)

    if not os.path.isdir(ANNOTATIONS_DIR):
        print(f"[WARNING] Annotations directory not found: '{ANNOTATIONS_DIR}'")
        print("Skipping annotation parsing — add .xml files and re-run if needed.\n")
        return False

    xml_files = sorted([f for f in os.listdir(ANNOTATIONS_DIR) if f.lower().endswith(".xml")])

    if not xml_files:
        print(f"[WARNING] No .xml files found in '{ANNOTATIONS_DIR}'\n")
        return False

    print(f"Found {len(xml_files)} annotation file(s)\n")

    all_rows = []
    for xml_file in xml_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        rows = parse_cvat_xml(xml_path)
        all_rows.extend(rows)
        visible = sum(1 for r in rows if r["outside"] == 0)
        print(f"  {xml_file}: {len(rows)} boxes ({visible} visible, {len(rows)-visible} outside)")

    fieldnames = ["video", "frame", "label", "xtl", "ytl", "xbr", "ybr", "outside", "occluded", "moving"]
    with open(ANNOTATIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    visible_total = sum(1 for r in all_rows if r["outside"] == 0)
    print(f"\nTotal rows written to '{ANNOTATIONS_CSV}': {len(all_rows)}")
    print(f"  Visible (outside=0): {visible_total}  |  Not visible (outside=1): {len(all_rows)-visible_total}\n")
    return True


# -----------------------------------------------------------------
#  STEP 3: FRAME DIFFERENCING
# -----------------------------------------------------------------


def load_frames(video_folder: str) -> list[tuple[str, np.ndarray]]:
    """
    Load all .jpg frames from a folder in sorted order.

    Returns list of (filename, image_array) tuples.
    """
    files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(".jpg")])
    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(video_folder, fname))
        if img is not None:
            frames.append((fname, img))
        else:
            print(f"  [WARNING] Could not read: {fname}")
    return frames


def compute_diff_mask(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """
    Compute a binary motion mask between two consecutive frames.

    Converts to grayscale, applies Gaussian blur to suppress noise,
    takes the absolute pixel-wise difference, then thresholds to
    produce a binary mask where 255 = motion and 0 = static.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    blur_a = cv2.GaussianBlur(gray_a, BLUR_KERNEL, 0)
    blur_b = cv2.GaussianBlur(gray_b, BLUR_KERNEL, 0)

    diff = cv2.absdiff(blur_a, blur_b)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def find_motion_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes around motion regions in a binary mask.

    Filters out contours below MIN_CONTOUR_AREA to ignore noise.

    Returns list of (xtl, ytl, xbr, ybr) tuples.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes


def make_visualization(original: np.ndarray, mask: np.ndarray,
                        boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """
    Build a side-by-side debug image with three panels:
        Left:   original frame
        Center: binary motion mask
        Right:  original frame with detected bounding boxes drawn in green

    All panels are scaled to a common display height.
    """
    display_height = 480
    h, w = original.shape[:2]
    display_w = int(w * (display_height / h))

    orig_small = cv2.resize(original, (display_w, display_height))
    mask_small = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (display_w, display_height))

    annotated = original.copy()
    for (xtl, ytl, xbr, ybr) in boxes:
        cv2.rectangle(annotated, (xtl, ytl), (xbr, ybr), color=(0, 255, 0), thickness=6)
    annotated_small = cv2.resize(annotated, (display_w, display_height))

    return np.hstack([orig_small, mask_small, annotated_small])


def process_video_diff(video_name: str, frames_folder: str,
                       output_folder: str, csv_writer) -> int:
    """
    Run frame differencing on all frames for one video.

    Returns total number of motion detections found.
    """
    frames = load_frames(frames_folder)

    if len(frames) < 2:
        print(f"  [WARNING] Not enough frames to diff in: {frames_folder}")
        return 0

    if SAVE_VISUALS:
        os.makedirs(output_folder, exist_ok=True)

    total_detections = 0

    # Compare consecutive frame pairs: (0,1), (1,2), (2,3), ...
    for i in range(len(frames) - 1):
        fname_a, frame_a = frames[i]
        fname_b, frame_b = frames[i + 1]
        frame_number = i + 1  # Label by the second frame in each pair

        mask = compute_diff_mask(frame_a, frame_b)
        boxes = find_motion_boxes(mask)

        for (xtl, ytl, xbr, ybr) in boxes:
            csv_writer.writerow({
                "video":video_name,
                "frame":frame_number,
                "xtl":xtl,
                "ytl":ytl,
                "xbr":xbr,
                "ybr":ybr,
                "num_boxes": len(boxes),
            })

        total_detections += len(boxes)

        if SAVE_VISUALS:
            vis = make_visualization(frame_b, mask, boxes)
            out_name = os.path.splitext(fname_b)[0] + "_diff.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), vis)

    return total_detections


def run_frame_differencing():
    """Step 3 entry point: run frame differencing on all extracted frame folders."""
    print("=" * 60)
    print("STEP 3: Frame Differencing")
    print("=" * 60)

    if not os.path.isdir(FRAMES_DIR):
        print(f"[ERROR] Frames directory not found: '{FRAMES_DIR}'")
        print("Frame extraction must complete successfully first.\n")
        return False

    video_folders = sorted([
        d for d in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, d))
    ])

    if not video_folders:
        print(f"[WARNING] No video subfolders found in '{FRAMES_DIR}'\n")
        return False

    print(f"Found {len(video_folders)} video folder(s)\n")
    os.makedirs(DIFF_OUTPUT_DIR, exist_ok=True)

    csv_fields = ["video", "frame", "xtl", "ytl", "xbr", "ybr", "num_boxes"]

    with open(MOTION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for video_name in video_folders:
            frames_folder = os.path.join(FRAMES_DIR, video_name)
            output_folder = os.path.join(DIFF_OUTPUT_DIR, video_name)

            print(f"  Processing: {video_name}")
            detections = process_video_diff(video_name, frames_folder, output_folder, writer)
            print(f"  Motion detections: {detections}")

    print(f"\nMotion regions saved to '{MOTION_CSV}'")
    if SAVE_VISUALS:
        print(f"Visualizations saved to '{DIFF_OUTPUT_DIR}/'\n")
    return True


# -----------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------


def main():
    print("\nBaseball Detection Pre-Processing Pipeline")
    print("=" * 60 + "\n")

    # Step 1: Extract frames from .mov videos
    step1_ok = run_frame_extraction()
    if not step1_ok:
        print("[PIPELINE STOPPED] Fix the issue above and re-run.")
        return

    # Step 2: Parse CVAT XML annotations into a CSV
    # Non-fatal — pipeline continues even if no annotations exist yet
    run_annotation_parsing()

    # Step 3: Run frame differencing on extracted frames
    run_frame_differencing()

    print("=" * 60)
    print("Pipeline complete.")
    print(f"  Frames:          {FRAMES_DIR}/")
    print(f"  Annotations CSV: {ANNOTATIONS_CSV}")
    print(f"  Motion CSV:      {MOTION_CSV}")
    if SAVE_VISUALS:
        print(f"  Diff visuals:    {DIFF_OUTPUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()