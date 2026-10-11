"""Run the whiteboard extraction pipeline on a video."""

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

from whiteboard_extraction.data.datasets import PROCESSED_DIR
from whiteboard_extraction.utils.visualization import make_contact_sheet, create_directory_contact_sheet
from .sampling import sample_frames
from .keyframes import extract_keyframes
from .board_detection import detect_board_region, crop_to_board
from .enhancement import enhance_board

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

STAGES = (
    "sampled_frames",
    "keyframe_inspection",
    "board_detection",
    "board_crops",
    "occlusion_masks",
    "masked_previews",
    "reconstructed",
    "enhanced",
    "output",
)


def timestamp_name(timestamp: float) -> str:
    """Format seconds as HH-MM-SS-mmm."""
    milliseconds = max(0, round(timestamp * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}-{minutes:02}-{seconds:02}-{millis:03}"


def frame_name(index: int, timestamp: float) -> str:
    return f"keyframe_{index:04}_{timestamp_name(timestamp)}"


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image, raise error if OpenCV fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_stage_dirs(output_root: Path, video_name: str) -> dict[str, Path]:
    directories = {
        stage: output_root / (str(i) + "_" + stage) / video_name
        for i, stage in enumerate(STAGES)
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def run_pipeline(
    video_path: Path,
    output_root: Path = PROCESSED_DIR,
    interval: float = 2.0,
    threshold: float = 0.02,
    quota: int = 0,
    skip_occlusion: bool = False,
    skip_inpainting: bool = False,
    skip_enhancement: bool = False,
) -> None:
    video_path = video_path.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")
    if interval <= 0:
        raise ValueError("Sampling interval must be greater than zero.")

    video_name = video_path.stem
    directories = make_stage_dirs(output_root, video_name)

    print(f"Video: {video_path}")
    print(f"Output: {output_root.resolve()}")
    print(f"Sampling every {interval:g} seconds...")

    # Sample the video and persist sampled frames
    sampled = list(sample_frames(video_path, interval))
    if not sampled:
        raise RuntimeError("No frames were sampled. Check that OpenCV can decode this video.")

    sampled_rows = []
    for index, (timestamp, frame) in enumerate(sampled):
        name = f"sample_{index:05}_{timestamp_name(timestamp)}.jpg"
        save_image(directories["sampled_frames"] / name, frame)
        sampled_rows.append({
            "index": index,
            "timestamp_seconds": timestamp,
            "filename": name,
        })

    write_csv(
        directories["sampled_frames"] / "manifest.csv",
        sampled_rows,
        ["index", "timestamp_seconds", "filename"],
    )
    print(f"Sampled {len(sampled)} frames.")

    # Select keyframes based on pixel differences
    keyframes = extract_keyframes(sampled, threshold=threshold, quota=quota)
    keyframe_rows = []
    frames_to_process = []
    for index, (timestamp, frame) in enumerate(keyframes):
        name = frame_name(index, timestamp)
        save_image(directories["keyframe_inspection"] / f"{name}.jpg", frame)
        keyframe_rows.append({
            "index": index,
            "timestamp_seconds": timestamp,
            "filename": f"{name}.jpg",
        })
        frames_to_process.append((name, timestamp, frame))

    write_csv(
        directories["keyframe_inspection"] / "manifest.csv",
        keyframe_rows,
        ["index", "timestamp_seconds", "filename"]
    )
    print(f"Selected {len(keyframes)} keyframes.")

    # Load the segmentation model if requested
    occlusion_detector = None
    if not skip_occlusion:
        try:
            from .occlusion import detect_occluding_objects
            occlusion_detector = detect_occluding_objects
        except ImportError as error:
            raise RuntimeError("Occlusion detection requires ultralytics. Install with `python -m pip install ultralytics`, or use --skip-occlusion.") from error
    
    board_rows = []
    processed_boards = []

    for index, (name, timestamp, frame) in enumerate(frames_to_process):
        print(f"[{index+1}/{len(frames_to_process)}] Processing {name}...")

        # Board localization, crop
        try:
            box = detect_board_region(frame)
            board = crop_to_board(frame, box)
        except (ValueError, cv2.error) as error:
            print(f"\tBoard detection failed: {error}", file=sys.stderr)
            board_rows.append({
                "filename": name,
                "timestamp_seconds": timestamp,
                "detected": False,
                "x": "",
                "y": "",
                "width": "",
                "height": "",
                "error": str(error),
            })
            continue

        # Save an annotated full-frame visualization
        x, y, w, h = box
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        save_image(directories["board_detection"] / f"{name}.jpg", annotated)
        save_image(directories["board_crops"] / f"{name}.jpg", board)

        board_rows.append({
            "filename": name,
            "timestamp_seconds": timestamp,
            "detected": True,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "error": "",
        })

        # Visible-board mask
        if occlusion_detector is None:
            visible_mask = np.full(board.shape[:2], 255, dtype=np.uint8)
        else:
            try:
                visible_mask = occlusion_detector(board)
            except Exception as error:
                # Keep pipeline usable
                print(f"\tOcclusion detection failed; using an all-visible maks: {error}", file=sys.stderr)
                visible_mask = np.full(board.shape[:2], 255, dtype=np.uint8)

        save_image(directories["occlusion_masks"] / f"{name}.png", visible_mask)

        preview = cv2.bitwise_and(board, board, mask=visible_mask)
        save_image(directories["masked_previews"] / f"{name}.jpg", preview)

        # Inpaint masked regions
        if skip_inpainting:
            reconstructed = board.copy()
        else:
            occluded_mask = cv2.bitwise_not(visible_mask)
            reconstructed = cv2.inpaint(board, occluded_mask, 3, cv2.INPAINT_TELEA)
        
        save_image(directories["reconstructed"] / f"{name}.jpg", reconstructed)

        # Improve local contrast
        if skip_enhancement:
            enhanced = reconstructed.copy()
        else:
            enhanced = enhance_board(reconstructed)

        save_image(directories["enhanced"] / f"{name}.jpg", enhanced)
        save_image(directories["output"] / f"{name}.jpg", enhanced)

        processed_boards.append({
            "filename": name,
            "timestamp_seconds": timestamp,
            "board_box": box,
        })

    write_csv(
        directories["board_detection"] / "manifest.csv",
        board_rows,
        [
            "filename", "timestamp_seconds", "detected",
            "x", "y", "width", "height", "error",
        ],
    )
    write_csv(
        directories["output"] / "manifest.csv",
        [
            {
                "filename": row["filename"] + ".jpg",
                "timestamp_seconds": row["timestamp_seconds"],
                "x": row["board_box"][0],
                "y": row["board_box"][1],
                "width": row["board_box"][2],
                "height": row["board_box"][3],
            }
            for row in processed_boards
        ],
        [
            "filename", "timestamp_seconds",
            "x", "y", "width", "height",
        ],
    )

    print("Creating contact sheets...")
    for directory in directories.values():
        if directory.is_dir():
            create_directory_contact_sheet(directory)        

    print(
        f"Finished: {len(processed_boards)}/{len(keyframes)} "
        f"keyframes successfully processed."
    )
    print(f"Final images: {directories['output']}")


def main() -> None:
    parser = argparse.ArgumentParser(
    description="Extract clean whiteboard/blackboard frames from a video."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between sampled frames (default: 2).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Pixel-change threshold for keyframe selection (default: 0.02).",
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=0,
        help="Target minimum number of keyframes; 0 disables the target.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Root output directory (default: data/processed).",
    )
    parser.add_argument(
        "--skip-occlusion",
        action="store_true",
        help="Skip YOLO segmentation and mark all pixels visible.",
    )
    parser.add_argument(
        "--skip-inpainting",
        action="store_true",
        help="Skip OpenCV inpainting.",
    )
    parser.add_argument(
        "--skip-enhancement",
        action="store_true",
        help="Skip CLAHE contrast enhancement.",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            video_path=args.video,
            output_root=args.output_dir,
            interval=args.interval,
            threshold=args.threshold,
            quota=args.quota,
            skip_occlusion=args.skip_occlusion,
            skip_inpainting=args.skip_inpainting,
            skip_enhancement=args.skip_enhancement,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")

if __name__ == "__main__":
    main()