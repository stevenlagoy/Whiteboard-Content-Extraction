"""
Run frame sampling and keyframe extraction on a real video and save what was selected.
Example:
    python scripts/inspect_keyframes.py data/raw/test_footage/lecture.mp4 --interval 2 --threshold 0.02
"""

import argparse
import time
from pathlib import Path

import cv2

from whiteboard_extraction.data.datasets import PROCESSED_DIR
from whiteboard_extraction.utils.visualization import (
    format_timestamp,
    make_contact_sheet,
    save_keyframes,
)
from whiteboard_extraction.video.keyframes import change_fraction, extract_keyframes
from whiteboard_extraction.video.sampling import sample_frames


def main() -> None:
    parser = argparse.ArgumentParser(description==__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between sampled frames")
    parser.add_argument("--threshold", type=float, default=0.02, help="fraction of changed pixels needed to keep a frame")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    out_dir = args.out or PROCESSED_DIR / "keyframe_inspection" / args.video.stem
    stats = {"sampled": 0, "last_timestamp": 0.0}

    def counted():
        for timestamp, frame in sample_frames(args.video, args.interval):
            stats["sample"] += 1
            stats["last_timestamp"] = timestamp
            yield timestamp, frame

    start = time.perf_counter()
    keyframes = extract_keyframes(counted(), threshold=args.threshold)
    elapsed = time.perf_counter() - start

    if not keyframes:
        print("No frames were sampled. Check that the video path is correct.")
        return

    save_keyframes(keyframes, out_dir)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), make_contact_sheet(keyframes))

    duration = stats["last_timestamp"]
    print(f"Video length (approx.): {format_timestamp(duration)}")
    print(f"Sampled {stats["sampled"]} frames, kept {len(keyframes)} keyframes ({len(keyframes) / stats["sampled"]:.0%})")
    print(f"Processing time: {elapsed:.1f}s ({elapsed / max(duration, 1):.2f}x video runtime)\n")

    print(f"{'#':>3}  {'time':>8} change vs previous keyframe")
    for i, (timestamp, frame) in enumerate(keyframes):
        change = "first frame" if i == 0 else f"{change_fraction(keyframes[i - 1][i], frame):.1%}"
        print(f"{i + 1:>3}  {format_timestamp(timestamp):>8}  {change}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()