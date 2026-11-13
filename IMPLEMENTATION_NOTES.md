# Feature 7 + Feature 8 integration notes

## Included in this working copy

- Feature 8 remains the base branch and retains its video-processing stages.
- Full-resolution sampled frames are saved immediately rather than accumulated in memory.
- Board detection and cropping now happen before keyframe comparison.
- Keyframe selection compares fixed-size grayscale board thumbnails, not the entire classroom frame.
- Keyframe comparison uses the existing changed-pixel fraction plus SSIM (scikit-image).
- Only selected full-resolution frames are reloaded for downstream YOLO masking and enhancement.
- The visualization helper API supports the timestamped keyframe format used by the Feature 7 tests.
- README setup and pipeline instructions have been updated.

## Not yet implemented

Temporal reconstruction from neighboring frames and board tracking/perspective stabilization are intentionally not enabled in this first merge. Copying pixels from another timestamp without alignment and checking whether the writing changed can replace correct equations with incorrect content. The current inpainting stage remains a visual fallback, not true recovery of hidden writing.

## Validation

- `python -m compileall -q src tests`
- `pytest -q -m "not real_video"` — 26 passed, 2 deselected.
- The real-video tests were not included in this validation run because the clips are Git LFS assets and running both full-clip tests exceeded the available execution window.

## Run

Install dependencies with `pip install -r requirements.txt && pip install -e .`, then run:

`python -m whiteboard_extraction.video.pipeline data/raw/test_footage/calculus_clip_1.mp4 --skip-occlusion`

Omit `--skip-occlusion` to run YOLO segmentation. The first YOLO run may download model weights.
