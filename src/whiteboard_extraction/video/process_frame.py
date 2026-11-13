import cv2

from .board_detection import detect_board_region, crop_to_board
from .occlusion import detect_occluding_objects
from pathlib import Path

frame_name = "keyframe_022_00-00-43"

data_path = Path.cwd().parent / "data"
frame_path = data_path / "processed" / "keyframe_inspection" / "calculus_clip_1" / (frame_name + ".jpg")
output_path = data_path / "processed" / "mask_inspection" / "calculus_clip_1" / frame_name
output_path.mkdir(parents=True, exist_ok=True)

frame = cv2.imread(frame_path)

board_box = detect_board_region(frame)
board = crop_to_board(frame, board_box)
visible_mask = detect_occluding_objects(board)
# Retain visible pixels; occluded pixels previewed as black
cleaned_preview = cv2.bitwise_and(board, board, mask=visible_mask)

cv2.imwrite(output_path / (frame_name + "_board_crop.png"), board)
cv2.imwrite(output_path / (frame_name + "_visible_mask.png"), visible_mask)
cv2.imwrite(output_path / (frame_name + "_visible_preview.png"), cleaned_preview)
