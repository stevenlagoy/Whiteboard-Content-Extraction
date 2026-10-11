"""Pipeline orchestrator for the full chain of video -> frames -> recognition -> reconstrution -> output."""

from pathlib import Path

from whiteboard_extraction.export.docx_export import export_docx
from whiteboard_extraction.export.pdf_export import export_pdf
from whiteboard_extraction.recognition.base import Recognizer
from whiteboard_extraction.reconstruction.document import BoardDocument, build_document
from whiteboard_extraction.video.keyframes import extract_keyframes
from whiteboard_extraction.video.sampling import sample_frames

def process_video(video_path: Path, recognizer: Recognizer, output_dir: Path, job_id: str) -> BoardDocument:
    boards = sample_frames(video_path) # TODO crop_to_board(detect_board_region(...))
    keyframes = extract_keyframes(boards)
    document = build_document([recognizer.recognize(t, f) for t, f in keyframes])
    export_docx(document, output_dir / f"{job_id}.docx")
    export_pdf(document, output_dir / f"{job_id}.pdf")
    return document