"""Export a BoardDocument to a PDF file."""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from whiteboard_extraction.reconstruction.document import BoardDocument


def export_pdf(document: BoardDocument, output_path: Path) -> Path:
    """Write a BoardDocument to a PDF, mirroring the Word export's structure."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=LETTER)
    story = [Paragraph("Lecture Board Content", styles["Heading1"]), Spacer(1, 12)]

    for block in document.blocks:
        minutes, seconds = divmod(int(block.timestamp), 60)
        story.append(Paragraph(f"<b>{minutes:02d}:{seconds:02d}</b>", styles["Normal"]))
        story.append(Paragraph(block.text, styles["Normal"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    return output_path