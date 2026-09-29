"""Assemble recognized blocks into a single ordered, timestamped document."""

from dataclasses import dataclass

from whiteboard_extraction.recognition.base import RecognizedBlock


@dataclass
class BoardDocument:
    """The full reconstructed content of a lecture's board, in order."""
    blocks: list[RecognizedBlock]

    def as_plain_text(self) -> str:
        return "\n".join(f"[{b.timestamp:.0f}s] {b.text}" for b in self.blocks)


def build_document(all_blocks: list[list[RecognizedBlock]]) -> BoardDocument:
    """Merge per-keyframe recognition results into one ordered document."""
    flattened = [block for keyframe_blocks in all_blocks for block in keyframe_blocks]
    flattened.sort(key=lambda b: b.timestamp)
    return BoardDocument(blocks=flattened)