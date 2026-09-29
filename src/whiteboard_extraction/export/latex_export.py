"""Export equation blocks to a LaTeX (.tex) file."""

from pathlib import Path

from whiteboard_extraction.reconstruction.document import BoardDocument


def export_latex(document: BoardDocument, output_path: Path) -> Path:
    """Write recognized equation blocks to a compilable .tex file."""
    lines = [r"\documentclass{article}", r"\usepackage{amsmath}", r"\begin{document}"]
    for block in document.blocks:
        if block.kind == "equation":
            lines.append(f"\\[{block.text}\\]")
        else:
            lines.append(block.text.replace("_", r"\_"))
    lines.append(r"\end{document}")

    output_path.write_text("\n".join(lines))
    return output_path