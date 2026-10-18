"""Sample library shown on first run so every screen has something to display.

The data is arranged to exercise the interface, including its awkward cases:

* Folders nest seven levels deep and include empty folders, repeated names under different
  parents, markup characters, non-Latin text, and names at the 120 character limit.
* Videos range from 2 seconds to exactly 150 minutes, with 0, 1, 3, 60, and 150 recognized blocks,
  one very long block, blocks that share a timestamp, blocks at the first and last instant, and
  fractional timestamps.
* Dates cover minutes ago to over a year ago, processing that finished just inside and just outside
  the 24 hour window, exports that predate later edits, and times such as 12:05 AM, 12:00 PM, and
  11:59 PM.
* Statuses cover processed, not processed, scheduled, failed (also on a video that has text),
  cancelled, and a video that failed once before succeeding.

The dashboard's Recents list is built to show a mix of these on first load: a video processed
minutes ago, a scheduled one, a failed one, an unprocessed one, and so on.

Videos have no media file, so the player is replaced by a notice. Tasks that would start at once
or never finish (queued, running) are not seeded; select "Process this video" on an unprocessed
entry to watch one run.

Every date is relative to the time the app starts, and nothing is random, so ids and content are
the same on each start.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from whiteboard_extraction.recognition.base import RecognizedBlock

from .domain.models import JobState, ProcessingJob
from .repositories.base import Repository
from .services.library import LibraryService

# ---------------------------------------------------------------------------- text banks

EQUATIONS = (
    r"\int_0^1 x^2 \, dx = \frac{1}{3}",
    r"A\mathbf{x} = \lambda \mathbf{x}",
    r"\det(A - \lambda I) = 0",
    r"T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})",
    r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}",
    r"\frac{\partial L}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)\, x_i",
    r"e^{i\pi} + 1 = 0",
    r"\|\mathbf{u} + \mathbf{v}\| \le \|\mathbf{u}\| + \|\mathbf{v}\|",
    r"x < 5 \text{ and } y > 2 \text{ & } z \ne 0",
    r"\lim_{h \to 0} \frac{f(x+h) - f(x)}{h} = f'(x)",
    "\u03a3\u1d62 x\u1d62\u00b2 \u2265 0 (typed with Unicode symbols)",
)

PROSE = (
    "Definition: a basis is a linearly independent spanning set.",
    "Remember: the null space is a subspace of the domain.",
    "Example 3.2: solve the system by row reduction",
    "Note to self: check whether x < 5 & y > 2 still holds",
    'He wrote "orthogonal" and underlined it twice',
    "It's the rank-nullity theorem; don't forget the dimension count",
    "<b>Not bold</b> and a literal &amp; sign",
    "Homework 4 is due Friday",
    "Why does the gradient point uphill?",
    "Quiz next week: chapters 2 and 3",
    "\u56fa\u6709\u5024\u3068\u56fa\u6709\u30d9\u30af\u30c8\u30eb (eigenvalues and eigenvectors)",
    "D\u00e9couverte: l'espace vectoriel et ses sous-espaces",
    "Path C:\\temp\\notes and a 100% sign",
)

_SENTENCES = (
    "A basis for a vector space is a linearly independent set that spans the space.",
    "Every basis of a finite-dimensional space has the same number of vectors, called the dimension.",
    "Row operations preserve the row space, so a basis can be read from the reduced echelon form.",
    "The rank of a matrix equals the number of pivot columns, and the nullity counts the free columns.",
    "Together they account for every column, which is the rank-nullity theorem.",
    "A change of basis is itself a linear map, and its matrix is always invertible.",
)
LONG_PROSE = "\n\n".join(" ".join(_SENTENCES) for _ in range(3))
LONG_EQUATION = (
    r"\begin{aligned} (A+B)^2 &= A^2 + AB + BA + B^2 \\ "
    r"\text{if } AB = BA &\Rightarrow (A+B)^2 = A^2 + 2AB + B^2 \\ "
    r"\text{and } (A+B)^3 &= A^3 + 3A^2B + 3AB^2 + B^3 \end{aligned}"
)

LONG_ERROR = (
    "Demo failure: the decoder reported corrupt data at 01:12:45 (frame 108,675 of 161,010). "
    "Segmentation could not find the board in 212 consecutive frames because the camera was covered, "
    "and the keyframe step then ran out of memory while buffering the remaining footage. "
    "Re-export the recording from the classroom system and try again."
)

LONG_FOLDER_NAME = ("Very long folder name " * 8)[:119] + "!"
LONG_VIDEO_NAME = ("Very long video title " * 6)[:119] + "!"
HTML_VIDEO_NAME = "Odd \"name\" <b>&amp; 'quotes'"


# ------------------------------------------------------------------------------ generators

def _sorted(blocks) -> tuple[RecognizedBlock, ...]:
    return tuple(sorted(blocks, key=lambda block: block.timestamp))


def _notes(count: int, end: float, start: float = 0.0, offset: int = 0) -> tuple[RecognizedBlock, ...]:
    """count blocks spread evenly between start and end, mixing equations and prose."""
    blocks = []
    for index in range(count):
        n = index + offset
        equation = n % 3 == 1
        bank = EQUATIONS if equation else PROSE
        blocks.append(
            RecognizedBlock(
                text=bank[n % len(bank)],
                kind="equation" if equation else "prose",
                timestamp=round(start + (end - start) * (index + 0.5) / count, 2),
                confidence=round(0.55 + (n * 37 % 45) / 100, 2),
            )
        )
    return tuple(blocks)


def _course_notes(length: float) -> tuple[RecognizedBlock, ...]:
    return (
        RecognizedBlock("Linear transformations preserve addition and scaling", "prose", round(length * 0.1, 2)),
        RecognizedBlock(r"T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})", "equation", round(length * 0.35, 2)),
        RecognizedBlock(r"\det(A - \lambda I) = 0", "equation", round(length * 0.7, 2)),
    )


# --------------------------------------------------------------------------------- folders

# (id, name, parent id). A parent is always listed before its children.
FOLDERS: tuple[tuple[str, str, str | None], ...] = (
    ("demo-archive", "Archive 2025", None),
    ("demo-archive-fall", "Fall", "demo-archive"),
    ("demo-archive-old", "Old recordings (do not delete)", "demo-archive-fall"),
    ("demo-cs-593", "CS_593_application_deep_learning", None),
    ("demo-week-1", "Week 1", "demo-cs-593"),
    ("demo-lab-sessions", "Lab sessions", "demo-week-1"),
    ("demo-lab-1", "Lab 1 (extra credit)", "demo-lab-sessions"),
    ("demo-lab-1-raw", "Raw recordings", "demo-lab-1"),
    ("demo-week-2", "Week 2", "demo-cs-593"),
    ("demo-midterm", "Midterm review", "demo-cs-593"),
    ("demo-edge", "Edge cases and unusual names", None),
    ("demo-edge-chars", "R&D <draft> \"notes\" 'v2'", "demo-edge"),
    ("demo-edge-unicode", "S\u00e9minaire d'\u00e9t\u00e9 \u65e5\u672c\u8a9e \u2713", "demo-edge"),
    ("demo-edge-long", LONG_FOLDER_NAME, "demo-edge"),
    ("demo-edge-duplicates", "Duplicate names", "demo-edge"),
    ("demo-edge-empty", "Empty folder", "demo-edge"),
    ("demo-level-1", "Level 1", "demo-edge"),
    ("demo-level-2", "Level 2", "demo-level-1"),
    ("demo-level-3", "Level 3", "demo-level-2"),
    ("demo-level-4", "Level 4", "demo-level-3"),
    ("demo-level-5", "Level 5", "demo-level-4"),
    ("demo-level-6", "Level 6", "demo-level-5"),
    ("demo-empty-top", "Empty top-level folder", None),
    ("demo-ma-375", "MA_375_elem_linear_algebra", None),
    ("demo-ma-lectures", "Lectures", "demo-ma-375"),
    ("demo-ma-ch1", "Chapter 1: Systems & matrices", "demo-ma-lectures"),
    ("demo-ma-ch2", "Chapter 2: Vector spaces", "demo-ma-lectures"),
    ("demo-ma-ch3", "Chapter 3: Eigenvalues", "demo-ma-lectures"),
    ("demo-ma-office", "Office hours", "demo-ma-375"),
    ("demo-misc", "misc notes (lowercase sorts among capitals)", None),
    ("demo-phys-201", "PHYS_201_classical_mechanics", None),
    ("demo-phys-weekly", "Weekly lectures", "demo-phys-201"),
    ("demo-week-1-root", "Week 1", None),  # same name as a nested folder, different parent
)


# ----------------------------------------------------------------------------- video specs

@dataclass(frozen=True)
class DemoJob:
    state: JobState
    created: datetime
    started: datetime | None = None
    finished: datetime | None = None
    error: str | None = None
    progress: float = 0.0
    scheduled_for: datetime | None = None


@dataclass(frozen=True)
class DemoVideo:
    """One video. blocks=None means never processed; blocks=() means processed with nothing found."""

    key: str
    name: str
    folder: str | None
    length: float
    uploaded: datetime
    blocks: tuple[RecognizedBlock, ...] | None = None
    processed: datetime | None = None
    edited: datetime | None = None
    exported: datetime | None = None
    favorite: bool = False
    trim_start: float = 0.0
    trim_end: float | None = None
    filename: str = ""
    jobs: tuple[DemoJob, ...] = ()

    def __post_init__(self) -> None:
        if (self.blocks is None) != (self.processed is None):
            raise ValueError(f"{self.key}: blocks and processed must be given together")
        if self.blocks is None and (self.edited or self.exported):
            raise ValueError(f"{self.key}: an unprocessed video cannot be edited or exported")


def _catalog(now: datetime) -> list[DemoVideo]:
    def ago(**delta: float) -> datetime:
        return now - timedelta(**delta)

    def at(days: int, hour: int, minute: int) -> datetime:
        """A wall-clock time on a day in the past (three or more days back stays clear of Recents)."""
        return (now - timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)

    def after(moment: datetime, **delta: float) -> datetime:
        return min(moment + timedelta(**delta), now - timedelta(minutes=1))

    def done(key, name, folder, length, uploaded, blocks, *, minutes=18, **extra) -> DemoVideo:
        """A processed video; processing finished `minutes` after the upload."""
        return DemoVideo(key, name, folder, length, uploaded, blocks=blocks, processed=after(uploaded, minutes=minutes), **extra)

    def waiting(key, name, folder, length, uploaded, *jobs, **extra) -> DemoVideo:
        return DemoVideo(key, name, folder, length, uploaded, jobs=jobs, **extra)

    clips: list[DemoVideo] = []

    # --- The original sample set (ids and content kept stable) --------------------------------
    clips += [
        waiting("3", "3", "demo-cs-593", 61, ago(days=20)),
        done("2", "Video 2 with a medium length", "demo-cs-593", 86 * 60 + 44, ago(days=130), _course_notes(86 * 60 + 44)),
        done("1", "Video 1", "demo-cs-593", 95 * 60 + 16, ago(days=138), _course_notes(95 * 60 + 16),
             favorite=True, exported=ago(days=120)),
        done("intro", "Introduction to the course", "demo-week-1", 48 * 60 + 5, ago(days=140), _notes(9, 48 * 60 + 5)),
        done("6", "Video 6", "demo-ma-375", 90 * 60 + 3, ago(hours=3), _course_notes(90 * 60 + 3), favorite=True),
        done("5", "Video 5 with a longer title which exceeds the width of the paragraph element",
             "demo-ma-375", 75 * 60 + 53, ago(days=5), _notes(14, 75 * 60 + 53)),
        waiting("4", "Video 4", "demo-ma-375", 12 * 60 + 30, ago(days=5, hours=2),
                DemoJob(JobState.FAILED, ago(days=5), ago(days=5), ago(days=5),
                        "Demo failure: the recognition engine stopped at frame 1042.", 0.4)),
    ]

    # --- Dashboard variety: the eight most recent items, newest first --------------------------
    clips += [
        DemoVideo("just-processed", "Just finished: eigenvalues recap", "demo-ma-ch3", 54 * 60 + 20, ago(minutes=25),
                  blocks=_notes(17, 54 * 60 + 20), processed=ago(minutes=4)),
        waiting("scheduled-tomorrow", "Scheduled for tomorrow morning", "demo-week-2", 72 * 60, ago(minutes=30),
                DemoJob(JobState.SCHEDULED, ago(minutes=29),
                        scheduled_for=(now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0))),
        waiting("failed-long", "Failed with a very long error message", "demo-midterm", 2 * 3600 + 5 * 60 + 30, ago(minutes=70),
                DemoJob(JobState.FAILED, ago(minutes=69), ago(minutes=68), ago(minutes=62), LONG_ERROR, 0.62)),
        DemoVideo("tiny", "Two-second clip", "demo-edge", 2.0, ago(hours=2),
                  blocks=(RecognizedBlock("Hello", "prose", 0.0, 0.99),), processed=ago(minutes=100)),
        waiting("new-unprocessed", "Just uploaded, not processed yet", "demo-week-2", 33 * 60 + 45, ago(hours=4)),
        waiting("cancelled", "Cancelled before it started", "demo-midterm", 45 * 60, ago(hours=10),
                DemoJob(JobState.CANCELLED, ago(hours=9, minutes=50), finished=ago(hours=9, minutes=40))),
        DemoVideo("recent-23h50", "Processed 23 hours 50 minutes ago (inside the 24 hour tile)", "demo-week-2",
                  3661, ago(hours=25), blocks=_notes(8, 3661), processed=ago(hours=23, minutes=50)),
    ]

    # --- Just outside the 24 hour window, and dates that differ per field -----------------------
    clips += [
        DemoVideo("recent-24h10", "Processed 24 hours 10 minutes ago (outside the tile)", "demo-ma-ch1", 41 * 60,
                  ago(hours=26), blocks=_notes(6, 41 * 60), processed=ago(hours=24, minutes=10)),
        done("edited-after-export", "Edited after it was exported (the export is stale)", "demo-ma-ch1", 58 * 60 + 40,
             ago(days=30), _notes(11, 58 * 60 + 40), exported=ago(days=29), edited=ago(days=2)),
        done("exported-never-edited", "Exported, never edited", "demo-ma-ch2", 47 * 60, ago(days=21), _notes(9, 47 * 60),
             exported=ago(days=20)),
        done("edited-never-exported", "Edited, never exported", "demo-ma-ch2", 51 * 60 + 10, ago(days=19),
             _notes(10, 51 * 60 + 10), edited=ago(days=18)),
        done("all-dates", "Uploaded, processed, edited, and exported on different days", "demo-ma-ch2", 63 * 60,
             ago(days=33), _notes(15, 63 * 60), minutes=75, edited=ago(days=26), exported=ago(days=24)),
        DemoVideo("late-processing", "Waited two weeks for processing", "demo-ma-ch3", 66 * 60, ago(days=14),
                  blocks=_notes(12, 66 * 60), processed=ago(days=2), exported=ago(days=1)),
        DemoVideo("old-year", "Uploaded over a year ago", "demo-archive-old", 70 * 60, ago(days=400),
                  blocks=_notes(10, 70 * 60), processed=ago(days=399), exported=ago(days=398)),
        waiting("old-unprocessed", "Old and never processed", "demo-archive-old", 55 * 60, ago(days=380)),
        done("old-fall-1", "Fall lecture, archived", "demo-archive-fall", 52 * 60, ago(days=300), _notes(7, 52 * 60)),
        done("old-2025-1", "Spring lecture, archived", "demo-archive", 49 * 60 + 30, ago(days=330), _notes(5, 49 * 60 + 30)),
    ]

    # --- Status history -------------------------------------------------------------------------
    failed_once = ago(days=6)
    clips += [
        DemoVideo("failed-then-ok", "Failed once, then processed", "demo-ma-ch3", 60 * 60, ago(days=7),
                  blocks=_notes(13, 60 * 60), processed=ago(days=5, hours=19),
                  jobs=(DemoJob(JobState.FAILED, failed_once, failed_once, failed_once + timedelta(minutes=9),
                                "Demo failure: out of memory while reading the keyframes.", 0.55),
                        DemoJob(JobState.COMPLETED, ago(days=5, hours=20), ago(days=5, hours=20),
                                ago(days=5, hours=19), progress=1.0))),
        DemoVideo("ok-then-failed", "Processed, then a re-run failed (text kept, status Failed)", "demo-ma-ch3", 57 * 60,
                  ago(days=10), blocks=_notes(9, 57 * 60), processed=ago(days=9),
                  jobs=(DemoJob(JobState.COMPLETED, ago(days=9, minutes=20), ago(days=9, minutes=19), ago(days=9), progress=1.0),
                        DemoJob(JobState.FAILED, ago(days=2), ago(days=2), ago(days=2) + timedelta(minutes=1),
                                "Demo failure: the re-run could not open the video file.", 0.0))),
    ]

    # --- Clock times that exercise 12-hour formatting ------------------------------------------
    clips += [
        done("midnight", "Uploaded at 12:05 AM", "demo-edge", 44 * 60, at(3, 0, 5), _notes(6, 44 * 60), minutes=8),
        done("noon", "Uploaded at 12:00 PM", "demo-edge", 46 * 60, at(3, 12, 0), _notes(7, 46 * 60), minutes=8),
        done("late-night", "Uploaded at 11:59 PM (processed after midnight)", "demo-edge", 43 * 60, at(3, 23, 59),
             _notes(5, 43 * 60), minutes=8),
    ]

    # --- Length and content extremes -----------------------------------------------------------
    clips += [
        done("blank", "Blank board: nothing recognized", "demo-edge", 25 * 60, ago(days=11), (), minutes=9),
        done("one-block", "One recognized block", "demo-edge", 40 * 60, ago(days=10),
             (RecognizedBlock("The only thing on the board: F = ma", "equation", 1312.0, 0.93),)),
        done("short7", "Seven second test", "demo-edge", 7, ago(days=9), _notes(1, 7), minutes=1),
        done("just-under-hour", "59:59 length boundary", "demo-edge", 3599, ago(days=8), _notes(14, 3599)),
        done("exactly-hour", "1:00:00 length boundary (blocks at the first and last instant)", "demo-edge", 3600,
             ago(days=8, hours=2),
             (RecognizedBlock("Written before the lecture started", "prose", 0.0, 0.71),
              RecognizedBlock("Final line on the board", "equation", 3600.0, 0.88))),
        done("long-text", "Very long recognized text", "demo-edge", 20 * 60, ago(days=12),
             _sorted((RecognizedBlock(LONG_PROSE, "prose", 120.0, 0.82),
                      RecognizedBlock(LONG_EQUATION, "equation", 480.0, 0.64), *_notes(3, 1200, offset=2)))),
        done("same-time", "Blocks that share a timestamp", "demo-edge", 300, ago(days=13),
             (RecognizedBlock("First of three at 0:30", "prose", 30.0), RecognizedBlock(r"x^2 + y^2 = r^2", "equation", 30.0),
              RecognizedBlock("Third at 0:30", "prose", 30.0), RecognizedBlock("Two at 1:30", "prose", 90.0),
              RecognizedBlock(r"\pi r^2", "equation", 90.0), RecognizedBlock("Alone at 2:30.75", "prose", 150.75))),
        done("fractional", "Fractional length and timestamps", "demo-edge", 1805.9, ago(days=14),
             (RecognizedBlock("At 0.4 seconds", "prose", 0.4), RecognizedBlock("At 59.999 seconds", "prose", 59.999),
              RecognizedBlock("At 60.0 seconds", "prose", 60.0), RecognizedBlock("At 60.001 seconds", "prose", 60.001),
              RecognizedBlock("At the very end, 1805.9", "prose", 1805.9))),
        done("sparse", "Sparse: three blocks in two hours", "demo-lab-1-raw", 7200, ago(days=16),
             (RecognizedBlock(PROSE[0], "prose", 2400.0), RecognizedBlock(EQUATIONS[2], "equation", 4800.0),
              RecognizedBlock("Last note, one minute from the end", "prose", 7140.0)), minutes=40),
        done("dense", "Dense: 60 blocks in ten minutes", "demo-week-2", 600, ago(days=4), _notes(60, 600)),
        done("max-150", "Maximum length: 150 minutes, 150 blocks", "demo-midterm", 9000, ago(days=6), _notes(150, 9000),
             minutes=95, favorite=True, edited=ago(days=5, hours=3), exported=ago(days=5)),
        done("midterm-1", "Midterm review session", "demo-midterm", 95 * 60, ago(days=7), _notes(20, 95 * 60)),
    ]

    # --- Trimmed videos (blocks fall inside the processed range) -------------------------------
    clips += [
        done("trim-both", "Trimmed: 10:00 to 25:30 of a 45 minute video", "demo-edge", 2700, ago(days=15),
             _notes(10, 1530, start=600), trim_start=600, trim_end=1530),
        done("trim-start-only", "Trimmed at the start only (from 2:00)", "demo-edge", 3000, ago(days=15, hours=1),
             _notes(8, 3000, start=120), trim_start=120),
        done("trim-end-only", "Trimmed at the end only (until 10:00)", "demo-edge", 3000, ago(days=15, hours=2),
             _notes(5, 600), trim_end=600),
        done("trim-tiny", "Trimmed to five seconds", "demo-edge", 5400, ago(days=15, hours=3),
             (RecognizedBlock("The one line in the clip", "prose", 2702.5),), trim_start=2700, trim_end=2705, minutes=1),
        done("lab1-raw-b", "Lab 1 raw recording, part B (trimmed)", "demo-lab-1-raw", 6252, ago(days=40, hours=1),
             _notes(30, 5400, start=300), trim_start=300, trim_end=5400, minutes=35),
    ]

    # --- Names ----------------------------------------------------------------------------------
    clips += [
        done("max-name", LONG_VIDEO_NAME, "demo-edge-long", 42 * 60, ago(days=17), _notes(4, 42 * 60)),
        done("html-name", HTML_VIDEO_NAME, "demo-edge-chars", 38 * 60, ago(days=17, hours=2), _notes(4, 38 * 60)),
        done("underscores", "Final_review_SESSION (v2)_FINAL_final", "demo-edge-chars", 39 * 60, ago(days=17, hours=3),
             _notes(3, 39 * 60)),
        done("spaces", "Spaced   out   name", "demo-edge-chars", 37 * 60, ago(days=17, hours=4), _notes(3, 37 * 60)),
        done("accents", "S\u00e9ance d'introduction \u00e0 l'alg\u00e8bre lin\u00e9aire", "demo-edge-unicode", 50 * 60,
             ago(days=18), _notes(6, 50 * 60), favorite=True),
        done("cjk", "\u7dda\u5f62\u4ee3\u6570 \u7b2c1\u56de\u8b1b\u7fa9", "demo-edge-unicode", 48 * 60, ago(days=18, hours=2),
             _notes(6, 48 * 60, offset=10)),
        done("emoji", "\U0001f393 Orientation day", "demo-edge-unicode", 30 * 60, ago(days=18, hours=3), _notes(3, 30 * 60)),
        done("dup-a", "Lecture 1", "demo-edge-duplicates", 50 * 60, ago(days=20), _notes(5, 50 * 60)),
        done("dup-b", "Lecture 1", "demo-edge-duplicates", 51 * 60, ago(days=20, hours=1), _notes(5, 51 * 60, offset=4)),
        done("dup-c", "Lecture 1", "demo-week-2", 52 * 60, ago(days=21), _notes(5, 52 * 60, offset=7)),
        done("deepest", "Deepest video: seven levels down", "demo-level-6", 18 * 60, ago(days=22), _notes(8, 18 * 60),
             favorite=True),
        done("loose-1", "Loose video at the top level", None, 33 * 60, ago(days=23), _notes(6, 33 * 60), favorite=True),
        done("loose-2", "Loose recording: 2:29:59", None, 8999, ago(days=24), _notes(25, 8999), minutes=80),
        done("misc-note", "Hallway recording", "demo-misc", 4 * 60, ago(days=25), _notes(2, 4 * 60)),
        done("week-1-root", "Week 1 overview (top-level folder with the same name)", "demo-week-1-root", 35 * 60,
             ago(days=26), _notes(5, 35 * 60)),
    ]

    # --- Folders with a few ordinary videos -----------------------------------------------------
    clips += [
        done("lab-setup", "Lab environment setup", "demo-lab-sessions", 15 * 60 + 30, ago(days=44), _notes(8, 15 * 60 + 30)),
        waiting("lab-qa", "Lab Q&A (never processed)", "demo-lab-sessions", 12 * 60, ago(days=43)),
        done("lab1-demo", "Lab 1 demonstration", "demo-lab-1", 22 * 60, ago(days=41), _notes(6, 22 * 60)),
        done("lab1-raw-a", "Lab 1 raw recording, part A", "demo-lab-1-raw", 1694, ago(days=40), _notes(18, 1694)),
        done("w2-lecture", "Week 2 lecture: backpropagation", "demo-week-2", 78 * 60 + 20, ago(days=9), _notes(22, 78 * 60 + 20)),
        done("ch1-gauss", "Gaussian elimination", "demo-ma-ch1", 55 * 60, ago(days=35), _notes(12, 55 * 60)),
        done("ch1-matrices", "Matrix multiplication", "demo-ma-ch1", 49 * 60, ago(days=34), _notes(9, 49 * 60)),
        done("ch2-span", "Span and independence", "demo-ma-ch2", 52 * 60, ago(days=28), _notes(11, 52 * 60)),
        done("ch3-diag", "Diagonalization", "demo-ma-ch3", 61 * 60, ago(days=12), _notes(14, 61 * 60)),
    ]

    clips += _physics_lectures(ago, after)
    return clips


_PHYSICS_TOPICS = (
    "Units and vectors", "Kinematics in one dimension", "Projectile motion", "Newton's laws", "Friction and drag",
    "Circular motion", "Work and energy", "Potential energy", "Momentum and impulse", "Collisions", "Rotation",
    "Torque", "Angular momentum", "Equilibrium", "Gravitation", "Orbits", "Oscillations", "Damped motion",
    "Driven oscillators", "Waves on a string", "Fluids", "Thermodynamics review", "Lagrangian mechanics", "Course review",
)
_PHYSICS_LENGTHS = (3000, 4500, 3150, 4815, 2985, 4200, 3600, 5100)


def _physics_lectures(ago, after) -> list[DemoVideo]:
    """A semester of weekly lectures: a long list that makes the folder and its page scroll."""
    lectures = []
    for number, topic in enumerate(_PHYSICS_TOPICS, start=1):
        length = _PHYSICS_LENGTHS[number % len(_PHYSICS_LENGTHS)]
        uploaded = ago(days=175 - 7 * (number - 1))
        common = dict(key=f"phys-{number:02d}", name=f"Lecture {number:02d}: {topic}", folder="demo-phys-weekly",
                      length=length, uploaded=uploaded, favorite=number in (1, 12), filename=f"phys201-lecture-{number:02d}.mp4")
        if number == 13:
            lectures.append(DemoVideo(**common, jobs=(DemoJob(
                JobState.FAILED, uploaded, uploaded, uploaded + timedelta(minutes=2),
                "Demo failure: the recording has no video track (audio only).", 0.05),)))
        elif number == 24:
            lectures.append(DemoVideo(**common))
        else:
            lectures.append(DemoVideo(
                **common, blocks=_notes(3 + (number * 7) % 28, length, offset=number),
                processed=after(uploaded, minutes=20),
                edited=uploaded + timedelta(days=1) if number % 5 == 0 else None,
                exported=uploaded + timedelta(days=2) if number % 2 == 0 else None))
    return lectures


# ------------------------------------------------------------------------------- seeding

def seed_demo_library(library: LibraryService, repo: Repository, now: datetime) -> None:
    """Fill an empty library with the sample folders, videos, and task history."""
    for folder_id, name, parent_id in FOLDERS:
        library.create_folder(name, parent_id, folder_id=folder_id)

    for clip in _catalog(now):
        video_id = f"demo-video-{clip.key}"
        created = library.add_video(
            name=clip.name,
            folder_id=clip.folder,
            length_seconds=clip.length,
            source_filename=clip.filename,
            trim_start=clip.trim_start,
            trim_end=clip.trim_end,
            video_id=video_id,
        )
        repo.save_video(replace(
            created,
            uploaded_at=clip.uploaded,
            favorite=clip.favorite,
            ocr_blocks=_sorted(clip.blocks) if clip.blocks is not None else (),
            processed_at=clip.processed,
            edited_at=clip.edited,
            exported_at=clip.exported,
        ))
        for index, job in enumerate(clip.jobs, start=1):
            repo.save_job(ProcessingJob(
                id=f"demo-job-{clip.key}-{index}",
                video_id=video_id,
                state=job.state,
                created_at=job.created,
                estimated_seconds=max(6.0, min(created.clip_seconds * 0.02, 120.0)),
                scheduled_for=job.scheduled_for,
                started_at=job.started,
                finished_at=job.finished,
                progress=job.progress,
                error=job.error,
            ))