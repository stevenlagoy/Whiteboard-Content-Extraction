"""Folders and videos: creating, organizing, editing, and deleting them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from whiteboard_extraction.recognition.base import RecognizedBlock

from ..domain.errors import ConflictError, NotFoundError, ValidationError
from ..domain.models import Folder, JobState, Video, new_id
from ..repositories.base import Repository
from ..timecodes import format_timecode

Clock = Callable[[], datetime]
Direction = Literal["up", "down"]

NAME_LIMIT = 120
BLOCK_KINDS = ("equation", "prose")
PATH_SEPARATOR = " › "
ROOT_LABEL = "My files"
_TIME_TOLERANCE = 0.5  # probed durations are floats; allow rounding slack


@dataclass(frozen=True)
class FolderNode:
    folder: Folder
    children: tuple[FolderNode, ...]
    videos: tuple[Video, ...]


@dataclass(frozen=True)
class LibraryTree:
    folders: tuple[FolderNode, ...]
    videos: tuple[Video, ...]  # videos stored at the top level


@dataclass(frozen=True)
class FolderChoice:
    id: str | None
    label: str
    depth: int


def clean_name(raw: str | None, what: str) -> str:
    name = (raw or "").strip()
    if not name:
        raise ValidationError(f"{what} cannot be empty.")
    if len(name) > NAME_LIMIT:
        raise ValidationError(f"{what} must be {NAME_LIMIT} characters or fewer.")
    return name


def _folder_sort_key(folder: Folder) -> tuple[str, str]:
    return (folder.name.casefold(), folder.id)


def _video_sort_key(video: Video) -> tuple[int, datetime]:
    return (video.position, video.uploaded_at)


class LibraryService:
    def __init__(self, repo: Repository, clock: Clock) -> None:
        self._repo = repo
        self._clock = clock

    # ------------------------------------------------------------ folders

    def get_folder(self, folder_id: str) -> Folder:
        folder = self._repo.get_folder(folder_id)
        if folder is None:
            raise NotFoundError("That folder no longer exists.")
        return folder

    def create_folder(self, name: str, parent_id: str | None = None, *, folder_id: str | None = None) -> Folder:
        clean = clean_name(name, "Folder name")
        with self._repo.lock:
            if parent_id is not None:
                self.get_folder(parent_id)
            self._ensure_unique_folder_name(clean, parent_id)
            folder = Folder(id=folder_id or new_id(), name=clean, parent_id=parent_id)
            self._repo.save_folder(folder)
            return folder

    def rename_folder(self, folder_id: str, name: str) -> Folder:
        clean = clean_name(name, "Folder name")
        with self._repo.lock:
            folder = self.get_folder(folder_id)
            self._ensure_unique_folder_name(clean, folder.parent_id, ignore_id=folder.id)
            renamed = replace(folder, name=clean)
            self._repo.save_folder(renamed)
            return renamed

    def delete_folder(self, folder_id: str) -> Folder:
        with self._repo.lock:
            folder = self.get_folder(folder_id)
            has_subfolders = any(f.parent_id == folder_id for f in self._repo.folders())
            has_videos = any(v.folder_id == folder_id for v in self._repo.videos())
            if has_subfolders or has_videos:
                raise ConflictError("Move or delete everything inside this folder first.")
            self._repo.delete_folder(folder_id)
            return folder

    def folder_path(self, folder_id: str | None) -> list[Folder]:
        """Folders from the top level down to folder_id (empty for the top level)."""
        path: list[Folder] = []
        seen: set[str] = set()
        current = folder_id
        while current is not None and current not in seen:
            seen.add(current)
            folder = self.get_folder(current)
            path.append(folder)
            current = folder.parent_id
        path.reverse()
        return path

    def folder_label(self, folder_id: str | None) -> str:
        path = self.folder_path(folder_id)
        return PATH_SEPARATOR.join(f.name for f in path) if path else ROOT_LABEL

    def child_folders(self, parent_id: str | None) -> list[Folder]:
        return sorted(
            (f for f in self._repo.folders() if f.parent_id == parent_id),
            key=_folder_sort_key,
        )

    def folder_tree(self) -> LibraryTree:
        with self._repo.lock:
            folders = self._repo.folders()
            videos = self._repo.videos()

        def build(parent_id: str | None) -> tuple[FolderNode, ...]:
            siblings = sorted((f for f in folders if f.parent_id == parent_id), key=_folder_sort_key)
            return tuple(
                FolderNode(
                    folder=f,
                    children=build(f.id),
                    videos=tuple(sorted((v for v in videos if v.folder_id == f.id), key=_video_sort_key)),
                )
                for f in siblings
            )

        top_level = sorted((v for v in videos if v.folder_id is None), key=_video_sort_key)
        return LibraryTree(folders=build(None), videos=tuple(top_level))

    def folder_choices(self) -> list[FolderChoice]:
        choices = [FolderChoice(None, ROOT_LABEL, 0)]

        def walk(parent_id: str | None, prefix: list[str]) -> None:
            for folder in self.child_folders(parent_id):
                names = [*prefix, folder.name]
                choices.append(FolderChoice(folder.id, PATH_SEPARATOR.join(names), len(names)))
                walk(folder.id, names)

        walk(None, [])
        return choices

    def _ensure_unique_folder_name(self, name: str, parent_id: str | None, ignore_id: str | None = None) -> None:
        wanted = name.casefold()
        for folder in self._repo.folders():
            if folder.parent_id == parent_id and folder.id != ignore_id and folder.name.casefold() == wanted:
                raise ConflictError(f'A folder named "{folder.name}" already exists here.')

    # ------------------------------------------------------------- videos

    def find_video(self, video_id: str) -> Video | None:
        return self._repo.get_video(video_id)

    def get_video(self, video_id: str) -> Video:
        video = self._repo.get_video(video_id)
        if video is None:
            raise NotFoundError("That video no longer exists.")
        return video

    def all_videos(self) -> list[Video]:
        return self._repo.videos()

    def videos_in(self, folder_id: str | None) -> list[Video]:
        return sorted((v for v in self._repo.videos() if v.folder_id == folder_id), key=_video_sort_key)

    def recent(self, limit: int) -> list[Video]:
        newest_first = sorted(self._repo.videos(), key=lambda v: v.last_activity_at, reverse=True)
        return newest_first[:limit]

    def favorites(self) -> list[Video]:
        return sorted((v for v in self._repo.videos() if v.favorite), key=lambda v: v.name.casefold())

    def add_video(
        self,
        *,
        name: str,
        folder_id: str | None,
        length_seconds: float,
        source_filename: str = "",
        media_file: str | None = None,
        has_thumbnail: bool = False,
        trim_start: float = 0.0,
        trim_end: float | None = None,
        video_id: str | None = None,
    ) -> Video:
        clean = clean_name(name, "Video name")
        self._check_trim(length_seconds, trim_start, trim_end)
        with self._repo.lock:
            if folder_id is not None:
                self.get_folder(folder_id)
            video = Video(
                id=video_id or new_id(),
                name=clean,
                folder_id=folder_id,
                length_seconds=length_seconds,
                uploaded_at=self._clock(),
                source_filename=source_filename,
                media_file=media_file,
                has_thumbnail=has_thumbnail,
                position=self._next_position(folder_id),
                trim_start=trim_start,
                trim_end=trim_end,
            )
            self._repo.save_video(video)
            return video

    def rename_video(self, video_id: str, name: str) -> Video:
        clean = clean_name(name, "Video name")
        with self._repo.lock:
            video = replace(self.get_video(video_id), name=clean, edited_at=self._clock())
            self._repo.save_video(video)
            return video

    def set_favorite(self, video_id: str, favorite: bool) -> Video:
        with self._repo.lock:
            video = replace(self.get_video(video_id), favorite=favorite)
            self._repo.save_video(video)
            return video

    def move_video(self, video_id: str, folder_id: str | None, before_id: str | None = None) -> Video:
        """Move a video into folder_id, placing it before before_id (or last)."""
        with self._repo.lock:
            video = self.get_video(video_id)
            if folder_id is not None:
                self.get_folder(folder_id)
            if before_id == video_id:
                return video
            siblings = [v for v in self.videos_in(folder_id) if v.id != video_id]
            index = len(siblings)
            if before_id is not None:
                ids = [v.id for v in siblings]
                if before_id not in ids:
                    raise ValidationError("The video to place it before is not in that folder.")
                index = ids.index(before_id)
            moved = replace(video, folder_id=folder_id)
            siblings.insert(index, moved)
            self._repo.save_videos([replace(v, position=i) for i, v in enumerate(siblings)])
            return self.get_video(video_id)

    def shift_video(self, video_id: str, direction: Direction) -> Video:
        """Swap a video with its neighbour inside its folder."""
        if direction not in ("up", "down"):
            raise ValidationError("Direction must be up or down.")
        with self._repo.lock:
            video = self.get_video(video_id)
            ordered = self.videos_in(video.folder_id)
            index = next(i for i, v in enumerate(ordered) if v.id == video_id)
            target = index - 1 if direction == "up" else index + 1
            if 0 <= target < len(ordered):
                ordered[index], ordered[target] = ordered[target], ordered[index]
            self._repo.save_videos([replace(v, position=i) for i, v in enumerate(ordered)])
            return self.get_video(video_id)

    def delete_video(self, video_id: str) -> Video:
        with self._repo.lock:
            video = self.get_video(video_id)
            own_jobs = [j for j in self._repo.jobs() if j.video_id == video_id]
            if any(j.state is JobState.RUNNING for j in own_jobs):
                raise ConflictError("Cancel this video's running task before deleting it.")
            for job in own_jobs:
                self._repo.delete_job(job.id)
            self._repo.delete_video(video_id)
            return video

    def update_ocr(self, video_id: str, blocks: Sequence[RecognizedBlock]) -> Video:
        with self._repo.lock:
            video = self.get_video(video_id)
            cleaned = [self._clean_block(block, video) for block in blocks]
            edited = replace(video, ocr_blocks=self._ordered(cleaned), edited_at=self._clock())
            self._repo.save_video(edited)
            return edited

    def record_processing_result(self, video_id: str, blocks: Sequence[RecognizedBlock]) -> Video | None:
        with self._repo.lock:
            video = self.find_video(video_id)
            if video is None:
                return None
            processed = replace(video, ocr_blocks=self._ordered(blocks), processed_at=self._clock())
            self._repo.save_video(processed)
            return processed

    def record_export(self, video_id: str) -> Video:
        with self._repo.lock:
            video = replace(self.get_video(video_id), exported_at=self._clock())
            self._repo.save_video(video)
            return video

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _ordered(blocks: Sequence[RecognizedBlock]) -> tuple[RecognizedBlock, ...]:
        return tuple(sorted(blocks, key=lambda b: b.timestamp))

    @staticmethod
    def _clean_block(block: RecognizedBlock, video: Video) -> RecognizedBlock:
        text = block.text.strip()
        if not text:
            raise ValidationError("A text block cannot be empty.")
        if block.kind not in BLOCK_KINDS:
            raise ValidationError("A text block must be an equation or prose.")
        if not 0 <= block.timestamp <= video.length_seconds + _TIME_TOLERANCE:
            raise ValidationError(
                f"The timestamp {format_timecode(block.timestamp)} is outside the video "
                f"(length {format_timecode(video.length_seconds)})."
            )
        return RecognizedBlock(text=text, kind=block.kind, timestamp=block.timestamp, confidence=block.confidence)

    def _next_position(self, folder_id: str | None) -> int:
        positions = [v.position for v in self._repo.videos() if v.folder_id == folder_id]
        return max(positions, default=-1) + 1

    @staticmethod
    def _check_trim(length: float, start: float, end: float | None) -> None:
        if length <= 0:
            raise ValidationError("The video has no readable length.")
        if start < 0:
            raise ValidationError("The trim start cannot be negative.")
        if start >= length:
            raise ValidationError(f"The trim start must be before the end of the video ({format_timecode(length)}).")
        if end is not None:
            if end <= start:
                raise ValidationError("The trim end must be after the trim start.")
            if end > length + _TIME_TOLERANCE:
                raise ValidationError(f"The trim end is past the end of the video ({format_timecode(length)}).")
