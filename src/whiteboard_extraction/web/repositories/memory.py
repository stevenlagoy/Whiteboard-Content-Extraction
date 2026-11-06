"""In-memory repository. State lasts for the life of the process."""

from __future__ import annotations

import threading

from ..domain.models import Folder, ProcessingJob, Video


class InMemoryRepository:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._folders: dict[str, Folder] = {}
        self._videos: dict[str, Video] = {}
        self._jobs: dict[str, ProcessingJob] = {}

    def folders(self) -> list[Folder]:
        with self.lock:
            return list(self._folders.values())

    def get_folder(self, folder_id: str) -> Folder | None:
        with self.lock:
            return self._folders.get(folder_id)

    def save_folder(self, folder: Folder) -> None:
        with self.lock:
            self._folders[folder.id] = folder

    def delete_folder(self, folder_id: str) -> None:
        with self.lock:
            self._folders.pop(folder_id, None)

    def videos(self) -> list[Video]:
        with self.lock:
            return list(self._videos.values())

    def get_video(self, video_id: str) -> Video | None:
        with self.lock:
            return self._videos.get(video_id)

    def save_video(self, video: Video) -> None:
        with self.lock:
            self._videos[video.id] = video

    def save_videos(self, videos: list[Video]) -> None:
        with self.lock:
            for video in videos:
                self._videos[video.id] = video

    def delete_video(self, video_id: str) -> None:
        with self.lock:
            self._videos.pop(video_id, None)

    def jobs(self) -> list[ProcessingJob]:
        with self.lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> ProcessingJob | None:
        with self.lock:
            return self._jobs.get(job_id)

    def save_job(self, job: ProcessingJob) -> None:
        with self.lock:
            self._jobs[job.id] = job

    def delete_job(self, job_id: str) -> None:
        with self.lock:
            self._jobs.pop(job_id, None)
