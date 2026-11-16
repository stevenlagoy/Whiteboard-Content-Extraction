# Whiteboard Content Extraction: web app

A Flask application for uploading lecture videos, watching them get processed, editing the recognized text, and exporting Word and PDF documents. It runs from a computer or a phone browser and needs no command line after it starts.

The page and layout design is in [`docs/WEB-UI-DESIGN.md`](../../../docs/WEB-UI-DESIGN.md).

## Requirements

- Python 3.10 or newer
- Node.js 20 or newer (only to run the JavaScript tests)
- No other system software. Video reading uses OpenCV from `pip`; the default processing engine is simulated, so no OCR program, GPU, or API key is needed.

## Install

Run from the repository root.

**Linux or macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/whiteboard_extraction/web/webapp.txt
pip install -e .
```

**Windows (cmd)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r src\whiteboard_extraction\web\webapp.txt
pip install -e .
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r src\whiteboard_extraction\web\webapp.txt
pip install -e .
```

## Run

With the virtual environment active:

```bash
python -m whiteboard_extraction.web.app
```

Open <http://127.0.0.1:5000>. This starts Flask's development server with automatic reload. Stop it with `Ctrl+C`.

Other ways to start the same app:

```bash
# Flask CLI
flask --app whiteboard_extraction.web.app:create_app run --port 5000

# Production-style server (waitress), reachable from other devices on the network
waitress-serve --call --listen=0.0.0.0:5000 whiteboard_extraction.web.app:create_app
```

The first start shows sample folders and videos so every screen has content. To start with an empty library, set `WCE_SEED_DEMO=0` (see below).

### Settings

Settings are environment variables, read when the app starts.

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPLOAD_ROOT` | `instance` | Folder for uploaded videos (`uploads/`) and thumbnails (`thumbnails/`). |
| `SECRET_KEY` | random per start | Signs the session cookie that carries confirmation messages. Set a fixed value to keep sessions across restarts. |
| `WCE_ENGINE` | `simulated` | Processing engine: `simulated` or `pipeline`. |
| `WCE_RECOGNIZER` | `ocr` | Recognizer used by the `pipeline` engine: `ocr`, `llm`, or `cnn`. |
| `WCE_LLM_MODEL` | empty | Model name passed to the `llm` recognizer. |
| `WCE_CNN_CHECKPOINT` | empty | Checkpoint path passed to the `cnn` recognizer. |
| `WCE_SEED_DEMO` | `1` | `0` starts with an empty library. |
| `WCE_MAX_UPLOAD_GB` | `4` | Largest accepted upload, in gigabytes. |
| `WCE_MAX_CONCURRENT_JOBS` | `1` | Videos processed at the same time. |

Setting a variable:

```bash
# Linux or macOS
WCE_SEED_DEMO=0 python -m whiteboard_extraction.web.app
```
```cmd
:: Windows cmd
set WCE_SEED_DEMO=0
python -m whiteboard_extraction.web.app
```
```powershell
# Windows PowerShell
$env:WCE_SEED_DEMO = "0"
python -m whiteboard_extraction.web.app
```

### Engines

- `simulated` (default): reports steady progress for a time proportional to the clip length (6 seconds to 2 minutes) and returns sample equations and prose. Use it to try the whole interface.
- `pipeline`: runs the repository's stages in order: frame sampling (`video/sampling.py`), board detection and cropping (`video/segmentation.py`), keyframes (`video/keyframes.py`), the selected recognizer (`recognition/`), and document assembly (`reconstruction/document.py`). A stage that raises an error marks the task as failed and shows the message on the video page.

## What is stored where

- Folders, videos, recognized text, and tasks are held in memory and reset when the server restarts.
- Uploaded videos and thumbnails are written under `UPLOAD_ROOT` and are not removed on restart. Deleting a video in the app removes its files.
- Exports are generated on demand and are not kept on disk.

## Pages

| Page | URL | What it shows |
| --- | --- | --- |
| Dashboard | `/` | Status counts, Recents, Favorites, the file tree, the task panel. |
| Folder | `/folders/<folder_id>` | A folder's subfolders and ordered videos; upload here, new subfolder, rename, delete. |
| Video | `/videos/<video_id>` | Player, editable recognized text, dates, rename, move, retry, export, delete. |
| Upload and process | `/upload` (optional `?folder=<folder_id>`) | Choose a file, trim, name and place it, start now or schedule. |

Sample content uses fixed ids so the examples below work as written. Folders: `demo-cs-593`, `demo-week-1` (inside `demo-cs-593`), `demo-ma-375`. Videos: `demo-video-1`, `demo-video-2`, `demo-video-3`, `demo-video-4`, `demo-video-5`, `demo-video-6`, `demo-video-intro`. `demo-video-4` shows a failed task and `demo-video-3` is unprocessed.

## Endpoints

Pages and files (GET):

| Path | Returns |
| --- | --- |
| `/` | Dashboard page |
| `/folders/<folder_id>` | Folder page (404 if missing) |
| `/videos/<video_id>` | Video page (404 if missing) |
| `/upload` | Upload page |
| `/tasks/panel` | HTML fragment with the current tasks (used by the page every 1 to 5 seconds) |
| `/videos/<video_id>/media` | The stored video, with range requests for seeking |
| `/videos/<video_id>/thumbnail` | The thumbnail image, or the placeholder |
| `/videos/<video_id>/export/docx` | Word download of the recognized text |
| `/videos/<video_id>/export/pdf` | PDF download of the recognized text |
| `/healthz` | `{"status": "ok"}` |

Changes (POST). Each form submission answers with a 303 redirect to the `next` form field (a path on this site; default `/`) and a confirmation or error message shown on the next page. A failure never returns a server error page.

| Path | Fields | Effect |
| --- | --- | --- |
| `/folders` | `name`, optional `parent_id` | Create a folder |
| `/folders/<folder_id>/rename` | `name` | Rename a folder |
| `/folders/<folder_id>/delete` | none | Delete an empty folder |
| `/videos/<video_id>/favorite` | `favorite` = `true` or `false` | Set the favorite state |
| `/videos/<video_id>/rename` | `name` | Rename a video |
| `/videos/<video_id>/move` | `folder_id` (empty for the top level), optional `before_id` | Move a video, optionally before another |
| `/videos/<video_id>/shift` | `direction` = `up` or `down` | Move one place within its folder |
| `/videos/<video_id>/delete` | none | Delete a video and its files (refused while it is running) |
| `/videos/<video_id>/ocr` | repeated `timestamp`, `kind`, `text`; repeated `remove` (row numbers) | Save the edited recognized text |
| `/videos/<video_id>/process` | none | Queue a video for processing |
| `/jobs` | `video` (file), `name`, `folder_id`, `trim_start`, `trim_end`, `start_mode` = `now` or `later`, `start_at` (`YYYY-MM-DDTHH:MM`) | Upload and queue a video. With `Accept: application/json` it answers `{"redirect": "/"}` or `{"error": "..."}` (status 400). |
| `/jobs/<job_id>/cancel` | none | Cancel a waiting task, or ask a running one to stop |

Examples (the app must be running; add `-i` to see headers):

```bash
curl http://127.0.0.1:5000/healthz
curl -X POST -d "name=Week 2" http://127.0.0.1:5000/folders
curl -X POST -d "name=Labs&parent_id=demo-cs-593" http://127.0.0.1:5000/folders
curl -X POST -d "folder_id=demo-ma-375" http://127.0.0.1:5000/videos/demo-video-3/move
curl -X POST -d "favorite=true" http://127.0.0.1:5000/videos/demo-video-3/favorite
curl -X POST -d "name=Renamed lecture" http://127.0.0.1:5000/videos/demo-video-3/rename
curl -X POST -d "direction=up" http://127.0.0.1:5000/videos/demo-video-2/shift
curl -X POST http://127.0.0.1:5000/videos/demo-video-3/process
curl -s http://127.0.0.1:5000/tasks/panel
curl -o notes.docx http://127.0.0.1:5000/videos/demo-video-1/export/docx
curl -o notes.pdf  http://127.0.0.1:5000/videos/demo-video-1/export/pdf
curl -H "Accept: application/json" \
     -F video=@lecture.mp4 -F name="Week 1" -F trim_start=0:10 -F trim_end=1:00 -F start_mode=now \
     http://127.0.0.1:5000/jobs
```

## Using the app

1. **Upload**: select the large plus button (bottom right), a plus in the sidebar, or a folder's plus. Choose a file, optionally set the start and end of the part to process (type times or press "Use current position" while previewing), choose a name and folder, and choose when to start. Select "Add to queue".
2. **Watch**: the task panel shows each task with the video length, elapsed time, time remaining, and a progress bar. The dashboard tiles and each row's status badge update when a task changes state.
3. **Review**: open a video, play it, and edit the recognized text. Select the play button beside a timestamp to jump the video there.
4. **Export**: select "Export Word document" or "Export PDF" on the video page.
5. **Organize**: drag videos onto folders or onto other videos, or use the arrow buttons on a folder page. The star marks favorites. Create folders with the folder-plus button above the file tree.

The in-page player uses the browser's decoder. MP4 (H.264) and WebM play in current browsers; MKV, AVI, and some MOV files upload and process but may not preview.

## Tests

With the virtual environment active, from the repository root:

```bash
pytest tests/web                # Python tests (services, engines, routes, pages, exports, media)
ruff check src/whiteboard_extraction/web tests/web
```

JavaScript tests (jsdom, Node 20 or newer):

```bash
cd tests/web/js
npm install
npm test
```

The Python tests use fakes for the processing engine, video reader, clock, and executor, so they run in a few seconds without threads, video files, or a browser. `tests/web/test_media.py` generates a small MP4 with OpenCV and is skipped if OpenCV cannot write one.

## Layout

```
src/whiteboard_extraction/web/
├── app.py                 create_app: builds and wires everything
├── config.py              AppConfig and environment variables
├── container.py           The services shared by requests
├── presentation.py        Template filters and helpers
├── storage.py             Where uploads and thumbnails live on disk
├── timecodes.py           Seconds <-> H:MM:SS
├── demo_data.py           Sample library
├── domain/                Entities (models.py) and errors
├── repositories/          Storage protocol and the in-memory implementation
├── services/              library, jobs, status, ingest, exports, media
├── processing/            Engine protocol, simulated and pipeline engines, registry, job runner
├── routes/                pages (GET), actions (POST), files, forms, support
├── templates/             Page templates and components/ (rows, tree, icons, task panel)
└── static/
    ├── scss/              Styles, compiled to static/css/packed.css when a page loads
    └── js/                ES modules for in-place updates, drag and drop, tasks, upload
tests/web/                 Python tests; js/ holds the JavaScript tests
docs/WEB-UI-DESIGN.md      Layouts, interaction, principles, diagrams
```

Style changes: edit the files in `static/scss/`. The generated `static/css/packed.css` is rebuilt automatically and is not committed.

## Swapping parts

Every external part is a constructor argument of `create_app`, so tests and alternative setups replace one piece without touching the rest.

```python
from whiteboard_extraction.web.app import create_app
from whiteboard_extraction.web.config import AppConfig

app = create_app(
    AppConfig(engine="simulated"),
    engine=MyEngine(),            # name, estimate_seconds(clip_seconds), process(request, reporter)
    media_probe=MyProbe(),        # probe(path), write_thumbnail(path, destination, at_seconds)
    repository=MyRepository(),    # the Repository protocol in repositories/base.py
    clock=my_clock,               # callable returning a datetime
    executor=my_executor,         # any concurrent.futures.Executor
)
```

- **Recognition engine**: write a class with the three members above, or add a factory to `ENGINE_FACTORIES` in `processing/registry.py` to select it with `WCE_ENGINE`. Inside the pipeline engine, replace any field of `PipelineStages` (sampler, board detector, cropper, keyframe selector, document builder).
- **Export format**: add an `ExportFormat(key, label, extension, mimetype, writer)` to the formats passed to `ExportService`; the video page lists it and `/videos/<id>/export/<key>` serves it.

## Troubleshooting

- **"OpenCV is required to read video files"**: run `pip install opencv-python-headless`.
- **Upload says the file could not be read as a video**: the file is damaged or uses a codec this OpenCV build cannot open; convert it to MP4 (H.264).
- **Port 5000 is in use**: add `--port 5001` to the `flask` command, or `--listen=127.0.0.1:5001` to `waitress-serve`.
- **The stylesheet looks stale**: reload the page once; the stylesheet rebuilds when a file in `static/scss/` changes.
- **Large uploads are rejected**: raise `WCE_MAX_UPLOAD_GB`.
