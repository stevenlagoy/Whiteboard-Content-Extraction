"""Flask app"""

import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_file

UPLOAD_DIR = Path("instance/uploads")
OUTPUT_DIR = Path("instance/outputs")

def create_app() -> Flask:
    app = Flask(__name__)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("upload.html")

    @app.route("/process", methods=["POST"])
    def process():
        video = request.files.get("video")
        if video is None or video.filename == "":
            return render_template("upload.html", error="Please choose a video file")

        job_id = uuid.uuid4().hex
        video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
        video.save(video_path)

        # TODO: wire into real pipeline
        # keyframes = extract_keyframes(sample_frames(video_path))
        # blocks = [recognizer.recognize(t, f) for t, f in keyframes]
        # document = build_document(blocks)
        # export_docx(document, OUTPUT_DIR / f"{job_id}.docx")
        # export_pdf(document, OUTPUT_DIR / f"{job_id}.pdf")

        return render_template("results.html", job_id=job_id)

    @app.route("/download/<job_id>/<file_type>")
    def download(job_id: str, file_type: str):
        extension = {"word": "docx", "pdf": "pdf"}.get(file_type)
        if extension is None:
            return "Unknown file type.", 404
        output_path = OUTPUT_DIR / f"{job_id}.{extension}"
        if not output_path.exists():
            return "File not found or not yet ready.", 404
        return send_file(output_path, as_attachment=True)

    return app

if __name__ == "__main__":
    create_app().run(debug=True)