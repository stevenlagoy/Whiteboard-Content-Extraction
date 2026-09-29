# User Stories

## Instructor

1) AS AN INSTRUCTOR, I want to upload a recording of my lecture, so that I don't need any special recording setup beyond what my classroom already has.
    - The web app accepts a video file upload from a standard classroom recording (no special camera, mount, or capture software required).
    - Upload succeeds for at least the common video formats produced by the university's existing classroom recording system.
    - A failed or unsupported upload shows a clear error message rather than crashing or hanging silently.

2) AS AN INSTRUCTOR, I want the tool to work on recordings up to 150 minutes long, so that it covers a full class period without me trimming the video first.
    - A 150-minute video is accepted and processed end-to-end without failure.
    - Processing time for a 150-minute video does not exceed 150 minutes (1x runtime, per requirement 12).
    - Memory or storage limits are documented if they cap the maximum supported length below 150 minutes.

3) AS AN INSTRUCTOR, I want to receive my board content as a Word document and a PDF, so that I can edit or distribute it the way I already share other course materials.
    - Every successful run produces both a .docx and a .pdf file.
    - Both files contain the same reconstructed content and open correctly in Microsoft Word and a standard PDF viewer.
    - The .docx is editable (text is real text, not an embedded image).

4) AS AN INSTRUCTOR, I want the interface to require no coding or command-line steps, so that I can use it without technical help.
    - The entire workflow (upload, process, download) is completable through the browser UI alone.
    - No step instructs or requires the user to open a terminal, edit a config file, or run a script.
    - A first-time user can complete the workflow without external documentation.

5) AS AN INSTRUCTOR, I want a LaTeX export option, so that I can drop equations directly into other course materials I write in LaTeX.
    - A .tex file is available as an export option alongside Word and PDF.
    - Recognized equations render correctly when the .tex file is compiled.
    - Missing LaTeX export does not block Word/PDF export or count as a failed run.

## Student

6) AS A STUDENT, I want the board content presented with timestamps, so that I can connect a piece of text back to the moment it was written in the lecture.
    - Every recognized block of content (equation or prose) in the exported document is labeled with a timestamp corresponding to its appearance in the source video.
    - Timestamps are accurate to within a defined tolerance (e.g., a few seconds) of when the content was actually written.
    - Timestamp formatting is consistent across the Word and PDF exports.

7) AS A STUDENT, I want equations and prose to be recognized accurately, so that I can trust the exported notes instead of re-watching the video to check them.
    - Recognition accuracy on a held-out test set meets a minimum threshold agreed on by the team (to be set once baseline results exist).
    - Accuracy, precision, recall, F1, and WER are reported for both equations and prose, per requirement 11.
    - Errors are visually distinguishable or flagged in some way, rather than silently presented as confident, correct text (stretch, if time allows).

8) AS A STUDENT, I want to search the recovered content by keyword, so that I can jump straight to the timestamp where a topic was covered.
    - The UI provides a search box that filters or highlights recovered content by keyword.
    - A search result links to or displays the associated timestamp.
    - Search responsiveness meets the end-to-end performance goal for search responsiveness (per the System Evaluation plan).

## Accessibility

9) AS A DAC STAFF MEMBER, I want the exported document to meet ADA accessibility guidelines, so that it's usable with the assistive technology our students already rely on.
    - Exported Word and PDF documents pass a basic accessibility check (e.g., proper heading structure, alt text where relevant, readable by a screen reader without manual reformatting).
    - DAC staff review at least one sample export and confirm it meets their baseline expectations.
    - Accessibility requirements are documented so future changes to the export step don't silently break them.

10) AS A DAC STAFF MEMBER, I want a screen-reader-optimized output format, so that a student using a screen reader can navigate the content without extra manual reformatting.
    - A dedicated screen-reader-optimized export (or a validated mode of the existing export) is available.
    - A screen reader can navigate the document by heading or section without encountering unlabeled content.
    - This is validated with an actual screen reader (e.g., NVDA or VoiceOver), not just visual inspection.

## Team

11) AS A MEMBER OF THE PROJECT TEAM, I want quantitative accuracy metrics (accuracy, precision, recall, F1, confusion matrix, WER) for each DL method, so that we can justify which method we recommend.
    - Each evaluated DL method (custom CNN, pretrained OCR, multimodal LLM) has a reported accuracy, precision, recall, F1-score, confusion matrix, and WER on the same test set.
    - Metrics are computed with a repeatable script, not one-off manual calculation.
    - Results are presented in a comparable format (e.g., one table) for the final demo.

12) AS A MEMBER OF THE PROJECT TEAM, I want processing time to stay under 1x the video's runtime, so that the tool is practical for an instructor to use after class rather than overnight.
    - Processing time is measured end-to-end (upload to export) for at least one test video of each supported length tier (e.g., 50, 75, 150 minutes).
    - Measured processing time is at or below the video's own runtime in each case.
    - If a chosen DL method violates this on its own, that tradeoff is documented in the model comparison.

13) AS A MEMBER OF THE PROJECT TEAM, I want the pipeline built so DL methods can be swapped in and out, so that comparing a custom CNN, a pretrained OCR model, and a multimodal LLM doesn't require rewriting the rest of the app.
    - The recognition step is implemented behind a common interface (e.g., a shared function signature or class) that each DL method conforms to.
    - Switching between methods requires changing a config value or a single call site, not editing the preprocessing or export code.
    - At least two methods have been plugged into this interface and run successfully, confirming the abstraction holds in practice.

14) AS A MEMBER OF THE PROJECT TEAM, I want the system to tolerate camera motion and multiple video sources, so that it holds up on real, imperfect classroom footage rather than only clean test clips.
    - At least one test video with camera motion (panning, zoom, or shake) is processed without the pipeline crashing.
    - Recognition accuracy on that footage is measured and compared to accuracy on stable footage, even if it's lower.
    - Support for more than one video source per session is either demonstrated or explicitly deferred with a documented reason.

15) AS A MEMBER OF THE PROJECT TEAM, I want an output retention policy (e.g., 30 days), so that we have a clear, defensible answer about data handling if asked.
    - A stated retention period is implemented (e.g., a scheduled deletion job or documented manual process).
    - The policy is written down somewhere a reviewer or DAC contact could read it, not just known informally by the team.
    - Uploaded videos and generated outputs are both covered by the policy, not just one or the other.