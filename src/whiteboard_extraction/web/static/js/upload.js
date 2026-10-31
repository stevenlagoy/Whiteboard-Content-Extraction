// Upload screen: preview and trim the chosen file, then upload it with visible progress.

import { formatTimecode } from "./timecode.js";

const form = document.querySelector("[data-upload-form]");
if (form) setUp(form);

function setUp(form) {
  const $ = (selector) => form.querySelector(selector);
  const file = $("[data-file]");
  const preview = $("[data-preview]");
  const note = $("[data-preview-note]");
  const trimStart = $('[data-trim="start"]');
  const trimEnd = $('[data-trim="end"]');
  const name = $("[data-name]");
  const errorBox = $("[data-form-error]");
  const progress = $("[data-upload-progress]");
  const submit = $("[data-submit]");
  let objectUrl = null;

  file.addEventListener("change", () => {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    const chosen = file.files[0];
    preview.hidden = !chosen;
    note.hidden = true;
    if (!chosen) return;
    objectUrl = URL.createObjectURL(chosen);
    preview.src = objectUrl;
    if (!name.value) name.placeholder = chosen.name.replace(/\.[^.]+$/, "");
  });

  preview.addEventListener("loadedmetadata", () => {
    trimEnd.placeholder = formatTimecode(preview.duration);
    for (const button of form.querySelectorAll("[data-use-position]")) button.hidden = false;
  });

  preview.addEventListener("error", () => {
    preview.hidden = true;
    note.textContent = "This browser cannot preview that format. You can still type trim times.";
    note.hidden = false;
  });

  form.addEventListener("click", (event) => {
    const button = event.target.closest("[data-use-position]");
    if (!button) return;
    const field = button.dataset.usePosition === "start" ? trimStart : trimEnd;
    field.value = formatTimecode(preview.currentTime);
  });

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    errorBox.setAttribute("tabindex", "-1");
    errorBox.focus();
    progress.hidden = true;
    submit.disabled = false;
    submit.textContent = "Add to queue";
  }

  function showProgress(percent) {
    progress.setAttribute("aria-valuenow", String(percent));
    progress.querySelector(".progressFill").style.width = `${percent}%`;
    submit.textContent = `Uploading ${percent}%`;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    submit.disabled = true;
    progress.hidden = false;
    showProgress(0);

    const request = new XMLHttpRequest();
    request.open("POST", form.action);
    request.setRequestHeader("Accept", "application/json");
    request.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) showProgress(Math.round((e.loaded / e.total) * 100));
    });
    request.addEventListener("load", () => {
      let reply = {};
      try {
        reply = JSON.parse(request.responseText);
      } catch {
        // Not JSON: fall through to the generic message.
      }
      if (request.status >= 200 && request.status < 300 && reply.redirect) {
        location.assign(reply.redirect);
      } else {
        showError(reply.error ?? `The upload failed (status ${request.status}).`);
      }
    });
    request.addEventListener("error", () => showError("The upload could not reach the server."));
    request.send(new FormData(form));
  });
}
