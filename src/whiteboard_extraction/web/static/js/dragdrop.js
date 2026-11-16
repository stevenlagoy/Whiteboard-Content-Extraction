// Drag videos onto folders to move them, or onto other videos to reorder them.

import { applyResponse, fetchPage } from "./regions.js";

const TARGETS = "[data-drop-before], [data-drop-folder]";
let draggedId = null;
let highlighted = null;

function highlight(element) {
  if (highlighted === element) return;
  highlighted?.classList.remove("isDropTarget");
  element?.classList.add("isDropTarget");
  highlighted = element;
}

function placement(target) {
  if (target.matches("[data-drop-before]")) {
    return { folderId: target.dataset.folderId, beforeId: target.dataset.dropBefore };
  }
  return { folderId: target.dataset.dropFolder, beforeId: null };
}

async function move(source, { folderId, beforeId }) {
  const body = new FormData();
  body.set("folder_id", folderId);
  if (beforeId) body.set("before_id", beforeId);
  body.set("next", location.pathname + location.search);
  await applyResponse(await fetchPage(source.dataset.moveUrl, { method: "POST", body }));
}

document.addEventListener("dragstart", (event) => {
  const source = event.target.closest?.("[data-video-id]");
  if (!source) return;
  draggedId = source.dataset.videoId;
  source.classList.add("isDragging");
  event.dataTransfer.setData("text/plain", draggedId);
  event.dataTransfer.effectAllowed = "move";
});

document.addEventListener("dragover", (event) => {
  if (!draggedId) return;
  const target = event.target.closest?.(TARGETS);
  if (target && placement(target).beforeId !== draggedId) {
    event.preventDefault();
    highlight(target);
  } else {
    highlight(null);
  }
});

document.addEventListener("drop", async (event) => {
  const target = event.target.closest?.(TARGETS);
  const source = document.querySelector(`[data-video-id="${CSS.escape(draggedId ?? "")}"]`);
  highlight(null);
  if (!target || !source || placement(target).beforeId === draggedId) return;
  event.preventDefault();
  await move(source, placement(target));
});

document.addEventListener("dragend", () => {
  document.querySelector(".isDragging")?.classList.remove("isDragging");
  highlight(null);
  draggedId = null;
});
