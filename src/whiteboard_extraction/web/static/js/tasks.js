// Keep the floating task panel current and refresh the page when a task changes state.

import { fetchPage, isEditing, refreshLive } from "./regions.js";

const dock = document.querySelector("[data-dock]");
const ACTIVE_DELAY_MS = 1000;
const IDLE_DELAY_MS = 5000;

let signature = dock?.querySelector(".tasksList")?.dataset.signature ?? "";
let stale = false;

function render(html) {
  dock.querySelector("[data-tasks-region]").innerHTML = html;
  const list = dock.querySelector(".tasksList");
  const active = Number(list?.dataset.active ?? 0);
  dock.querySelector("[data-tasks-count]").textContent = String(active);
  dock.querySelector("[data-tasks-panel]").hidden = Number(list?.dataset.total ?? 0) === 0;
  return { signature: list?.dataset.signature ?? "", active };
}

export async function refreshTasks() {
  if (!dock) return 0;
  const response = await fetchPage(dock.dataset.tasksUrl);
  if (!response.ok) return 0;
  const state = render(await response.text());
  if (state.signature !== signature) stale = true;
  signature = state.signature;
  return state.active;
}

async function tick() {
  let active = 0;
  if (!document.hidden) {
    try {
      active = await refreshTasks();
    } catch {
      // The next tick tries again.
    }
    if (stale && !isEditing()) {
      stale = false;
      await refreshLive().catch(() => {});
    }
  }
  setTimeout(tick, active > 0 ? ACTIVE_DELAY_MS : IDLE_DELAY_MS);
}

if (dock) setTimeout(tick, ACTIVE_DELAY_MS);
