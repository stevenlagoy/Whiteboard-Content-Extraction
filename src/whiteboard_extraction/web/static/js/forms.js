// Progressive enhancement for forms: confirmation, in-place submission, popovers, dismissible messages.

import { applyResponse, fetchPage } from "./regions.js";
import { refreshTasks } from "./tasks.js";

document.addEventListener("submit", async (event) => {
  const form = event.target;
  if (!(form instanceof HTMLFormElement)) return;

  const question = form.dataset.confirm;
  if (question && !window.confirm(question)) {
    event.preventDefault();
    return;
  }
  if (!("enhance" in form.dataset)) return;

  event.preventDefault();
  let response;
  try {
    response = await fetchPage(form.action, { method: form.method.toUpperCase(), body: new FormData(form) });
  } catch {
    form.submit();
    return;
  }
  await applyResponse(response);
  refreshTasks();
});

document.addEventListener("click", (event) => {
  const dismiss = event.target.closest?.("[data-dismiss]");
  if (dismiss) {
    dismiss.closest(".message")?.remove();
    return;
  }
  for (const popover of document.querySelectorAll("details[data-popover][open]")) {
    if (!popover.contains(event.target)) popover.open = false;
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  for (const popover of document.querySelectorAll("details[data-popover][open]")) {
    popover.open = false;
    popover.querySelector("summary")?.focus();
  }
});

// The toggle event does not bubble, so listen in the capture phase.
document.addEventListener(
  "toggle",
  (event) => {
    const popover = event.target;
    if (popover.matches?.("details[data-popover]") && popover.open) popover.querySelector("input")?.focus();
  },
  true,
);
