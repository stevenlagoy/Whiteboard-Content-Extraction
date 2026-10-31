// Partial page updates: fetch a page and swap only the regions that change.

const REGIONS = ["sidebar", "main", "messages"];

function currentUrl() {
  return location.pathname + location.search;
}

export function fetchPage(url, options = {}) {
  const headers = { "X-Requested-With": "fetch", "X-Page-URL": currentUrl(), ...(options.headers ?? {}) };
  return fetch(url, { ...options, headers, redirect: "follow" });
}

function restoreFocus(key) {
  const target = key ? document.querySelector(`[data-focus-key="${CSS.escape(key)}"]`) : null;
  if (target) {
    target.focus();
  } else if (document.activeElement === document.body) {
    document.getElementById("main")?.focus({ preventScroll: true });
  }
}

export function swapRegions(html) {
  const incoming = new DOMParser().parseFromString(html, "text/html");
  const focusKey = document.activeElement?.dataset?.focusKey;
  const scrolls = [...document.querySelectorAll("[data-scroll-key]")].map((el) => [el.dataset.scrollKey, el.scrollTop]);

  for (const name of REGIONS) {
    const next = incoming.querySelector(`[data-region="${name}"]`);
    const current = document.querySelector(`[data-region="${name}"]`);
    if (next && current) current.innerHTML = next.innerHTML;
  }
  document.title = incoming.title;

  for (const [key, top] of scrolls) {
    const el = document.querySelector(`[data-scroll-key="${key}"]`);
    if (el) el.scrollTop = top;
  }
  restoreFocus(focusKey);
  document.dispatchEvent(new CustomEvent("regions:updated"));
}

// Swap regions when the response is for the page we are on; otherwise navigate to it.
export async function applyResponse(response) {
  const target = new URL(response.url);
  if (!response.ok || target.pathname + target.search !== currentUrl()) {
    location.assign(response.url);
    return;
  }
  swapRegions(await response.text());
}

export async function refreshPage() {
  await applyResponse(await fetchPage(location.href));
}

// Update the current page after a background task changed state.
export async function refreshLive() {
  const main = document.querySelector("main[data-live]");
  if (!main) return;
  if (main.dataset.live === "reload") {
    if (!main.dataset.dirty) location.reload();
    return;
  }
  await refreshPage();
}

export function isEditing() {
  const active = document.activeElement;
  return Boolean(active?.matches?.("input, textarea, select") && active.closest("[data-region]"));
}

document.addEventListener("input", (event) => {
  const main = event.target.closest?.("main[data-live]");
  if (main) main.dataset.dirty = "1";
});
