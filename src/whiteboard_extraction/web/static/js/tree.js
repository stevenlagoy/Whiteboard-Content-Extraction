// Expand and collapse folders in the sidebar; remember the choice between visits.

const STORAGE_KEY = "wce.collapsedFolders";

function loadCollapsed() {
  try {
    return new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]"));
  } catch {
    return new Set();
  }
}

function saveCollapsed(collapsed) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...collapsed]));
  } catch {
    // Storage can be unavailable (private mode); the tree still works for this page.
  }
}

function setExpanded(item, expanded) {
  const toggle = item.querySelector(":scope > .treeRow > [data-tree-toggle]");
  const children = item.querySelector(":scope > .treeChildren");
  if (!toggle || !children) return;
  toggle.setAttribute("aria-expanded", String(expanded));
  children.hidden = !expanded;
}

export function applyTreeState() {
  const collapsed = loadCollapsed();
  for (const item of document.querySelectorAll(".treeFolder")) {
    setExpanded(item, !collapsed.has(item.dataset.folderId));
  }
  // Always show where the user currently is.
  for (const current of document.querySelectorAll('.tree [aria-current="page"]')) {
    for (let item = current.closest(".treeFolder"); item; item = item.parentElement?.closest(".treeFolder")) {
      setExpanded(item, true);
    }
  }
}

document.addEventListener("click", (event) => {
  const toggle = event.target.closest?.("[data-tree-toggle]");
  if (!toggle) return;
  const item = toggle.closest(".treeFolder");
  const expanded = toggle.getAttribute("aria-expanded") !== "true";
  setExpanded(item, expanded);
  const collapsed = loadCollapsed();
  if (expanded) collapsed.delete(item.dataset.folderId);
  else collapsed.add(item.dataset.folderId);
  saveCollapsed(collapsed);
});
