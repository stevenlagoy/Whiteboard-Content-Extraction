import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

const JS_URL = new URL("../../../src/whiteboard_extraction/web/static/js/", import.meta.url);

// Loads the real ES modules from static/js into a jsdom window, with a fake location
// (jsdom cannot stub location.assign or location.reload).
async function boot(html, { url = "http://localhost/", fetchImpl } = {}) {
  const dom = new JSDOM(html, { url, pretendToBeVisual: true });
  const { window } = dom;
  globalThis.window = window;
  const parsed = new URL(url);
  const loc = {
    pathname: parsed.pathname, search: parsed.search, href: parsed.href, assigned: [], reloads: 0,
    assign(target) { this.assigned.push(target); }, reload() { this.reloads += 1; },
  };
  Object.defineProperty(globalThis, "location", { value: loc, configurable: true, writable: true });
  for (const key of ["document", "CustomEvent", "DOMParser", "FormData", "HTMLFormElement", "CSS", "localStorage", "Event", "XMLHttpRequest", "URL"]) {
    Object.defineProperty(globalThis, key, { value: window[key] ?? globalThis[key], configurable: true, writable: true });
  }
  window.CSS = globalThis.CSS = { escape: (s) => String(s).replace(/"/g, '\\"') };
  globalThis.fetch = fetchImpl ?? (async () => { throw new Error("no fetch"); });
  window.confirm = () => true;
  const stamp = Math.random();
  const load = (name) => import(new URL(name, JS_URL).href + `?v=${stamp}`);
  return { dom, window, load, loc };
}

test("timecode formatting mirrors the server", async () => {
  const { load } = await boot("<html></html>");
  const { formatTimecode } = await load("timecode.js");
  assert.equal(formatTimecode(0), "0:00");
  assert.equal(formatTimecode(61.9), "1:01");
  assert.equal(formatTimecode(3600), "1:00:00");
  assert.equal(formatTimecode(5716), "1:35:16");
  assert.equal(formatTimecode(-3), "0:00");
});

const PAGE = (extra = "") => `<html><head><title>Old</title></head><body>
  <aside data-region="sidebar"><div class="treeScroll" data-scroll-key="tree"><button data-focus-key="k1">old</button></div></aside>
  <div data-region="messages"></div><main id="main" tabindex="-1" data-region="main" data-live="swap"><p id="p">old main</p></main>${extra}</body></html>`;

const INCOMING = `<html><head><title>New</title></head><body>
  <aside data-region="sidebar"><div class="treeScroll" data-scroll-key="tree"><button data-focus-key="k1">new</button></div></aside>
  <div data-region="messages"><div class="message">saved</div></div><main data-region="main"><p id="p">new main</p></main></body></html>`;

test("swapRegions replaces regions, keeps scroll and focus, updates the title", async () => {
  const { window, load } = await boot(PAGE());
  const { swapRegions } = await load("regions.js");
  const scroller = window.document.querySelector(".treeScroll");
  Object.defineProperty(scroller, "scrollTop", { value: 120, writable: true });
  window.document.querySelector("[data-focus-key=k1]").focus();
  let announced = 0;
  window.document.addEventListener("regions:updated", () => announced++);

  swapRegions(INCOMING);

  assert.equal(window.document.querySelector("#p").textContent, "new main");
  assert.equal(window.document.querySelector("[data-region=messages]").textContent.trim(), "saved");
  assert.equal(window.document.title, "New");
  assert.equal(window.document.activeElement.textContent, "new");
  assert.equal(window.document.querySelector("main").getAttribute("id"), "main"); // attributes on the region survive
  assert.equal(announced, 1);
});

test("focus falls back to the main region when the focused element disappears", async () => {
  const { window, load } = await boot(PAGE());
  const { swapRegions } = await load("regions.js");
  window.document.querySelector("[data-focus-key=k1]").focus();
  swapRegions(INCOMING.replace("data-focus-key=\"k1\"", "data-focus-key=\"other\""));
  assert.equal(window.document.activeElement.id, "main");
});

test("applyResponse swaps for the same page and navigates otherwise", async () => {
  const { window, load, loc } = await boot(PAGE(), { url: "http://localhost/folders/a?x=1" });
  const { applyResponse } = await load("regions.js");
  const res = (url, ok = true) => ({ url, ok, text: async () => INCOMING });
  await applyResponse(res("http://localhost/folders/a?x=1"));
  assert.equal(window.document.querySelector("#p").textContent, "new main");
  await applyResponse(res("http://localhost/"));
  await applyResponse(res("http://localhost/folders/a?x=1", false));
  assert.deepEqual(loc.assigned, ["http://localhost/", "http://localhost/folders/a?x=1"]);
});

test("fetchPage sends the page URL so fragments know where the user is", async () => {
  let seen;
  const { load } = await boot(PAGE(), { url: "http://localhost/videos/v1?t=2", fetchImpl: async (u, o) => { seen = o; return {}; } });
  const { fetchPage } = await load("regions.js");
  await fetchPage("/tasks/panel");
  assert.equal(seen.headers["X-Page-URL"], "/videos/v1?t=2");
  assert.equal(seen.headers["X-Requested-With"], "fetch");
});

test("refreshLive reloads the video page only when nothing was edited", async () => {
  const html = `<html><body><main data-region="main" data-live="reload"><input id="i"></main></body></html>`;
  const { window, load, loc } = await boot(html);
  const { refreshLive } = await load("regions.js");
  await refreshLive();
  assert.equal(loc.reloads, 1);
  window.document.querySelector("#i").dispatchEvent(new window.Event("input", { bubbles: true }));
  await refreshLive();
  assert.equal(loc.reloads, 1);
});

const TREE = `<html><body><ul class="tree">
  <li class="treeFolder" data-folder-id="a"><div class="treeRow"><button data-tree-toggle aria-expanded="true">t</button>
    <a aria-current="page" class="treeLabel">A</a></div><ul class="treeChildren" id="c-a">
      <li class="treeFolder" data-folder-id="b"><div class="treeRow"><button data-tree-toggle aria-expanded="true">t</button></div><ul class="treeChildren"><li><a aria-current="page">leaf</a></li></ul></li></ul></li>
  <li class="treeFolder" data-folder-id="c"><div class="treeRow"><button data-tree-toggle aria-expanded="true">t</button></div><ul class="treeChildren"><li>x</li></ul></li></ul></body></html>`;

test("tree toggles, remembers collapsed folders, and reveals the current page", async () => {
  const { window, load } = await boot(TREE);
  const { applyTreeState } = await load("tree.js");
  const toggle = (id) => window.document.querySelector(`[data-folder-id=${id}] > .treeRow > [data-tree-toggle]`);
  const kids = (id) => window.document.querySelector(`[data-folder-id=${id}] > .treeChildren`);

  toggle("c").click();
  assert.equal(toggle("c").getAttribute("aria-expanded"), "false");
  assert.equal(kids("c").hidden, true);
  assert.deepEqual(JSON.parse(window.localStorage.getItem("wce.collapsedFolders")), ["c"]);

  // A fresh render (as after a region swap) restores the saved state.
  window.document.body.innerHTML = TREE.replace(/^.*<body>|<\/body>.*$/gs, "");
  applyTreeState();
  assert.equal(kids("c").hidden, true);

  // Collapsing the ancestor of the current page is undone for display, not for storage.
  window.localStorage.setItem("wce.collapsedFolders", JSON.stringify(["a", "b"]));
  applyTreeState();
  assert.equal(kids("a").hidden, false);
  assert.equal(kids("b").hidden, false);

  assert.equal(kids("c").hidden, false); // c is not in the saved list any more
  toggle("c").click();
  assert.equal(kids("c").hidden, true);
  toggle("c").click();
  assert.equal(kids("c").hidden, false);
});

const ROWS = `<html><body>
  <ul><li id="row" class="videoRow" draggable="true" data-video-id="v1" data-move-url="/videos/v1/move" data-folder-id="f1">row</li>
  <li id="row2" class="videoRow" data-video-id="v2" data-move-url="/videos/v2/move" data-folder-id="f1" data-drop-before="v2">second</li></ul>
  <div id="folder" data-drop-folder="f2">folder</div><div id="root" data-drop-folder="">root</div></body></html>`;

function dragEvent(window, type, target) {
  const e = new window.Event(type, { bubbles: true, cancelable: true });
  e.dataTransfer = { data: {}, setData(k, v) { this.data[k] = v; }, effectAllowed: "" };
  target.dispatchEvent(e);
  return e;
}

test("dropping a video on a folder posts a move to the end of that folder", async () => {
  const calls = [];
  const { window, load } = await boot(ROWS, { fetchImpl: async (u, o) => { calls.push([u, o]); return { ok: true, url: "http://localhost/", text: async () => "" }; } });
  await load("dragdrop.js");
  const $ = (s) => window.document.querySelector(s);
  dragEvent(window, "dragstart", $("#row"));
  assert.ok($("#row").classList.contains("isDragging"));
  const over = dragEvent(window, "dragover", $("#folder"));
  assert.equal(over.defaultPrevented, true);
  assert.ok($("#folder").classList.contains("isDropTarget"));
  dragEvent(window, "drop", $("#folder"));
  await new Promise((r) => setTimeout(r, 10));
  const [url, options] = calls[0];
  assert.equal(url, "/videos/v1/move");
  assert.equal(options.method, "POST");
  assert.equal(options.body.get("folder_id"), "f2");
  assert.equal(options.body.has("before_id"), false);
  dragEvent(window, "dragend", $("#row"));
  assert.ok(!$("#row").classList.contains("isDragging"));
  assert.ok(!$("#folder").classList.contains("isDropTarget"));
});

test("dropping on a video reorders; the top-level target sends an empty folder id", async () => {
  const calls = [];
  const { window, load } = await boot(ROWS, { fetchImpl: async (u, o) => { calls.push(o.body); return { ok: true, url: "http://localhost/", text: async () => "" }; } });
  await load("dragdrop.js");
  const $ = (s) => window.document.querySelector(s);
  dragEvent(window, "dragstart", $("#row"));
  dragEvent(window, "drop", $("#row2"));
  dragEvent(window, "drop", $("#root"));
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(calls[0].get("folder_id"), "f1");
  assert.equal(calls[0].get("before_id"), "v2");
  assert.equal(calls[1].get("folder_id"), "");
});

test("a video cannot be dropped before itself, and unrelated drops are ignored", async () => {
  const calls = [];
  const { window, load } = await boot(ROWS, { fetchImpl: async (u, o) => { calls.push(u); return {}; } });
  await load("dragdrop.js");
  const $ = (s) => window.document.querySelector(s);
  dragEvent(window, "dragstart", $("#row2"));
  const over = dragEvent(window, "dragover", $("#row2"));
  assert.equal(over.defaultPrevented, false);
  dragEvent(window, "drop", $("#row2"));
  dragEvent(window, "drop", window.document.body);
  await new Promise((r) => setTimeout(r, 10));
  assert.deepEqual(calls, []);
});

const FORMS = `<html><head><title>t</title></head><body>
  <aside data-region="sidebar"></aside><div data-region="messages"></div>
  <main id="main" tabindex="-1" data-region="main"><p id="p">old</p></main>
  <form id="enh" method="post" action="http://localhost/videos/v1/favorite" data-enhance><input name="favorite" value="true"><button>go</button></form>
  <form id="conf" method="post" action="http://localhost/x" data-confirm="Sure?"><button>del</button></form>
  <details data-popover id="pop" open><summary>New</summary><input id="inp"></details><div class="message" id="msg"><button data-dismiss id="dismiss">x</button></div>
  <div data-dock data-tasks-url="/tasks/panel"><span data-tasks-count>0</span><details data-tasks-panel hidden><div data-tasks-region></div></details></div></body></html>`;

test("enhanced forms submit in place and refresh the regions", async () => {
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push([url, options]);
    if (url.includes("/tasks/panel")) return { ok: true, text: async () => '<div class="tasksList" data-signature="" data-active="0" data-total="0"></div>' };
    return { ok: true, url: "http://localhost/", text: async () => INCOMING };
  };
  const { window, load } = await boot(FORMS, { fetchImpl });
  await load("forms.js");
  const form = window.document.querySelector("#enh");
  form.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(calls[0][0], "http://localhost/videos/v1/favorite");
  assert.equal(calls[0][1].method, "POST");
  assert.equal(calls[0][1].body.get("favorite"), "true");
  assert.equal(window.document.querySelector("#p").textContent, "new main");
});

test("a declined confirmation cancels the submit; an accepted one lets it through", async () => {
  const { window, load } = await boot(FORMS);
  await load("forms.js");
  const form = window.document.querySelector("#conf");
  window.confirm = () => false;
  let event = new window.Event("submit", { bubbles: true, cancelable: true });
  form.dispatchEvent(event);
  assert.equal(event.defaultPrevented, true);
  window.confirm = () => true;
  event = new window.Event("submit", { bubbles: true, cancelable: true });
  form.dispatchEvent(event);
  assert.equal(event.defaultPrevented, false); // not enhanced: the browser submits normally
});

test("messages dismiss, and popovers close on outside click and Escape", async () => {
  const { window, load } = await boot(FORMS);
  await load("forms.js");
  const $ = (s) => window.document.querySelector(s);
  $("#dismiss").click();
  assert.equal($("#msg"), null);
  assert.equal($("#pop").open, true);
  $("#inp").click();
  assert.equal($("#pop").open, true);
  window.document.body.click();
  assert.equal($("#pop").open, false);
  $("#pop").open = true;
  window.document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  assert.equal($("#pop").open, false);
});

test("the task panel renders, shows its count, and hides when empty", async () => {
  const fragment = (active, total, sig) => `<div class="tasksList" data-signature="${sig}" data-active="${active}" data-total="${total}"><ul><li>t</li></ul></div>`;
  let reply = fragment(1, 1, "j:running");
  const { window, load } = await boot(FORMS, { fetchImpl: async () => ({ ok: true, text: async () => reply }) });
  const { refreshTasks } = await load("tasks.js");
  const panel = window.document.querySelector("[data-tasks-panel]");
  assert.equal(await refreshTasks(), 1);
  assert.equal(window.document.querySelector("[data-tasks-count]").textContent, "1");
  assert.equal(panel.hidden, false);
  reply = fragment(0, 0, "");
  assert.equal(await refreshTasks(), 0);
  assert.equal(panel.hidden, true);
});

test("player timestamps seek and play", async () => {
  const html = `<html><body><video data-player></video><button data-seek="65" id="b">go</button></body></html>`;
  const { window, load } = await boot(html);
  const video = window.document.querySelector("video");
  let played = 0;
  video.play = () => { played++; return Promise.resolve(); };
  await load("player.js");
  window.document.querySelector("#b").click();
  assert.equal(video.currentTime, 65);
  assert.equal(played, 1);
});

const UPLOAD = `<html><body><form data-upload-form action="http://localhost/jobs" method="post">
  <p data-form-error hidden></p><input type="file" data-file><video data-preview hidden></video><p data-preview-note hidden></p>
  <input data-trim="start"><button type="button" data-use-position="start" hidden></button>
  <input data-trim="end"><button type="button" data-use-position="end" hidden></button>
  <input data-name placeholder="Uses the file name"><div data-upload-progress hidden role="progressbar"><span class="progressFill"></span></div>
  <button data-submit>Add to queue</button></form></body></html>`;

test("upload page: metadata reveals trim helpers; position buttons fill the fields", async () => {
  const { window, load } = await boot(UPLOAD);
  await load("upload.js");
  const $ = (s) => window.document.querySelector(s);
  const video = $("[data-preview]");
  Object.defineProperty(video, "duration", { value: 125.7 });
  video.dispatchEvent(new window.Event("loadedmetadata"));
  assert.equal($('[data-trim="end"]').placeholder, "2:05");
  assert.equal($("[data-use-position=start]").hidden, false);
  video.currentTime = 31;
  $("[data-use-position=start]").click();
  assert.equal($('[data-trim="start"]').value, "0:31");
  video.currentTime = 3725;
  $("[data-use-position=end]").click();
  assert.equal($('[data-trim="end"]').value, "1:02:05");
  video.dispatchEvent(new window.Event("error"));
  assert.equal(video.hidden, true);
  assert.match($("[data-preview-note]").textContent, /cannot preview/);
});

class FakeXHR {
  static last;
  constructor() { this.upload = { listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; } }; this.listeners = {}; this.headers = {}; FakeXHR.last = this; }
  open(method, url) { this.method = method; this.url = url; }
  setRequestHeader(k, v) { this.headers[k] = v; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  send(body) { this.body = body; }
  reply(status, payload) { this.status = status; this.responseText = typeof payload === "string" ? payload : JSON.stringify(payload); this.listeners.load(); }
}

test("upload page: progress, success navigation, and error recovery", async () => {
  const { window, load, loc } = await boot(UPLOAD);
  window.XMLHttpRequest = globalThis.XMLHttpRequest = FakeXHR;
  await load("upload.js");
  const $ = (s) => window.document.querySelector(s);

  $("form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  const xhr = FakeXHR.last;
  assert.equal(xhr.method, "POST");
  assert.equal(xhr.headers.Accept, "application/json");
  assert.equal($("[data-submit]").disabled, true);
  xhr.upload.listeners.progress({ lengthComputable: true, loaded: 25, total: 100 });
  assert.equal($("[data-upload-progress]").getAttribute("aria-valuenow"), "25");
  assert.equal($(".progressFill").style.width, "25%");
  assert.equal($("[data-submit]").textContent, "Uploading 25%");
  xhr.reply(400, { error: "Choose a video file (MP4, MOV, MKV, WEBM, or AVI)." });
  assert.equal($("[data-form-error]").hidden, false);
  assert.match($("[data-form-error]").textContent, /Choose a video file/);
  assert.equal($("[data-submit]").disabled, false);
  assert.equal($("[data-upload-progress]").hidden, true);

  $("form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  FakeXHR.last.reply(200, { redirect: "/" });
  assert.deepEqual(loc.assigned, ["/"]);

  $("form").dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  FakeXHR.last.reply(502, "<html>bad gateway</html>");
  assert.match($("[data-form-error]").textContent, /status 502/);
});
