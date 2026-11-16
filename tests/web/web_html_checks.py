"""A small HTML structure checker used by the page tests."""

from __future__ import annotations

from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class PageChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.problems: list[str] = []
        self.ids: list[str] = []
        self.label_targets: set[str] = set()
        self.controls: list[dict] = []
        self.headings: list[str] = []
        self.images_without_alt = 0
        self._label_depth = 0

    def _inspect(self, tag: str, attrs: list) -> dict:
        names = [name for name, _ in attrs]
        for name in set(names):
            if names.count(name) > 1:
                self.problems.append(f"<{tag}> repeats the attribute {name!r}")
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "label" and "for" in values:
            self.label_targets.add(values["for"])
        if tag == "img" and "alt" not in values:
            self.images_without_alt += 1
        if tag in ("input", "select", "textarea") and values.get("type") != "hidden":
            self.controls.append({**values, "tag": tag, "in_label": self._label_depth > 0})
        if tag in ("h1", "h2", "h3"):
            self.headings.append(tag)
        return values

    def handle_starttag(self, tag, attrs):
        self._inspect(tag, attrs)
        if tag not in VOID:
            self.stack.append(tag)
            if tag == "label":
                self._label_depth += 1

    def handle_startendtag(self, tag, attrs):
        self._inspect(tag, attrs)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
            if tag == "label":
                self._label_depth -= 1
        else:
            self.problems.append(f"</{tag}> closes {self.stack[-1] if self.stack else 'nothing'}")

    def unlabeled_controls(self) -> list[dict]:
        return [
            c for c in self.controls
            if not c["in_label"] and "aria-label" not in c and c.get("id") not in self.label_targets
        ]


def check(html: str) -> PageChecker:
    checker = PageChecker()
    checker.feed(html)
    checker.close()
    if checker.stack:
        checker.problems.append(f"unclosed tags: {checker.stack}")
    dupes = {i for i in checker.ids if checker.ids.count(i) > 1}
    if dupes:
        checker.problems.append(f"duplicate ids: {sorted(dupes)}")
    return checker
