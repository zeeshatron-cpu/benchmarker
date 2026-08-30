"""Fenzo AI adapter (web automation).

Fenzo AI is only available as a web app, so there is no API to call — we drive
the real UI with a headless browser. Everything site-specific (URL, the input
box, where the content lands, optional login) is supplied through config, so this
adapter needs no code changes if Fenzo's markup shifts: you update selectors in
``config.yaml``.

**Fenzo is a course generator, not a chatbot.** Submitting a prompt navigates
through several pages (``/home`` → ``/course/<id>`` → ``/course/.../<lesson>``)
and renders a generated lesson as multiple ``.markdown-viewer`` sections (the
last one is often an empty trailer). So the adapter does NOT wait for a single
chat bubble — it polls the page, joins the text of every non-empty content
section, and returns once that combined text stops changing (generation settles
after ~20–30s). Because generation navigates repeatedly, the poll tolerates the
transient "execution context destroyed" errors that happen mid-navigation.

Playwright is an optional dependency; the import is lazy. Chromium is expected at
the platform's configured path (in this environment Playwright is preconfigured
— do not run ``playwright install``).

Config block (see config.example.yaml) — the ``fenzo`` adapter options:
    url:            page with the prompt box (default ``https://fenzo.ai/home``)
    input_selector: prompt box selector (default ``#fenzo-input-box``)
    send_selector:  send button (optional; Enter is used if absent)
    response_selector: content sections to join (default ``.markdown-viewer``)
    ready_selector: optional selector to wait for before typing (e.g. after login)
    settle_ms:      grace after submitting before polling begins (ms)
    stable_ms:      content is "done" once the joined text is unchanged this long (ms)
    timeout_ms:     overall cap on waiting for the generated content (ms)
    storage_state:  path to a Playwright storage-state JSON for a logged-in session
    headless:       bool
"""

from __future__ import annotations

import time
from typing import Any

from .base import ModelAdapter


class FenzoWebAdapter(ModelAdapter):
    def __init__(
        self,
        name: str = "fenzo",
        url: str = "https://fenzo.ai/home",
        input_selector: str = "#fenzo-input-box",
        send_selector: str | None = None,
        response_selector: str = ".markdown-viewer",
        ready_selector: str | None = None,
        settle_ms: int = 1000,
        stable_ms: int = 2500,
        timeout_ms: int = 120000,
        storage_state: str | None = None,
        headless: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.url = url
        self.input_selector = input_selector
        self.send_selector = send_selector
        self.response_selector = response_selector
        self.ready_selector = ready_selector
        self.settle_ms = settle_ms
        self.stable_ms = stable_ms
        self.timeout_ms = timeout_ms
        self.storage_state = storage_state
        self.headless = headless
        self._pw = None
        self._browser = None
        self._context = None

    def _ensure_browser(self):
        if self._context is not None:
            return
        from playwright.sync_api import sync_playwright  # lazy optional import

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        ctx_kwargs: dict[str, Any] = {}
        if self.storage_state:
            ctx_kwargs["storage_state"] = self.storage_state
        self._context = self._browser.new_context(**ctx_kwargs)

    def _collect(self, page) -> str:
        """Join the text of every non-empty content section on the page."""
        loc = page.locator(self.response_selector)
        parts: list[str] = []
        for i in range(loc.count()):
            t = (loc.nth(i).inner_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts)

    def _wait_for_stable_reply(self, page, before: int = 0) -> str:
        """Poll the joined content until it stops changing (generation done).

        `before` is accepted for backward-compatibility and ignored — the join
        strategy doesn't need a pre-count baseline.
        """
        page.wait_for_timeout(self.settle_ms)
        prev = ""
        stable_for_ms = 0
        poll_ms = 1000
        deadline = time.monotonic() + self.timeout_ms / 1000
        while time.monotonic() < deadline:
            try:
                cur = self._collect(page)
            except Exception:
                # Mid-navigation (course generation hops pages) — retry.
                cur = prev
            if cur and cur == prev:
                stable_for_ms += poll_ms
                if stable_for_ms >= self.stable_ms:
                    return cur
            else:
                stable_for_ms = 0
                prev = cur
            page.wait_for_timeout(poll_ms)
        # Timed out — return the best content captured rather than nothing.
        return prev

    def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if not self.url:
            raise ValueError(
                "fenzo adapter needs a 'url' in config to drive the web app"
            )
        self._ensure_browser()
        page = self._context.new_page()
        page.set_default_timeout(self.timeout_ms)
        try:
            page.goto(self.url)
            if self.ready_selector:
                page.wait_for_selector(self.ready_selector)

            box = page.locator(self.input_selector).first
            box.click()
            box.fill(prompt)

            if self.send_selector:
                page.locator(self.send_selector).first.click()
            else:
                box.press("Enter")

            text = self._wait_for_stable_reply(page)
            return text, {
                "source": "web",
                "final_url": page.url,
                "chars": len(text),
            }
        finally:
            page.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
            self._pw = None
