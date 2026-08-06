"""Fenzo AI adapter (web automation).

Fenzo AI is only available as a web app, so there is no API to call — we drive
the real UI with a headless browser. Everything site-specific (URL, the input
box, the send control, where the reply lands, optional login) is supplied
through config, so this adapter needs no code changes if Fenzo's markup shifts:
you update selectors in ``config.yaml``.

Playwright is an optional dependency; the import is lazy. Chromium is expected at
the platform's configured path (in this environment Playwright is preconfigured
— do not run ``playwright install``).

Streaming-safe reply capture: chat UIs stream the answer token by token, so a
fixed wait either truncates long replies or wastes time on short ones. Instead we
wait for a *new* assistant bubble to appear, then poll its text until it stops
growing for ``stable_ms`` (bounded by ``timeout_ms``). ``settle_ms`` is only a
small initial grace before polling begins.

Config block (see config.example.yaml) — the ``fenzo`` adapter options:
    url:            chat page URL
    input_selector: selector for the prompt box (default ``#fenzo-input-box``)
    send_selector:  selector for the send button (optional; Enter used if absent)
    response_selector: selector matching assistant reply bubbles (default ``.markdown-viewer``)
    ready_selector: optional selector to wait for before typing (e.g. after login)
    settle_ms:      initial grace before polling for reply text (ms)
    stable_ms:      reply is "done" once its text is unchanged this long (ms)
    timeout_ms:     per-step / overall reply timeout (ms)
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
        url: str = "",
        input_selector: str = "#fenzo-input-box",
        send_selector: str | None = None,
        response_selector: str = ".markdown-viewer",
        ready_selector: str | None = None,
        settle_ms: int = 500,
        stable_ms: int = 1200,
        timeout_ms: int = 90000,
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

    def _wait_for_stable_reply(self, page, before: int) -> str:
        """Wait for a new bubble, then until its text stops growing."""
        # A new assistant bubble has appeared.
        page.wait_for_function(
            "([sel, n]) => document.querySelectorAll(sel).length > n",
            arg=[self.response_selector, before],
        )
        page.wait_for_timeout(self.settle_ms)

        bubbles = page.locator(self.response_selector)
        last = bubbles.nth(bubbles.count() - 1)

        prev = ""
        stable_for_ms = 0
        poll_ms = 300
        deadline = time.monotonic() + self.timeout_ms / 1000
        while time.monotonic() < deadline:
            cur = (last.inner_text() or "").strip()
            if cur and cur == prev:
                stable_for_ms += poll_ms
                if stable_for_ms >= self.stable_ms:
                    return cur
            else:
                stable_for_ms = 0
                prev = cur
            page.wait_for_timeout(poll_ms)
        # Timed out mid-stream — return what we have rather than nothing.
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

            # Count existing assistant bubbles so we can identify the new one.
            before = page.locator(self.response_selector).count()

            box = page.locator(self.input_selector).first
            box.click()
            box.fill(prompt)

            if self.send_selector:
                page.locator(self.send_selector).first.click()
            else:
                box.press("Enter")

            text = self._wait_for_stable_reply(page, before)
            return text, {"source": "web", "url": self.url}
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
