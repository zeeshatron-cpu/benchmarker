"""Fenzo AI adapter (web automation).

Fenzo AI is only available as a web app, so there is no API to call — we drive
the real UI with a headless browser. Everything site-specific (URL, the input
box, the send control, where the reply lands, optional login) is supplied
through config, so this adapter needs no code changes if Fenzo's markup shifts:
you update selectors in ``config.yaml``.

Playwright is an optional dependency; the import is lazy. Chromium is expected at
the platform's configured path (in this environment Playwright is preconfigured
— do not run ``playwright install``).

Config block (see config.example.yaml) — the ``fenzo`` adapter options:
    url:            chat page URL
    input_selector: CSS/text selector for the prompt textarea
    send_selector:  selector for the send button (optional; Enter used if absent)
    response_selector: selector matching assistant message bubbles
    ready_selector: optional selector to wait for before typing (e.g. after login)
    settle_ms:      idle time to wait after the reply appears to catch streaming
    timeout_ms:     per-step timeout
    storage_state:  path to a Playwright storage-state JSON for a logged-in session
    headless:       bool
"""

from __future__ import annotations

from typing import Any

from .base import ModelAdapter


class FenzoWebAdapter(ModelAdapter):
    def __init__(
        self,
        name: str = "fenzo",
        url: str = "",
        input_selector: str = "textarea",
        send_selector: str | None = None,
        response_selector: str = "[data-role='assistant'], .assistant-message",
        ready_selector: str | None = None,
        settle_ms: int = 1500,
        timeout_ms: int = 60000,
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

            # Wait for a new assistant bubble to appear, then let streaming settle.
            page.wait_for_function(
                "([sel, n]) => document.querySelectorAll(sel).length > n",
                arg=[self.response_selector, before],
            )
            page.wait_for_timeout(self.settle_ms)

            bubbles = page.locator(self.response_selector)
            text = bubbles.nth(bubbles.count() - 1).inner_text().strip()
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
