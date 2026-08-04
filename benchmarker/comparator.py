"""Step 3 — Compare results.

For each query we now hold every model's answer. This step turns that into a
judgement focused on the question the task actually asks: *how are Fenzo's
responses good and bad compared to the other LLMs?*

Two judges are available:

* ``LLMJudge`` — an Anthropic model reads the query and all answers blind
  (labels are anonymised so the judge can't play favourites), scores each on a
  rubric, and writes an explicit strengths/weaknesses assessment of the subject
  (Fenzo) relative to the field. This is the real signal.
* ``HeuristicJudge`` — a no-network fallback that ranks on objective proxies
  (answered vs errored, response length, latency). Keeps the pipeline runnable
  offline and in CI; clearly labelled so nobody mistakes it for quality.

Both return the same :class:`Comparison` shape so the log/report code is
judge-agnostic.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import ModelResponse

JUDGE_SYSTEM = (
    "You are a rigorous, impartial evaluator of AI assistant responses. "
    "You judge only on the merits of each answer to the user's query: "
    "correctness, relevance, completeness, clarity, and usefulness. "
    "You do not know which product produced which answer and must not guess."
)

# JSON schema the LLM judge is constrained to (structured outputs).
_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["label", "score", "rationale"],
            },
        },
        "subject_strengths": {"type": "array", "items": {"type": "string"}},
        "subject_weaknesses": {"type": "array", "items": {"type": "string"}},
        "subject_verdict": {"type": "string", "enum": ["better", "comparable", "worse"]},
        "summary": {"type": "string"},
    },
    "required": [
        "scores",
        "subject_strengths",
        "subject_weaknesses",
        "subject_verdict",
        "summary",
    ],
}


@dataclass
class ModelScore:
    model: str
    score: float
    rationale: str
    latency_s: float
    length_chars: int
    ok: bool
    rank: int = 0


@dataclass
class Comparison:
    query_id: str
    prompt: str
    subject: str
    judged_by: str  # "llm:<model>" or "heuristic"
    scores: list[ModelScore]
    subject_verdict: str  # better | comparable | worse
    subject_strengths: list[str]
    subject_weaknesses: list[str]
    summary: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _metrics(r: ModelResponse) -> tuple[float, int]:
    return r.latency_s, len(r.text or "")


class HeuristicJudge:
    """Offline ranking on objective proxies — not a quality measure."""

    name = "heuristic"

    def compare(
        self, prompt: str, subject: str, responses: list[ModelResponse]
    ) -> Comparison:
        scores: list[ModelScore] = []
        for r in responses:
            latency, length = _metrics(r)
            # Reward answering at all and a "reasonable" length (200-1500 chars);
            # mild penalty for latency. Purely a placeholder signal.
            if not r.ok or length == 0:
                s = 0.0
            else:
                length_fit = 1.0 - min(abs(length - 600) / 600, 1.0)
                latency_fit = 1.0 - min(latency / 30.0, 1.0)
                s = round(4 + 5 * length_fit + latency_fit, 2)
            scores.append(
                ModelScore(
                    model=r.model,
                    score=s,
                    rationale="objective proxy (length fit + answered + latency)",
                    latency_s=round(latency, 3),
                    length_chars=length,
                    ok=r.ok,
                )
            )
        _rank(scores)
        verdict = _subject_verdict(scores, subject)
        return Comparison(
            query_id=responses[0].query_id if responses else "",
            prompt=prompt,
            subject=subject,
            judged_by=self.name,
            scores=scores,
            subject_verdict=verdict,
            subject_strengths=["(heuristic judge — enable the LLM judge for real analysis)"],
            subject_weaknesses=[],
            summary="Heuristic ranking on objective proxies; no quality judgement.",
        )


class LLMJudge:
    """Anthropic-backed blind evaluation of every answer for one query."""

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 2048) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"llm:{model}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def compare(
        self, prompt: str, subject: str, responses: list[ModelResponse]
    ) -> Comparison:
        # Anonymise: shuffle and label A/B/C so the judge can't infer the vendor.
        shuffled = list(responses)
        random.shuffle(shuffled)
        labels = {r.model: chr(ord("A") + i) for i, r in enumerate(shuffled)}
        subject_label = labels[subject] if subject in labels else None

        blocks = []
        for r in shuffled:
            body = r.text if r.ok else f"[ERROR: {r.error}]"
            blocks.append(f"### Answer {labels[r.model]}\n{body}")
        answers_text = "\n\n".join(blocks)

        user = (
            f"User query:\n{prompt}\n\n"
            f"Candidate answers (anonymised):\n\n{answers_text}\n\n"
            f"Score each answer from 1 (poor) to 10 (excellent) with a one-line "
            f"rationale. Then assess Answer {subject_label} specifically: list its "
            f"concrete strengths and weaknesses relative to the other answers, give "
            f"a verdict (better / comparable / worse than the field), and a short "
            f"summary. Refer to answers only by their letter labels."
        )

        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        data = json.loads(text)

        # Map anonymised labels back to model names.
        label_to_model = {v: k for k, v in labels.items()}
        by_model = {r.model: r for r in responses}
        scores: list[ModelScore] = []
        judged = {s["label"]: s for s in data["scores"]}
        for label, model in label_to_model.items():
            r = by_model[model]
            latency, length = _metrics(r)
            js = judged.get(label, {"score": 0, "rationale": "no score returned"})
            scores.append(
                ModelScore(
                    model=model,
                    score=float(js["score"]),
                    rationale=js["rationale"],
                    latency_s=round(latency, 3),
                    length_chars=length,
                    ok=r.ok,
                )
            )
        _rank(scores)

        return Comparison(
            query_id=responses[0].query_id if responses else "",
            prompt=prompt,
            subject=subject,
            judged_by=self.name,
            scores=scores,
            subject_verdict=data["subject_verdict"],
            subject_strengths=data["subject_strengths"],
            subject_weaknesses=data["subject_weaknesses"],
            summary=data["summary"],
        )


def _rank(scores: list[ModelScore]) -> None:
    for i, s in enumerate(sorted(scores, key=lambda x: x.score, reverse=True), 1):
        s.rank = i


def _subject_verdict(scores: list[ModelScore], subject: str) -> str:
    by_model = {s.model: s for s in scores}
    if subject not in by_model:
        return "comparable"
    subj = by_model[subject].score
    others = [s.score for s in scores if s.model != subject]
    if not others:
        return "comparable"
    avg = sum(others) / len(others)
    if subj > avg + 0.75:
        return "better"
    if subj < avg - 0.75:
        return "worse"
    return "comparable"


def make_judge(judge_cfg: dict[str, Any], allow_llm: bool = True):
    """Build an LLM judge if configured and importable, else the heuristic one."""
    if allow_llm and judge_cfg.get("type") == "anthropic":
        try:
            import anthropic  # noqa: F401

            return LLMJudge(model=judge_cfg.get("model", "claude-opus-5"))
        except ImportError:
            pass
    return HeuristicJudge()
