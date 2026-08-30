"""Step 3 — Compare results.

For each query we now hold every model's answer (possibly several samples each).
This step turns that into a judgement focused on the question the task actually
asks: *how are Fenzo's responses good and bad compared to the other LLMs?*

Two judges are available:

* ``LLMJudge`` — an Anthropic model reads the query and all answers blind
  (labels are anonymised so the judge can't play favourites), scores each on a
  rubric, and writes an explicit strengths/weaknesses assessment of the subject
  (Fenzo) relative to the field. This is the real signal.
* ``HeuristicJudge`` — a no-network fallback that ranks on objective proxies
  (answered vs errored, response length, latency). Keeps the pipeline runnable
  offline and in CI; clearly labelled so nobody mistakes it for quality.

Both share a base :class:`Judge` that handles **repeats**: when the runner
collected N samples per model, each sample index is judged as its own round and
the per-model scores are averaged, with the standard deviation reported so the
consumer can see how noisy the verdict is. Blind ordering uses a deterministic
per-round seed, so a re-run reproduces the same anonymisation.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import ModelResponse

JUDGE_SYSTEM = (
    "You are a rigorous, impartial evaluator of AI assistant responses. "
    "The answers may be explanations or short lessons on the user's topic; "
    "judge each on its merits for someone trying to learn: correctness, "
    "relevance, completeness, clarity, structure, and usefulness. "
    "Judge substance, not length — do not reward an answer for being longer, "
    "and do not penalise a concise answer that fully addresses the topic. "
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
    score: float  # mean across rounds
    rationale: str
    latency_s: float  # mean across samples
    length_chars: int  # mean across samples
    ok: bool
    rank: int = 0
    score_stdev: float = 0.0
    samples: int = 1


@dataclass
class RoundResult:
    """One judge pass over one sample per model."""

    scores: dict[str, float]  # model -> score
    rationales: dict[str, str]
    subject_strengths: list[str]
    subject_weaknesses: list[str]
    summary: str


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
    run_id: str = ""
    rounds: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(r: ModelResponse) -> tuple[float, int]:
    return r.latency_s, len(r.text or "")


def _rounds_by_repeat(responses: list[ModelResponse]) -> list[dict[str, ModelResponse]]:
    """Split responses into rounds keyed by repeat index: [{model: response}]."""
    by_repeat: dict[int, dict[str, ModelResponse]] = {}
    for r in responses:
        by_repeat.setdefault(r.repeat, {})[r.model] = r
    return [by_repeat[k] for k in sorted(by_repeat)]


class Judge:
    """Base class: runs one round per repeat, then aggregates across rounds."""

    name = "judge"

    def _round(
        self, prompt: str, subject: str, resp_by_model: dict[str, ModelResponse]
    ) -> RoundResult:
        raise NotImplementedError

    def compare(
        self, prompt: str, subject: str, responses: list[ModelResponse]
    ) -> Comparison:
        rounds = _rounds_by_repeat(responses)
        run_id = responses[0].run_id if responses else ""
        query_id = responses[0].query_id if responses else ""

        round_results = [self._round(prompt, subject, rd) for rd in rounds]

        # Aggregate numeric scores per model across rounds.
        per_model_scores: dict[str, list[float]] = {}
        rationales: dict[str, str] = {}
        for rr in round_results:
            for model, sc in rr.scores.items():
                per_model_scores.setdefault(model, []).append(sc)
                rationales.setdefault(model, rr.rationales.get(model, ""))

        # Aggregate objective metrics across every sample of each model.
        metrics: dict[str, list[tuple[float, int, bool]]] = {}
        for r in responses:
            latency, length = _metrics(r)
            metrics.setdefault(r.model, []).append((latency, length, r.ok))

        scores: list[ModelScore] = []
        for model, vals in per_model_scores.items():
            m = metrics.get(model, [(0.0, 0, False)])
            scores.append(
                ModelScore(
                    model=model,
                    score=round(statistics.fmean(vals), 2),
                    score_stdev=round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
                    samples=len(vals),
                    rationale=rationales.get(model, ""),
                    latency_s=round(statistics.fmean([x[0] for x in m]), 3),
                    length_chars=int(statistics.fmean([x[1] for x in m])),
                    ok=all(x[2] for x in m),
                )
            )
        _rank(scores)

        # Qualitative narrative: from the first round; verdict recomputed from
        # the aggregated means so it stays consistent with the reported scores.
        first = round_results[0] if round_results else RoundResult({}, {}, [], [], "")
        strengths = _dedupe([s for rr in round_results for s in rr.subject_strengths])
        weaknesses = _dedupe([w for rr in round_results for w in rr.subject_weaknesses])

        return Comparison(
            query_id=query_id,
            prompt=prompt,
            subject=subject,
            judged_by=self.name,
            scores=scores,
            subject_verdict=_subject_verdict(scores, subject),
            subject_strengths=strengths or first.subject_strengths,
            subject_weaknesses=weaknesses or first.subject_weaknesses,
            summary=first.summary,
            run_id=run_id,
            rounds=len(round_results),
        )


class HeuristicJudge(Judge):
    """Offline ranking on objective proxies — not a quality measure."""

    name = "heuristic"

    def _round(
        self, prompt: str, subject: str, resp_by_model: dict[str, ModelResponse]
    ) -> RoundResult:
        scores: dict[str, float] = {}
        rationales: dict[str, str] = {}
        for model, r in resp_by_model.items():
            latency, length = _metrics(r)
            if not r.ok or length == 0:
                s = 0.0
            else:
                length_fit = 1.0 - min(abs(length - 600) / 600, 1.0)
                latency_fit = 1.0 - min(latency / 30.0, 1.0)
                s = round(4 + 5 * length_fit + latency_fit, 2)
            scores[model] = s
            rationales[model] = "objective proxy (length fit + answered + latency)"
        return RoundResult(
            scores=scores,
            rationales=rationales,
            subject_strengths=["(heuristic judge — enable the LLM judge for real analysis)"],
            subject_weaknesses=[],
            summary="Heuristic ranking on objective proxies; no quality judgement.",
        )


class LLMJudge(Judge):
    """Anthropic-backed blind evaluation of one sample per model per round."""

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

    def _round(
        self, prompt: str, subject: str, resp_by_model: dict[str, ModelResponse]
    ) -> RoundResult:
        responses = list(resp_by_model.values())
        # Deterministic anonymisation: seed by (query, repeat) so a re-run
        # produces the same A/B/C ordering — reproducible, still not vendor-order.
        seed = f"{responses[0].query_id}:{responses[0].repeat}" if responses else "0"
        rng = random.Random(seed)
        shuffled = list(responses)
        rng.shuffle(shuffled)
        labels = {r.model: chr(ord("A") + i) for i, r in enumerate(shuffled)}
        subject_label = labels.get(subject)

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

        label_to_model = {v: k for k, v in labels.items()}
        judged = {s["label"]: s for s in data["scores"]}
        scores: dict[str, float] = {}
        rationales: dict[str, str] = {}
        for label, model in label_to_model.items():
            js = judged.get(label, {"score": 0, "rationale": "no score returned"})
            scores[model] = float(js["score"])
            rationales[model] = js["rationale"]

        return RoundResult(
            scores=scores,
            rationales=rationales,
            subject_strengths=data["subject_strengths"],
            subject_weaknesses=data["subject_weaknesses"],
            summary=data["summary"],
        )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


def make_judge(judge_cfg: dict[str, Any], allow_llm: bool = True) -> Judge:
    """Build an LLM judge if configured and importable, else the heuristic one."""
    if allow_llm and judge_cfg.get("type") == "anthropic":
        try:
            import anthropic  # noqa: F401

            return LLMJudge(model=judge_cfg.get("model", "claude-opus-5"))
        except ImportError:
            pass
    return HeuristicJudge()
