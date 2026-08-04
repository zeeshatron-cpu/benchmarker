"""End-to-end tests that exercise the pipeline with mock adapters only.

No network, no API keys — these must pass in CI. They cover each of the four
steps: record, run, compare, log/report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarker.comparator import HeuristicJudge, make_judge
from benchmarker.config import load_config
from benchmarker.logstore import (
    load_comparisons,
    log_comparisons,
    log_responses,
)
from benchmarker.models import build_adapter
from benchmarker.recorder import add_query, load_queries
from benchmarker.report import build_markdown
from benchmarker.runner import group_by_query, run_batch


def _mock_config(tmp_path: Path) -> dict:
    return {
        "subject": "fenzo",
        "judge": {"name": "judge", "type": "anthropic", "model": "claude-opus-5"},
        "paths": {
            "queries": str(tmp_path / "queries.jsonl"),
            "responses_log": str(tmp_path / "responses.jsonl"),
            "comparisons_log": str(tmp_path / "comparisons.jsonl"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "models": [
            {"name": "fenzo", "type": "mock", "persona": "balanced"},
            {"name": "claude-opus-5", "type": "mock", "persona": "verbose"},
            {"name": "gpt-4o", "type": "mock", "persona": "concise"},
        ],
    }


def test_record_and_load(tmp_path):
    qpath = tmp_path / "queries.jsonl"
    q1 = add_query(qpath, "First question?", tags=["a"])
    q2 = add_query(qpath, "Second question?")
    assert q1.id == "q-0001"
    assert q2.id == "q-0002"
    loaded = load_queries(qpath)
    assert [q.id for q in loaded] == ["q-0001", "q-0002"]
    assert loaded[0].tags == ["a"]


def test_run_fans_out_across_all_models(tmp_path):
    cfg = load_config(_write_json(tmp_path, _mock_config(tmp_path)))
    add_query(cfg.path("queries"), "How do I center a div?")
    add_query(cfg.path("queries"), "What is a thread?")

    adapters = [build_adapter(s) for s in cfg.models]
    queries = load_queries(cfg.path("queries"))
    responses = run_batch(queries, adapters)

    # 2 queries x 3 models.
    assert len(responses) == 6
    assert all(r.ok for r in responses)
    grouped = group_by_query(responses)
    assert set(grouped) == {"q-0001", "q-0002"}
    assert all(len(v) == 3 for v in grouped.values())


def test_mock_responses_are_deterministic(tmp_path):
    a = build_adapter({"name": "fenzo", "type": "mock", "persona": "balanced"})
    r1 = a.ask("q-1", "same prompt")
    r2 = a.ask("q-1", "same prompt")
    assert r1.text == r2.text


def test_heuristic_judge_ranks_and_logs(tmp_path):
    cfg = load_config(_write_json(tmp_path, _mock_config(tmp_path)))
    add_query(cfg.path("queries"), "Explain microservices trade-offs.")
    adapters = [build_adapter(s) for s in cfg.models]
    queries = load_queries(cfg.path("queries"))
    responses = run_batch(queries, adapters)

    judge = HeuristicJudge()
    comp = judge.compare(queries[0].prompt, cfg.subject, responses)

    assert comp.query_id == "q-0001"
    assert comp.subject == "fenzo"
    assert comp.judged_by == "heuristic"
    ranks = sorted(s.rank for s in comp.scores)
    assert ranks == [1, 2, 3]
    assert comp.subject_verdict in {"better", "comparable", "worse"}

    log_responses(cfg.path("responses_log"), responses)
    n = log_comparisons(cfg.path("comparisons_log"), [comp])
    assert n == 1
    rows = load_comparisons(cfg.path("comparisons_log"))
    assert len(rows) == 1
    assert rows[0]["query_id"] == "q-0001"


def test_make_judge_falls_back_to_heuristic_without_llm():
    # Force the heuristic path (allow_llm=False) regardless of environment.
    judge = make_judge({"type": "anthropic", "model": "claude-opus-5"}, allow_llm=False)
    assert isinstance(judge, HeuristicJudge)


def test_report_mentions_subject_and_verdicts(tmp_path):
    cfg = load_config(_write_json(tmp_path, _mock_config(tmp_path)))
    add_query(cfg.path("queries"), "Center a div?")
    add_query(cfg.path("queries"), "Thread vs process?")
    adapters = [build_adapter(s) for s in cfg.models]
    queries = load_queries(cfg.path("queries"))
    responses = run_batch(queries, adapters)
    grouped = group_by_query(responses)
    judge = HeuristicJudge()
    prompts = {q.id: q.prompt for q in queries}
    comps = [judge.compare(prompts[qid], "fenzo", rs) for qid, rs in grouped.items()]

    md = build_markdown(comps, "fenzo")
    assert "Fenzo response comparison report" in md
    assert "fenzo" in md
    assert "Average judge score by model" in md
    assert "Queries judged: **2**" in md


def test_bad_adapter_type_raises():
    with pytest.raises(ValueError):
        build_adapter({"name": "x", "type": "does-not-exist"})


def _write_json(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p
