"""Command-line entry point for the Fenzo benchmarker pipeline.

    python -m benchmarker record "How do I center a div?" --tags css
    python -m benchmarker run                 # steps 2-4 over recorded queries
    python -m benchmarker report              # rebuild report from the log

``run`` executes the full loop: give queries to all models, compare, update
logs, and render a report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os

from .comparator import Comparison, make_judge
from .config import load_config
from .envfile import load_dotenv
from .logstore import log_comparisons, log_responses
from .models import build_adapter
from .recorder import add_query, load_queries
from .report import write_report
from .runner import group_by_query, new_run_id, run_batch

# Which env var each backend type needs for auth.
_KEY_FOR_TYPE = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _preflight_keys(cfg) -> list[str]:
    """Return human-readable messages for any missing API keys the config needs."""
    needed: dict[str, str] = {}
    specs = list(cfg.models) + [cfg.judge]
    for spec in specs:
        env_var = _KEY_FOR_TYPE.get(spec.get("type"))
        if env_var and not os.environ.get(env_var):
            needed[env_var] = spec.get("name", spec.get("type"))
    return [f"{var} (needed by '{who}')" for var, who in needed.items()]


def _cmd_record(args, cfg) -> int:
    q = add_query(
        cfg.path("queries"),
        prompt=args.prompt,
        tags=args.tags or [],
        query_id=args.id,
    )
    print(f"recorded {q.id}: {q.prompt}")
    return 0


def _cmd_run(args, cfg) -> int:
    queries = load_queries(cfg.path("queries"))
    if args.query_id:
        queries = [q for q in queries if q.id in set(args.query_id)]
    if not queries:
        print("no queries to run — record some first", file=sys.stderr)
        return 1

    # Preflight: fail fast with a clear message if a real backend lacks its key,
    # rather than a cryptic SDK error mid-run. --no-llm-judge exempts the judge.
    missing = _preflight_keys(cfg)
    if args.no_llm_judge:
        missing = [m for m in missing if not m.startswith("ANTHROPIC_API_KEY (needed by 'judge')")]
    if missing:
        print("missing API key(s):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "add them to a local .env file (copy .env.example) — never commit it.",
            file=sys.stderr,
        )
        return 1

    adapters = [build_adapter(spec) for spec in cfg.models]
    subject = cfg.subject
    judge = make_judge(cfg.judge, allow_llm=not args.no_llm_judge)
    run_id = new_run_id()
    repeats = max(1, args.repeats)
    print(
        f"running {len(queries)} queries x {len(adapters)} models x {repeats} "
        f"repeat(s); subject={subject}; judge={judge.name}; run_id={run_id}"
    )

    try:
        # Step 2 — give queries to all models.
        responses = run_batch(queries, adapters, repeats=repeats, run_id=run_id)
        n_ok = sum(1 for r in responses if r.ok)
        print(f"collected {len(responses)} responses ({n_ok} ok)")

        # Step 4a — persist raw responses.
        log_responses(cfg.path("responses_log"), responses)

        # Step 3 — compare per query.
        prompts = {q.id: q.prompt for q in queries}
        grouped = group_by_query(responses)
        comparisons: list[Comparison] = []
        for qid, resps in grouped.items():
            comparisons.append(judge.compare(prompts[qid], subject, resps))
            print(f"  judged {qid}: fenzo verdict = {comparisons[-1].subject_verdict}")

        # Step 4b — persist comparisons.
        log_comparisons(cfg.path("comparisons_log"), comparisons)
    finally:
        for a in adapters:
            a.close()

    out = write_report(comparisons, subject, cfg.path("reports_dir"))
    print(f"wrote report: {out}")
    return 0


def _cmd_report(args, cfg) -> int:
    from .logstore import load_comparisons
    from .comparator import Comparison, ModelScore

    rows = load_comparisons(cfg.path("comparisons_log"))
    if not rows:
        print("no comparisons logged yet — run the pipeline first", file=sys.stderr)
        return 1

    # Default to the most recent run so re-running the same queries doesn't
    # double-count in the aggregate. "Most recent" = the run of the last row in
    # the append-only log, not a string max of run_ids (ids aren't temporally
    # ordered — see runner.new_run_id). `--all` or `--run-id` override.
    logged_run_ids = [r.get("run_id", "") for r in rows if r.get("run_id")]
    if args.run_id:
        rows = [r for r in rows if r.get("run_id") == args.run_id]
    elif not args.all and logged_run_ids:
        latest = logged_run_ids[-1]
        rows = [r for r in rows if r.get("run_id") == latest]
        print(f"reporting on latest run {latest} (use --all for full history)")

    if not rows:
        print("no comparisons matched the selected run", file=sys.stderr)
        return 1

    comparisons = [
        Comparison(
            query_id=r["query_id"],
            prompt=r["prompt"],
            subject=r["subject"],
            judged_by=r["judged_by"],
            scores=[ModelScore(**s) for s in r["scores"]],
            subject_verdict=r["subject_verdict"],
            subject_strengths=r["subject_strengths"],
            subject_weaknesses=r["subject_weaknesses"],
            summary=r["summary"],
            run_id=r.get("run_id", ""),
            rounds=r.get("rounds", 1),
            created_at=r.get("created_at", ""),
        )
        for r in rows
    ]
    out = write_report(comparisons, cfg.subject, cfg.path("reports_dir"))
    print(f"wrote report from {len(comparisons)} logged comparisons: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmarker", description=__doc__)
    p.add_argument("-c", "--config", help="path to config.yaml/.json (default: built-in mock config)")
    sub = p.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record a query")
    rec.add_argument("prompt")
    rec.add_argument("--tags", nargs="*", default=[])
    rec.add_argument("--id", default=None)
    rec.set_defaults(func=_cmd_record)

    run = sub.add_parser("run", help="run models, compare, log, report")
    run.add_argument("--query-id", nargs="*", help="limit to these query ids")
    run.add_argument("--no-llm-judge", action="store_true", help="force the heuristic judge")
    run.add_argument(
        "--repeats", type=int, default=1,
        help="samples per (query, model) for variance (default 1)",
    )
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="rebuild the report from the comparisons log")
    rep.add_argument("--all", action="store_true", help="aggregate the full log history")
    rep.add_argument("--run-id", help="report on a specific run_id")
    rep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    # Load API keys from a local .env (gitignored) so runs are plug-and-go.
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
