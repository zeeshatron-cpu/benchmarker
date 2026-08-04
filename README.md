# Fenzo response-comparison benchmarker

An automated pipeline that evaluates **how Fenzo AI's responses are good and bad
compared to other LLMs** given the same query. It records queries, sends each one
to Fenzo and a set of comparison models, has an impartial LLM judge compare the
answers, and appends the verdicts to durable logs plus a human-readable report.

Because Fenzo AI is a **web app with no public API**, the pipeline drives its UI
with a headless browser; the other models are reached through their APIs. Every
backend hides behind one small adapter interface, so adding or swapping a model
is a config edit, not a code change.

## The four steps

```
 ┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ 1. RECORD    │   │ 2. RUN                     │   │ 3. COMPARE       │   │ 4. LOG        │
 │  queries     │──▶│  same query → all models   │──▶│  blind LLM judge │──▶│  JSONL logs   │
 │  (JSONL)     │   │  Fenzo (web) + Claude + GPT│   │  scores + verdict│   │  + MD report  │
 └──────────────┘   └───────────────────────────┘   └──────────────────┘   └──────────────┘
     recorder.py         runner.py + models/            comparator.py        logstore.py
                                                                             report.py
```

1. **Record queries** — `recorder.py` appends queries to `queries/queries.jsonl`.
2. **Give queries to all models** — `runner.py` fans each query out across every
   configured adapter (`models/`) concurrently, capturing text, latency, tokens,
   and errors per model.
3. **Compare results** — `comparator.py` runs a blind, rubric-based LLM judge
   (answers anonymised A/B/C so it can't favour a vendor). It scores every answer
   and produces an explicit *Fenzo strengths / weaknesses / verdict* assessment.
   An offline heuristic judge is the fallback when no API key is present.
4. **Update logs** — `logstore.py` appends raw responses and judgements to
   append-only JSONL; `report.py` rolls the history into a Markdown report.

## Quick start (zero setup — all mock, no keys, no network)

```bash
python -m benchmarker record "How do I center a div?" --tags css
python -m benchmarker record "Explain processes vs threads."
python -m benchmarker run
```

With no config file the pipeline uses a built-in **all-mock** line-up (a mock
"fenzo" plus two mock competitors) and the heuristic judge, so you can watch the
whole flow immediately. The report lands in `reports/`.

## Real run

1. `pip install -r requirements.txt` (plus `openai` / `playwright` if used).
2. `cp config.example.yaml config.yaml` and edit:
   - Point the **`fenzo`** adapter at the real chat URL and fill in the CSS
     selectors for its input box / send button / assistant bubbles. Capture a
     logged-in session once with
     `playwright codegen --save-storage=fenzo_auth.json https://app.fenzo.ai`.
   - Keep or adjust the **`claude-opus-5`** / **`gpt-4o`** comparison adapters.
   - The **judge** uses Claude via the `anthropic` SDK (`ANTHROPIC_API_KEY` or an
     `ant auth login` profile).
3. Run:

```bash
python -m benchmarker -c config.yaml record "What are microservices trade-offs?"
python -m benchmarker -c config.yaml run
python -m benchmarker -c config.yaml report   # rebuild report from full history
```

Force the offline judge for a dry run: `run --no-llm-judge`.

## Layout

```
benchmarker/
  config.py        # load YAML/JSON config (built-in mock default)
  recorder.py      # step 1: record queries
  runner.py        # step 2: fan queries out across adapters
  comparator.py    # step 3: LLM judge + heuristic fallback
  logstore.py      # step 4a: append-only JSONL logs
  report.py        # step 4b: Markdown report
  cli.py           # ties the steps together
  models/
    base.py        # ModelAdapter interface (ask -> ModelResponse)
    mock.py        # deterministic offline model
    anthropic_model.py  # Claude via the anthropic SDK
    openai_model.py     # GPT via the openai SDK
    fenzo_web.py        # Fenzo AI via Playwright browser automation
config.example.yaml
queries/queries.sample.jsonl
tests/test_pipeline.py   # end-to-end, mock-only, runs in CI
```

## Design notes

- **Adapter pattern** keeps Fenzo's web automation, API models, and mocks
  interchangeable. `models/__init__.py` maps a config `type` to a class.
- **Blind judging** anonymises answers before the judge sees them, and uses
  structured outputs (`output_config.format`) so the verdict is machine-parseable.
- **Fault isolation** — one flaky model (or a Fenzo selector that moved) yields an
  error on its own response and never aborts the batch.
- **Append-only logs** make history cumulative and auditable; the report is always
  regenerable from `logs/comparisons.jsonl`.

## Extending

Add a model: write a `ModelAdapter` subclass, register it in
`models/__init__.py`, and add a config entry. Swap the judge model, add metrics
to `comparator.ModelScore`, or emit an HTML/artifact report from the same
`Comparison` objects.
