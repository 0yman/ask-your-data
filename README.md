# Port Analyst Agent

A tool-calling agent that answers analytical questions about container
terminal operations by writing and executing **read-only SQL** against a
DuckDB star schema — with guardrails that make "read-only" a control rather
than a request, and an evaluation harness that measures whether the answers
are actually right.

```
question ──▶ ┌──────────────────────────────────────────────┐
             │ agent loop (max 10 steps)                    │
             │                                              │
             │   model ──▶ tool call ──▶ guardrails ──▶ DB   │
             │     ▲                          │             │
             │     └──── result or error ─────┘             │
             │          (a failure is a message,            │
             │           not an exception)                  │
             └──────────────────────────────────────────────┘
                              │
                              ▼
                   answer + full trace + SQL
```

---

## Quickstart

```bash
pip install -r requirements-dev.txt
python scripts/build_warehouse.py      # deterministic; ~18s
python -m pytest                       # 102 tests, offline, no API key

cp .env.example .env                   # add a free key from
                                       # https://aistudio.google.com/apikey
PYTHONPATH=src python -m agent.cli --trace \
  "Which berth underperforms most relative to how many cranes it has?"
```

```
--- step 1 ---  list_tables() -> ok
--- step 2 ---  describe_table(table='dim_berth') -> ok
--- step 3 ---  describe_table(table='fact_vessel_call') -> ok
--- step 4 ---  run_sql(sql='SELECT b.berth_code, b.crane_count, AVG(vc.moves_per_hour)…') -> ok
--- step 5 ---  final_answer(...) -> ok

The berth that underperforms most relative to its crane count is B07. It has
3 cranes and an average productivity of 18.19 moves per hour, giving the
lowest productivity per crane at 6.06.
```

Or `docker compose up --build`. Without a key the service still starts:
`/schema` and `/sql` work, `/ask` returns 503, and `/health` says `degraded`.

---

## The data

A star schema for a container terminal — two fact tables, five conformed
dimensions, three years of daily activity:

| Table | Grain | Rows |
|---|---|---|
| `fact_vessel_call` | one vessel visit to one berth | 4,821 |
| `fact_container_movement` | one customer/cargo group within a call | 16,923 |
| `dim_date` / `dim_vessel` / `dim_berth` / `dim_customer` / `dim_cargo_type` | — | 1,096 / 60 / 8 / 12 / 7 |

Generated from a fixed seed, so a clean clone reproduces the exact database
the evaluation's gold queries were written against. The data is synthetic but
not random: it carries patterns that are *findable*, which is what makes an
analytical question worth asking.

| Planted pattern | What the data shows |
|---|---|
| Seasonal throughput | August peak 204,832 TEU vs January trough 115,138 |
| Customs holds delay release | 10.27 days average dwell vs 5.06 without |
| Hazardous cargo is slowest | 9.88 days vs 3.77 for refrigerated |
| Productivity scales with cranes | B06 (6 cranes) 44.4 moves/hr, B08 (1 crane) 7.8 |
| One berth is a laggard | B07 has 3 cranes but 18.2 moves/hr, where B02's 3 cranes give 22.2 |

That last one is the interesting one: it is invisible in raw productivity
rankings and only appears when you normalise by crane count.

---

## Guardrails

The agent hands a language model a database connection. Everything that
reaches the model — the question, a column name, a row echoed back into
context — is untrusted input that can try to steer the SQL it writes. Telling
the model "only write SELECT statements" is a request. These are the controls:

**Layer 1 — every statement is parsed with `sqlglot` before execution.**

| Blocked | Why |
|---|---|
| Anything that is not a single SELECT | `SELECT 1; DROP TABLE x` is the classic stacked-statement injection |
| DML/DDL anywhere in the tree | an `INSERT` hidden behind a legitimate-looking `WITH` clause |
| `read_csv`, `read_parquet`, `read_json`, `glob` | DuckDB's file functions are the real exfiltration path — they turn a read-only SQL endpoint into arbitrary local file access |
| `ATTACH`, `INSTALL`, `LOAD`, `COPY`, `PRAGMA` | remote databases, extension loading, writing files out |
| Missing `LIMIT` | added rather than rejected: an accidental cross join should not return ten million rows into a context window |

**Layer 2 — the connection is opened `read_only=True`.** Belt and braces on
purpose: layer 1 knows *why* a query is wrong and can tell the agent so it
can correct itself, but it is code I wrote and could have a gap. Layer 2 is
enforced by DuckDB and has no opinions.

A rejection is returned to the model as a message, never raised — that is what
makes self-correction possible at all.

---

## Results

22 questions: 20 with hand-verified gold SQL, 2 that the warehouse
deliberately **cannot** answer. Model: `gemini-3.1-flash-lite` on the free
tier.

| Metric | Value |
|---|---|
| **Answer figure coverage** (do the right numbers reach the user?) | **0.944** |
| Gold coverage (does the result contain every gold fact?) | 0.900 |
| Execution accuracy (strict result-set equality) | 0.750 |
| Correctly declined on unanswerable questions | **1.000** |

### Why three numbers and not one

The gap between 0.750 and 0.944 is not the agent improving — it is three
different definitions of "correct", and the spread is the finding.

Strict execution accuracy compares result sets exactly. It turns out to
measure the *shape* of a query as much as its correctness. Three of the five
strict failures answered the question perfectly:

| Question | What the agent did | Why strict equality rejected it |
|---|---|---|
| q12 — which vessel type waits longest? | returned all four vessel types, ranked, then named Post-Panamax at 6.73 hours | gold used `LIMIT 1`; row counts differ |
| q14 — which berth underperforms per crane? | returned all eight berths with the intermediate columns | gold returned one row, two columns |
| q15 — share of calls waiting over 12h | kept `total_calls` and `long_wait_calls` alongside the percentage | extra columns |

Reporting only strict accuracy would understate the agent. Reporting only the
relaxed metric would overstate it — a query returning everything trivially
"covers" everything. So all three are reported, and the end-to-end one
(is the right number in the prose the user reads?) is the headline, because
that is what a user actually receives.

### The two genuine misses

**q10 — "What percentage of container movements involve hazardous cargo?"**
The gold query counts rows (4.99%). The agent weighted by `container_count`
(4.78%). Re-reading the question, "movements" means rows, so the agent is
wrong — but the question is ambiguous enough that a human analyst could have
made the same call. A sharper question would say "what share of movement
records".

**q16 — year-over-year TEU change.** The agent queried yearly totals and
computed the deltas in prose instead of using a window function. Every figure
it reported was correct; the result set simply lacked the `LAG` column the
gold query produced.

### By difficulty

| Difficulty | Strict execution accuracy |
|---|---|
| easy | 1.000 |
| medium | 0.714 |
| hard | 0.625 |

### Behaviour and cost

| Metric | Value |
|---|---|
| Mean steps per question | 3.96 |
| Mean queries per question | 1.00 |
| Self-corrections triggered | 0 |
| Mean latency | 28.8s |
| Total tokens | 121,086 in / 4,638 out |
| Transient 503/429 absorbed by retries | 36 |

Two things worth being explicit about:

**Zero self-corrections.** The recovery path never fired, because the schema
is in the system prompt and the model never guessed a column name wrong. The
machinery is real and covered by tests — a scripted model that writes a bad
query, reads the error, inspects the table and fixes it — but this run does
not prove it works against a live model.

### What the schema in the prompt is actually worth

The whole schema is sent on every call. That is an obvious thing to question,
so `--no-schema-prompt` withholds it and forces the agent to discover the
schema through `list_tables` and `describe_table` instead. Same 22 questions,
same model:

| Metric | Schema in prompt | Discovered via tools |
|---|---|---|
| Answer figure coverage | 0.944 | 0.944 |
| Execution accuracy | 0.750 | 0.750 |
| Gold coverage | 0.900 | 0.900 |
| **Correctly declined** | **1.000** | **0.500** |
| Mean steps | 3.96 | 4.05 |
| Mean queries | 1.00 | 1.14 |
| Mean latency | 28.8s | 34.1s |
| **Total prompt tokens** | **121,086** | **127,607** |

Two results I did not expect.

**It costs nothing — it saves.** Withholding the schema *increased* total
prompt tokens by 5%. The discovery calls and their results accumulate in the
conversation faster than the schema block would have cost, and each one is
resent on every subsequent turn. The intuition that a smaller system prompt
is cheaper is wrong here, and only measurement showed it.

**On answerable questions it changes nothing; on unanswerable ones it changes
everything.** Accuracy is identical to three decimal places — the model
discovers the schema perfectly well on its own. But decline accuracy halved.
Asked for stevedoring revenue that does not exist, the agent without the
schema spent all ten steps hunting for a column to hold it and terminated on
the step budget, returning *"I ran out of steps"* instead of *"that data is
not in the warehouse"*.

It did not hallucinate a number — the step budget caught it, which is what a
step budget is for. But knowing quickly that something is **absent** requires
seeing the whole schema at once, and no amount of exploring one table at a
time supplies that.

**36 transient failures.** The free tier returned 503 "high demand" and 429
throughout; every one was absorbed by exponential backoff with jitter, and no
question failed because of it. That is what the retry layer is for.

### Model availability is worth measuring

The default model was chosen by measurement rather than version number. Four
probes each, on the free tier, the day this was run:

| Model | Succeeded | Median latency |
|---|---|---|
| `gemini-3.6-flash` | 0/4 | — |
| `gemini-3.8-flash` | 2/4 | 9.6s |
| `gemini-3.5-flash-lite` | 2/4 | 2.0s |
| **`gemini-3.1-flash-lite`** | **3/4** | **1.8s** |

The newest model was entirely unavailable. Picking by version number would
have produced a project that does not run.

### Reproduce

```bash
make eval                                             # full run, needs a key
python eval/run_eval.py --limit 5                     # quick pass, saves quota
python eval/run_eval.py --no-schema-prompt            # the ablation above
python eval/run_eval.py --rescore eval/results.json   # re-score, no model calls
```

`--rescore` recomputes every metric from a previous run's stored SQL. Scoring
rules are judgement calls, and refining one should not cost another hundred
requests against a rate-limited API.

---

## Design decisions

**Execution accuracy, not SQL string matching.** There are many correct SQL
statements for any question, so comparing generated SQL to a reference as text
measures formatting. Both queries are executed and their result sets compared,
normalised for row order, float precision and boolean/integer encoding.

**A step budget, not a `while True`.** Every iteration is one API call. A
model that keeps calling tools without deciding it is finished will spend the
entire daily quota on one question. `max_steps` stops it; `max_sql_retries`
separately bounds how many failing queries are worth feeding back.

**Failures are messages, not exceptions.** A rejected query, a misspelled
table, a wrong column — each returns text the model can act on, and the
message says what *does* exist. `describe_table("dim_bert")` replies with the
list of real table names. This is the entire mechanism behind self-correction.

**Provider state is carried through the neutral abstraction.** The loop is
written against provider-agnostic message types so it can run on a scripted
stub in tests. That abstraction initially dropped Gemini 3.x's
`thought_signature`, which the API requires back on the turn after any tool
call — a 400 on every multi-turn conversation. `ToolCall.provider_state`
carries opaque provider data the loop never reads.

**One retry layer.** The `google-genai` SDK retries internally by default.
Layered under this project's backoff, five configured attempts became
twenty-five with compounding delays — which presents as a hung request rather
than a failure. SDK retries are disabled; this module's are kept, because only
it knows the agent's step budget.

---

## Layout

```
src/agent/
  config.py      settings, one place for every knob
  guardrails.py  sqlglot-based SQL validation  <- the security boundary
  warehouse.py   read-only DuckDB access, introspection, query timeout
  tools.py       tool specs + dispatcher; failures return messages
  llm.py         provider-neutral tool-calling; Gemini + scripted stub
  agent.py       the loop: steps, self-correction, tracing
  api.py         FastAPI
  cli.py         command line

eval/
  questions.jsonl  20 questions with gold SQL + 2 unanswerable
  run_eval.py      execution accuracy, gold coverage, decline accuracy
```

## API

| Endpoint | Needs a key | Purpose |
|---|---|---|
| `POST /ask` | yes | Question in; answer, SQL, and the full step trace out |
| `POST /sql` | no | Run a SELECT through the same guardrails the agent uses |
| `GET /schema` | no | Tables, columns, types, row counts |
| `GET /health` | no | `ok` / `degraded` / `down`, separately for warehouse and model |
| `GET /metrics` | no | Prometheus |

`/ask` returns the trace, not just the prose. For an agent that is the
difference between a product and a magic box: the caller sees which tools ran,
which SQL executed, what failed, and what it cost.

## Testing

102 tests, no network, no API key, under 5 seconds. The suite builds a
miniature warehouse whose every aggregate can be checked by hand, and drives
the loop with a scripted model so the scenarios that matter — a bad query
corrected, a blocked `DROP`, a model that never stops calling tools — are
reproducible rather than dependent on a model's mood.

```bash
python -m pytest
python -m ruff check src eval scripts tests
```

CI runs the suite on Python 3.11 and 3.12, rebuilds the warehouse, and
re-executes every gold query — so a schema change that would silently
invalidate the reported accuracy fails the build instead.

## Limitations

- **22 questions is a small sample.** A single question moves execution
  accuracy by 5 points. Treat the difficulty breakdown as directional.
- **The questions and the gold SQL are written by the same person who built
  the agent**, which risks encoding the same assumptions. q10 is a case where
  that showed: the question was ambiguous and the metric called the agent
  wrong for a defensible reading.
- **Synthetic data.** The schema and the planted patterns are realistic, but
  no real terminal's data was used.
- **One model, one run.** No temperature sweep, no repeated trials, so the
  numbers carry no variance estimate.

## License

MIT
