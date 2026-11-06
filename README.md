# Model Swap Audit

A `nanoagent.plugin/v0001` plugin that asks an agent to audit an API's call
log: given known models' outputs on fixed probe prompts at temperature 0, was
each call answered by the model it claims, by another known model, or by a
model outside the known set? The verifier checks every call against the label
the data generator wrote when it built the call.

## Tools

| Tool | Does | Code |
|---|---|---|
| `read_probes` | Reads a case's known models, its probe prompts, and each known model's reference output for each probe at temperature 0 | `src/plugin/record.py` |
| `read_log` | Reads a case's API call log: each call's id, claimed model, prompt, temperature and output | `src/plugin/record.py` |
| `compare_output` | Tells, for each known model named, whether a call's output equals that model's reference for the call's prompt, character for character; refuses a prompt that is not a probe and a model that is not known | `src/plugin/compare.py` |
| `audit_record` | Reads a whole case with the data generator's label for every call: the `audit-calls` task's reference, which the agent never sees | `src/plugin/record.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `audit-calls` | Audits an API's call log for model swaps against known models' reference outputs | `agent`, with `read_probes`, `read_log` and `compare_output`; reference `audit_record` |

The rules, in the task's instructions: a call at a temperature other than 0,
or on a prompt that is not a probe, is `unverifiable`, even when its output
equals a reference. Otherwise its `matched_models` are the known models whose
reference equals its output exactly, sorted by name, and it is `genuine` when
the claimed model is among them, `swapped` when another known model is (also
when the claimed model is not a known model), and `unknown` when none is.

The task's input is a case number, 1 to 6. The four known models (`atlas-7b`,
`borealis-13b`, `cobalt-34b`, `dynamo-70b`) and five probes are the same in
every case; some references differ only in a full stop. Each case builds its
log from the model that really answered each call (`src/plugin/cases.py`), and
each is there to test one thing:

| Case | Calls | Tests |
|---|---|---|
| 1 | 6 | mostly genuine calls, and one swap to a model whose reference differs only by a full stop |
| 2 | 6 | near misses from models outside the known set (one word changed, a full stop dropped, a trailing space), which are `unknown`, never the closest model |
| 3 | 6 | calls at a temperature above 0 or on a prompt that is not a probe, even when the output equals a reference |
| 4 | 6 | two known models with the same reference on one probe, so a call matches both |
| 5 | 6 | a claimed model that is not a known model |
| 6 | 20 | a mixed log with every verdict and every trap above |

## Layout

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # environment variable NAMEs this plugin needs (none)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS), the tools and the data generator
├── tests/           # test_verifier.py (right, wrong, missing, malformed and boundary audits) and test_cases.py
├── ui/              # outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, with its own audit rules (verifier/rules.py)
```

The generator labels each call from how it built it: the call's temperature
and prompt, which known models it declared to share the source's reference,
and whether the source is a known model. It never builds a near miss that
equals a known reference, since no rule could tell that call apart.
`verifier/` imports only the standard library and `contract`. For the task it
takes the pinned `audit_record` output, applies the rules to its data by
comparing strings, and refuses a reference whose labels disagree. It then
refuses audits with a call missing, duplicated, out of log order or not in the
log, and names the first call whose verdict or `matched_models` is wrong. The
score is the share of log positions whose audit is fully right.

`tests/test_cases.py` checks the generator's labels equal the rules on every
case and on 400 random logs, and that following the instructions with the
tools gives the labels. `tests/test_verifier.py` checks the verifier approves
each case's labels and refuses, wherever it changes the answer, each mistake
an agent could make: naming the closest model for an unknown call, ignoring
the temperature, auditing an unprobed prompt, naming one of two models that
share a reference, unsorted `matched_models`, ignoring the claimed model,
calling a genuine call swapped, calling an unlisted model's swap unknown, a
missing, duplicated, reordered or invented call, and an empty audit. Each
mistake shows in at least one case, and all of them in case 6.

## Develop and run

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`;
this plugin writes nothing.
