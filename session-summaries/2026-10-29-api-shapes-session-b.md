# Session Summary — 2026-10-29 · api-shapes · Session B, overnight workflow

**Plan:** NEXT_SESSION-api-shapes.md / #2, #3, #4, #5   **Commits:** batch/0.3.0, `2d38568..019a023`
**Deployed:** none   **Model:** Claude Opus 5.5 orchestrating; Sonnet 5.5 and Opus 5.5 agents

## Plan vs. actual
Planned: four streams (Cohere v2, Phase 12, Responses ingress, Responses backend) merged onto
`batch/0.3.0` behind a draft PR.
Shipped: all four, in two rounds. The unattended run merged stream 3; streams 1, 2, and 4 stopped
under the plan's rules. The same evening rdwj had stream 4 merged as-is, Phase 12 finished by
the orchestrator, and Cohere redone from scratch by a fresh Opus pass with an Opus review.
Scope: stayed in scope; merges ran as a shell script (`merge-stream.sh`) between agent runs.

## Stream outcomes
| Stream | Merge | How it got there | Suite after merge |
|---|---|---|---|
| 3 Responses ingress (#3) | `e880c9d` | Opus implement, Sonnet review, no findings | 1868 passed, 35 xfailed |
| 4 Responses backend (#4) | `62b7014` | Stopped on spend; merged as-is by rdwj | 2126 passed, 35 xfailed |
| 2 Phase 12 (#2) | `21308a7` | Stopped on one gap; orchestrator finished it | 2152 passed, 35 xfailed |
| 1 Cohere v2 (#5) | `019a023` | First pass rejected; second Opus pass, Opus review, no must-fix | 2494 passed, 35 xfailed |

- **Stream 3.** 428 new tests; shared client code moved to `library/_calls.py`.
- **Stream 4.** OpenRouter Responses provider, mistral-nemo fixtures, a pass-through test, and
  `tests/live/test_responses_live.py`. The ledger delta reached $0.5502 against the $0.50 cap:
  eleven `$0.05` unknown-cost placeholders make up $0.55 of it; reported costs come to about
  $0.0002. The implementer ran the live smoke past the cap instead of stopping.
- **Stream 2.** The rebase onto streams 3–4 conflicted in `client.py` (resolved; `responses()`
  got the same `timeout` keyword). Item 6 (timeout) had shipped with no test, which the reviewer
  missed; added one, mutation-checked. Added the non-streaming `stop_details`/`container` test,
  and the Responses row's `thinking_unsupported` drop-path check that neither stream could have
  written alone. These three test commits had no separate review.
- **Stream 1.** The first pass (Sonnet) failed 13 must-fix items on fixture/spec alignment and was
  left as WIP on `stream/cohere-v2`. The second pass started fresh off `batch/0.3.0`
  (`stream/cohere-v2-take2`): 184 Cohere cells across every ingress including Responses, three
  spec-backed strict refusals (`tool_named`, `parallel_tools_off`, `reasoning_effort`), and
  `tests/test_cohere_fixtures.py` checking fixtures against the reference's field lists. The
  reviewer re-fetched the v2 reference and found no spec contradiction.

## Verification & confidence
- Premise checks passed at start: `main` = `origin/main` = `2d38568`, Test Package green,
  1440 passed / 35 xfailed, `qwen3-coder:30b` present, key set, no `batch/0.3.0`.
- Every merge ran rebase, full offline suite, flake8 (CI selection), and black from
  `merge-stream.sh`. gitleaks clean on `main..batch/0.3.0`.
- Phase 12 definition of done: `grep -rn "integration" pyproject.toml CONTRIBUTING.md` empty.
- Confidence: high for streams 2 and 3; medium for stream 4's fixtures (re-recorded during the
  fix round, reviewed by re-check only) and for Cohere (hand-built, no live coverage by design).

## Judgment calls & deviations
- rdwj: stream 4 merges as-is; Phase 12 item 4 needs no Refusal block (anthropic 2.0.0 has none).
- Stream 4 made `openrouter_responses` refuse `parallel_tool_calls` because mistral-nemo ignored it
  in the recording: model behavior, not an endpoint limit. Merged with stream 4; unruled.
- Cohere design calls (none contradicts the spec), for rdwj: `tool_plan` maps to a Text block and
  goes back as `content`, not `tool_plan`; named `tool_choice` is refused rather than emulated;
  no thinking field is sent when the request has none, so reasoning models think by default;
  embedding `input_type` defaults to `search_document`; usage prefers `tokens` over
  `billed_units`; `ToolResult.is_error` is dropped.
- Cohere review should-fix, not done: tool results sent as bare strings where the spec words them
  as a JSON object string or content blocks; the fixture test stops at `delta.message`; a
  `fn["name"]` KeyError; no 96-text embed batch limit; a wrong citations comment. README line 20 (Cohere
  read as shipping in 0.2.0, Responses as unavailable) fixed by the orchestrator.
- About 17 agents in total, over the plan's 8–10.
- The merge script's first run misreported "xfailed" as a failure (grep bug); fixed and rerun.
- Files over the ~512-line guideline: `library/client.py` (~580), `tests/harness/scenarios.py` (916).

## Backlog delta
Filed none · Closed none (#2–#5 close when the PR merges to `main`) · Carry: Cohere should-fix
and design calls, the `parallel_tool_calls` refusal, stream 4's live cost via `usage.cost`, the
two oversized files.
