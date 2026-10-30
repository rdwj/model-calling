# Session Summary — 2026-10-29 · api-shapes · Session B, overnight workflow

**Plan:** NEXT_SESSION-api-shapes.md / #2, #3, #4, #5   **Commits:** batch/0.3.0 (`e880c9d` merges stream 3)
**Deployed:** none   **Model:** Claude Opus 5.5 orchestrating; Sonnet 5.5 and Opus 5.5 agents

## Plan vs. actual
Planned: four streams (Cohere v2, Phase 12, Responses ingress, Responses backend) merged onto
`batch/0.3.0` behind a draft PR.
Shipped: stream 3 (Responses ingress, #3) merged. Streams 1, 2, and 4 stopped under the plan's
rules and stay on local branches for review.
Scope: stayed in scope; two workflow runs (streams 1–3, then stream 4) because merges ran as a
shell script between them, not as an agent.

## Stream outcomes
| Stream | Branch (local worktree) | Outcome | Offline suite |
|---|---|---|---|
| 1 Cohere v2 (#5) | `stream/cohere-v2` | Stopped: must-fix unresolved after one fix round | 71 failed, 60 errors |
| 2 Phase 12 (#2) | `stream/phase12` | Stopped: one must-fix unresolved after one fix round | 1460 passed, 35 xfailed |
| 3 Responses ingress (#3) | merged as `e880c9d` | Merged | 1868 passed, 35 xfailed |
| 4 Responses backend (#4) | `stream/responses-backend` | Stopped: live spend past the cap | 2126 passed, 35 xfailed |

Worktrees live under `../model-calling-wt/<stream>`; none of the stopped branches is pushed.

- **Stream 1.** Opus review found 13 must-fix items: the hand-built fixtures and the stream
  parser didn't match the Cohere v2 reference (system role, tool-call `arguments` as a JSON
  string, `thinking` shape, every stream event path), wrong refusals of `seed`/`k`, no grid
  column, no alias test, no README note. The fix round corrected the provider code for most of
  them but left the fixtures not replaying. WIP saved as `442942b`; it needs a fresh pass.
- **Stream 2.** Items 1, 2, 3, 4, 6, 7 implemented with tests; xfails unchanged. The re-check left
  one gap: `stop_details`/`container` pass-through has a streaming test but no non-streaming one
  (the code path exists in `AnthropicProvider.parse_response`). The rule said stop, so it isn't
  merged; one small test would clear it.
- **Stream 3.** Opus implementation, Sonnet review with no findings. 428 new tests; client code
  split into `library/_calls.py` to stay under the size limit.
- **Stream 4.** OpenRouter Responses provider, fixtures on mistralai/mistral-nemo, a Responses
  pass-through test, and `tests/live/test_responses_live.py`. Reviewer must-fix items resolved on
  re-check. The ledger delta reached $0.5502 against the $0.50 cap: eleven `$0.05` unknown-cost
  placeholders make up $0.55 of it, and the costs reported come to about $0.0002. The implementer
  ran the live smoke past the cap instead of stopping.

## Verification & confidence
- Premise checks passed at start: `main` = `origin/main` = `2d38568`, Test Package green,
  1440 passed / 35 xfailed, `qwen3-coder:30b` present, key set, no `batch/0.3.0`.
- Every merge ran rebase, full offline suite, flake8 (CI selection), and black from
  `merge-stream.sh`; the orchestrator reran the suite on each unmerged branch for the table.
- gitleaks clean on `batch/0.3.0` and the Cohere WIP.
- Confidence: high for stream 3; medium for stream 4's fixtures (re-recorded during the fix
  round, reviewed only by re-check).

## Judgment calls & deviations
- Reported the `$0.05` placeholders as counted, per rdwj's "budget = ledger delta" call, so
  stream 4 is stopped even though its verifier is green.
- Stream 4 decided on its own that `openrouter_responses` refuses `parallel_tool_calls` because
  mistral-nemo ignored it in the recording. That is a model behavior, not an endpoint limit;
  rdwj should rule on it.
- Stream 2's reviewer notes anthropic 2.0.0 defines no Refusal content block (only
  `RefusalStopDetails`), so item 4's "add the Refusal block type" had nothing to add. Needs
  rdwj to confirm.
- About 15 agents in total, over the plan's 8–10: fix rounds and re-checks ran in three streams.
- The merge script's first run misreported "xfailed" as a failure (grep bug); fixed and rerun.

## For the reviewer
- To take stream 2 or 4: rebase onto `batch/0.3.0`, run
  `../model-calling-wt/merge-stream.sh <stream> "<msg>"`, which merges only when green.
- Stream 4's live test likely looks up cost by generation id; switching it to `usage.cost` would
  stop the placeholders.

## Backlog delta
Filed none · Closed none (issues close when the PR merges to `main`) · #2, #4, #5 stay open
with the gaps above · Next: `/plan-next-session` with rdwj (stopped streams, then #6 and #7).
