# Retrospective: api-shapes epic

**Date:** 2026-10-30
**Effort:** Rewrite model-calling as a LiteLLM replacement that treats OpenAI and Anthropic wire formats as equals, from harness to 0.3.1 and a migrated consumer.
**Issues:** #1–#8, #10, #11 (all closed); follow-ups #13–#17
**Commits:** `2f9a2a8` (epic plan) to `7aeeaa6`, about 200 commits over 15 sessions, 2026-10-26 to 2026-10-30

## What We Set Out To Do

The library offers `completion()` and `messages()` side by side, familiar to LiteLLM users. The
optional gateway speaks Chat Completions, Messages, and stateless Responses over Ollama, vLLM,
OpenAI-compatible backends, Anthropic, and Cohere, and names models only by hot-reloaded alias.
An offline grid validates every ingress and backend cell against the official SDK types, live
smoke tests run through OpenRouter within $5, Claude Code completes a tool task through the
gateway, and sanicode moves off LiteLLM.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Phases 10 (config from env) and 11 (aliases, hot reload) added mid-epic | Good pivot | Needed for deployment; they served the original goal rather than creep (2026-10-29 reconciliation) |
| 0.2.0 cut early, Phase 9 pulled forward | Scope deferral | A colleague needed a package by end of business 2026-10-30 |
| Phase 9 split; leftovers given an owner as Phase 12 | Good pivot | The reconciliation found offline leftovers rolling forward four sessions with no home |
| Sessions B and D run as unattended Workflows | Good pivot | Parallel streams in worktrees with a scripted merge; B needed interactive rescue for three of four streams, D for two of three |
| Pilot friction (#10, #11) shipped as 0.3.1 | Good pivot | Releasing 0.3.0 before the pilot let sanicode pin a published version |
| Compose amd64 check replaced by an in-cluster OpenShift run | Good pivot | ec2-dev-2 is gone; the cluster run tests the image the way it will deploy |
| sanicode keeps its Ollama `/v1` strip | Missed (in the plan) | The plan assumed 0.3.1 would make it unnecessary; the #10 decision (refuse, not strip) means it still is |

## What Went Well

- The grid as exit predicate. Strict xfails named their fixing phase, so the bug list was
  executable, and 35 remaining xfails are all deliberate backend limitations.
- Cost discipline: about $0.91 of the $5 live budget, $0.90 of that being 18 unknown-cost
  placeholders at $0.05. Everything else ran offline.
- Clean room held for 15 sessions, including the LiteLLM migration guide, which cites
  model-calling code for every claim and none of LiteLLM's.
- Fresh-agent review caught real defects each time: wrong env-var claims and a false
  passthrough claim in the guide, a missing alias test for #11, Cohere's spec gaps in Session B.
- The #10 URL-in-404 change paid off within the hour: it named the wrong path when the
  `anthropic/` prefix was first pointed at vLLM.
- A real consumer (sanicode, 7016 tests) runs on the published package.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| Workflow review rounds wasted on false findings (Session D trailer) and underspecified briefs | Process | CLAUDE.md lesson (`0485ff0`); state conventions in reviewer briefs |
| Session B's `merge-stream.sh` was described but never committed; Session D rewrote it | Process | Committed in `f5159a9` |
| Session B carries had no owner for two sessions | Follow-up issue | #13, #14, #15, #16 |
| No TLS setting; self-signed OpenShift backends need a test transport or process-wide `SSL_CERT_FILE` | Follow-up issue | #17 |
| Cohere has no live coverage; cloud providers through sanicode never called live | Accept | Documented in README and the Session C summary |
| vLLM `/v1/messages` checked by hand, no repeatable test | Accept | rdwj's call; the cluster's vLLM has no route |
| One doc commit (`1fb328d`) skipped a separate review | Accept | Four sentences, each checked against code |

## Action Items

- [ ] Merge rdwj/sanicode#244 (rdwj)
- [ ] #13 Cohere review findings
- [ ] #17 per-prefix TLS settings (blocks a repeatable vLLM-on-OpenShift test)
- [ ] #14, #15, #16 (low priority)

## Patterns

First retro in this repo; patterns come from the reconciliation and the 15 summaries.

**Start:** commit every script a plan names before the session that relies on it. Give each
deferred item an issue the session it is deferred, not at the next reconciliation. Put the
project's commit and style conventions in every reviewer brief.

**Stop:** writing plan assumptions about downstream code (sanicode's strip) without checking that
code. Deferring infrastructure checks (amd64, live suite on aliases) across several sessions.

**Continue:** strict xfails as the bug list; the spend ledger; a fresh reviewer that sees the diff
and decision but not the implementer's summary; premise checks at session start; asking before
cluster writes and releases.
