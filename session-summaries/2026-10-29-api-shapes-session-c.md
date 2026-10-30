# Session Summary — 2026-10-29 · api-shapes · Session C, 0.3.0 release and sanicode pilot

**Plan:** NEXT_SESSION-api-shapes.md / #7, #8   **Commits:** 9c46d41 (main, pushed); sanicode 0ba965c, 6674d82 (PR rdwj/sanicode#243)
**Deployed:** model-calling 0.3.0 to PyPI   **Model:** Claude Opus 5.5

## Plan vs. actual
Planned: release 0.3.0, then move sanicode off LiteLLM by PR.
Shipped: both. 0.3.0 is on PyPI (release `v0.3.0`); sanicode PR #243 removes LiteLLM, CI tests
green on 3.11 through 3.13. Slipped: merging #243 (rdwj's call). Scope: two decisions beyond the
plan, both asked first (Python floor, Ollama `/v1`).

## Shipped
- 9c46d41: `__version__` 0.3.0; README status names what 0.3.0 adds; README and API reference
  document the `timeout` setting (per prefix and per call) that Phase 12 added undocumented.
- Release `v0.3.0` with notes; trusted-publishing workflow green; wheel and sdist on PyPI.
- sanicode 0ba965c: requires Python 3.11 (CI matrix, classifiers, ruff target, tomli fallback).
- sanicode 6674d82: LiteLLM replaced with `model_calling.Client`; provider registry maps to
  model-calling prefixes; azure and bedrock removed; tests on a shared `fake_llm` fixture
  (`tests/conftest.py`) that drives a real `Client` over `httpx.MockTransport`.

## Verification & confidence
- Fresh 3.11 venv, `pip install model-calling==0.3.0`: `acompletion` and `amessages` against
  local Ollama (granite4:1b).
- Offline model-calling suite: 2494 passed, 35 xfailed (unchanged).
- sanicode: 7013 passed, 9 skipped before; 7015 passed, 9 skipped after; ruff clean. The 3.10
  baseline wasn't run (no 3.10 venv); the baseline is on 3.11.
- Live through sanicode's `LLMClient` to local Ollama with an old-style `/v1` endpoint: tier
  `test_connection` and pipeline `augment` both returned.
- An Opus review of the sanicode branch found eight items; fixed all but two Low ones (below).
  The key-leak test was mutation-checked after the fix (fails with the guard removed).
- Live spend: none.
- Confidence: high for the wire mapping of every provider (asserted per request); medium for
  cloud providers end to end, since none was called live (no keys used).

## Judgment calls & deviations
- rdwj: sanicode drops Python 3.10; sanicode strips a trailing `/v1` from Ollama endpoints.
- One process-wide `model_calling.Client` in sanicode (lock-guarded): server mode builds an
  `LLMClient` per scan and each `Client` owns a loop thread.
- Pipeline calls get a 600 s timeout; there is no config for it and model-calling's 300 s
  default looked short for whole-file augment prompts on local models.
- Unknown and removed providers raise `LLMProviderUnsupportedError` (a `ValueError`), since scan
  stages swallow `LLMNotConfiguredError` silently; `from_config` also logs a warning.
- sanicode clears the `openai` prefix's `OpenAI-Organization` header, which also drops it for
  real OpenAI calls (workaround for #11).

## Backlog delta
Filed #10 (Ollama `/v1` base gives a bare 404), #11 (org header follows `api_base`) · Closed
none · Commented #7 (pilot record), #8 · Memory none · CLAUDE.md none.

## Drift & forward-collisions
- Forward: #10 and #11 are small 0.3.x candidates; sanicode's workarounds can go once they land.
- Not filed, recorded on #7: exact `pydantic==2.14.1` pin downgrades consumers (policy, no
  breakage); library responses carry the prefixed model name; `Client` reads env keys at
  construction.

## For the reviewer
- Sanity-check: the Gemini, Mistral, Groq, and Together base URLs come from each vendor's
  public OpenAI-compatibility docs and were never called live.
- Not fixed from review (Low): Together's own docs name `TOGETHER_API_KEY` while sanicode reads
  `TOGETHERAI_API_KEY` only; `docs/air-gapped-deployment-guide.md` uses a
  `SANICODE_LLM_FAST_ENDPOINT` variable sanicode doesn't read (predates this work).
- sanicode's `scan` CI job fails on the PR as it has on `main` since April (action's find-sarif
  step); untouched.

## Risks / watch-fors
- uv didn't see 0.3.0 for several minutes after publish while pip did; `--refresh-package`
  fixed it. Expect the same on future releases.
- sanicode's next release needs a note: Python 3.11+, Azure OpenAI and AWS Bedrock removed.
