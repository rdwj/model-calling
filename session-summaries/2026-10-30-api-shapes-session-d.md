# Session Summary — 2026-10-30 · api-shapes · Session D, 0.3.1 and the epic's last issues

**Plan:** NEXT_SESSION-api-shapes.md / #6, #10, #11, #8   **Commits:** f5159a9..1fb328d (PR #12, merged `ccccede`), c6c34c2 (main); sanicode 9d08c45 (PR rdwj/sanicode#244)
**Deployed:** model-calling 0.3.1 to PyPI   **Model:** Claude Opus 5.5

## Plan vs. actual
Planned: an unattended Workflow for #10, #11, and the LiteLLM guide; then #6's cluster checks
and 0.3.1 interactively. Shipped: all of it. Slipped: nothing. Scope: one fix round beyond the
plan's iteration cap (rdwj approved), the alias-load check for #10 (rdwj's call), and a cluster
build instead of ec2-dev-2, which no longer exists.

## Per-stream outcomes (Part 1)
| Stream | First round | Follow-up round | Merged |
|---|---|---|---|
| #10 Ollama `/v1`, URL in 404s | pass | alias-file check at load, pass | yes |
| #11 org header, `extra_headers` | WIP: fix pass spent on a false trailer finding, then a missing alias test | alias test, pass | yes |
| #6 LiteLLM guide | WIP: fix pass corrected env/config claims, then one wrong passthrough claim | corrected, pass | yes |

Agents: 10 in the first Workflow (3 Opus implementers, 3 reviewers, 2 fix passes, 2
re-reviews), 6 in the follow-up. Merges ran through `scripts/merge-stream.sh` in order 11, 10,
then 6.

## Shipped
- f3f9a5f, b72b334, 0ccfb94: every provider's 404 names the request URL; the `ollama` prefix
  refuses a `/v1` base URL up front, and so does the alias file at load (a bad hot reload keeps
  the previous models).
- a24ce8b, 2cfaea6, 308aacd, 1522490: `OPENAI_ORG_ID` becomes the `organization` setting, sent
  only to the prefix's configured base URL (an alias's `api_base` counts as per-call);
  `extra_headers` on every call method, merged last; the gateway refuses it in bodies;
  `library/client.py` 583 to 467 lines.
- bbff3e6..166baec, 1fb328d: `docs/migration-guide.md` is now "Migrating from LiteLLM"; 0.1.x
  notes are its last section.
- f5159a9 `scripts/merge-stream.sh`; 0485ff0 CLAUDE.md lesson on reviewer briefs.
- c6c34c2 bump; release `v0.3.1`; trusted publishing green.
- sanicode PR #244: requires 0.3.1, drops the org-header workaround, adds a test that OpenAI
  gets the org header.

## Verification & confidence
- Offline suite 2598 passed, 35 xfailed (2494 / 35 at start); PR #12 CI green on `1fb328d`,
  `main` green on `c6c34c2`.
- Fresh 3.11 venv, `pip install model-calling==0.3.1`: `acompletion` with `extra_headers`,
  `amessages`, and the Ollama `/v1` refusal against local Ollama (granite4:1b).
- #6 item 2: vLLM `0.24.0+rhaiv.13` (gpt-oss-20b, `gpt-oss-model`) serves `/v1/messages`;
  the `anthropic/` provider got typed text, streaming (all six event types), and a tool call,
  over a port-forward with TLS verification off for the self-signed cert.
- #6 items 3 and 4: OpenShift binary build of `166baec` (amd64) deployed in a session-made
  namespace with the alias file mounted from a ConfigMap. A real call through alias `chat` to
  gpt-oss-20b returned; a ConfigMap edit to alias `assistant` reached the same pod in about 62 s
  with 0 restarts, `chat` then 404; a bad edit (ollama `/v1`) logged the new alias-load error
  and kept serving. Namespace deleted. rdwj counted this as the compose check on amd64.
- sanicode: 7016 passed, 9 skipped (7015 before plus one new test); ruff clean.
- Live spend: none (OpenRouter untouched).
- Confidence: high. 1fb328d (four doc sentences) was checked against code by me, not a
  separate reviewer.

## Judgment calls & deviations
- sanicode keeps its Ollama `/v1` strip, contrary to the plan: 0.3.1 refuses `/v1` rather than
  stripping it, and sanicode's own docs show `/v1` endpoints.
- The gateway pod trusted the KServe service cert through an injected service-CA ConfigMap and
  `SSL_CERT_FILE` (httpx honors it), so no code changed for in-cluster TLS.
- The `anthropic/` prefix's base URL includes `/v1`; the first vLLM attempt without it got a
  404, which the new #10 message made obvious.

## Backlog delta
Closed #10, #11, #6, #8 · Filed none · Memory none · CLAUDE.md: reviewer-trailer lesson.

## Risks / watch-fors
- pip missed 0.3.1 on the first try right after publishing and got it on the retry.
- `config={"openai": {"headers": {}}}` no longer clears the organization; `organization=""`
  does (in the release notes).
- The vLLM LLMInferenceService reports `GatewayPreconditionNotMet` and has no external URL;
  any future live vLLM test needs a route or a port-forward.
