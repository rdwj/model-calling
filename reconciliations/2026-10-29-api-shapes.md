# Reconciliation — 2026-10-29 · api-shapes

**Range:** summaries 2026-10-26..2026-10-29 (11 sessions, first reconcile)   **Plan:** NEXT_SESSION-api-shapes.md

The epic has no GitHub issues; its backlog is the plan's "Still open" list,
`KNOWN_FAILURES`, and the summaries' deferrals. Every row was checked
against the code at `3ee53b4`.

## Backlog reconciled
| Item | Was | Action | Why |
|------|-----|--------|-----|
| Messages tool `strict` dropped to Chat backends | Phase 4 strict xfail | Closed | `Tool.strict` (Phase 6 slice 1); no such xfail remains, all 35 are backend limitations |
| `/v1/models` lists no `anthropic/` models | Still open | Closed | Obsolete: lists aliases since Phase 11 |
| `cache_control` strict vs. ignore | Phase 3 slice 1 drift | Closed | Passes through in `formats/anthropic.py`; gateway drops by default; Phase 8 exercises it |
| C901 on `Provider.stream` / gateway | Phase 2 slice 1 | Closed | CI reports complexity with `--exit-zero` only |
| Registry startup probes | Watch-out | Closed | Registry deleted (`2fdee5e`) |
| Anthropic provider drops `logprobs`/`logit_bias`/`user` | Still open | Re-scoped | Decided: refuse in strict mode, like `seed` and the penalties |
| Backend-limitation strict xfails | Phase 5 question | Kept | Decided: stay strict xfails, so a backend gaining the capability flips loudly |
| Document / Refusal blocks | Still open (since Phase 2) | Kept | No handling in `formats/anthropic.py`; carried four sessions untouched, needs a home |
| Streams lose `stop_details`, `container` | Still open | Kept | Confirmed absent |
| `Provider.prepare` stream path; gateway default drop path | Phase 4 thin verification | Kept | No grid cell for either (only strict mode, `tests/test_gateway_grid.py:195`) |
| `tests/integration/` + root scripts | Not flagged | New: delete | Send `ollama/...` to the gateway, 404 since Phase 11 slice 1; deselected, so unnoticed. Decided: delete, update CONTRIBUTING |
| Legacy `ModelCallingClient` | Phase 11 slice 2 question | New: remove in Phase 9 | Decided: the official SDKs are the gateway clients; migration-guide entry; CI import check moves to `Client` |
| `PYPI_README.md` drift; live suite never run on aliases | Phase 11 deferrals | Kept | Phase 9 and Phase 8 respectively |

## Forward-collisions banked
- Phase 9 item 1 (pass-through): Messages into the `anthropic` provider already sends the client's body unchanged; narrow to Responses and to confirming vLLM's native Messages.
- Phase 9 item 3 (LiteLLM doc): `docs/migration-guide.md` "Upgrading from 0.1.x" already holds the breaking-change list; fold and rename, don't write fresh.
- Phase 7: reuse the `sdk_completion` grid pattern and `gateway._serve` with aliases.
- Phase 8: the gateway ignores `ANTHROPIC_BASE_URL` (only `MODEL_CALLING_ANTHROPIC_BASE_URL` sets its backend); budget-only thinking is dropped in non-strict mode.

## Critique
- On track? Yes. Phases 1–5, 6 slice 1, 10, and 11 are done in four days, and the two added phases (10, 11) served the project's original purpose rather than creep. Phase 9 is overloaded (five unrelated items) and should split.
- Recurring friction: the same thin spots repeat across summaries. Compose checks ran on arm64 builds only (Phase 10, 11), live runs keep getting skipped after refactors (Phase 3, 11), and Document blocks were deferred four times without a phase owning them.

## Guidance for next
- Phase 8 stays next; it is unblocked and is the headline test.
- Give the small offline leftovers (strict extras, stream fields, two grid cells, Document blocks, deleting the integration suite) one owned cleanup slice instead of a "Still open" list, so they stop rolling forward.
- Before 0.2.0, run the compose check on the amd64 image once (remote build) and the live suite on aliases.
