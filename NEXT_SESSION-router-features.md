# Next Session — router-features

## Next: <to be planned via /plan-next-session>

(No next-session focus selected yet. Run `/plan-next-session router-features` to
pick the first slice from the phases below.)

## Remaining epic phases

End-state: a transient backend failure doesn't reach the caller when a retry or
another backend can absorb it. The library retries by default (`max_retries=2`,
the SDKs' name and default), falls back across targets, and cools down a failing
target; gateway aliases list fallback targets in the alias file. memory-hub's
hand-rolled retries (`extraction.py`, `dreaming.py`) move onto model-calling,
and 0.4.0 ships it. Weighted load balancing is not in this epic.

Decided with rdwj 2026-10-30: cooldown only, no weighted spreading; retries on
by default; memory-hub is the pilot; a fallback triggers on the retryable set
after retries are exhausted, plus 401, 403, and 404, never 400; in strict mode
every target's params are checked before the first call.

### Phase 1: Retries

Retry transient failures in the library, so the gateway gets them through
`Client`.

**Work:**
1. Retryable: 408, 409, 429, any 5xx, `APIConnectionError`, `APITimeoutError`
   (the set the openai SDK documents). Everything else surfaces at once.
2. `max_retries` (default 2) per prefix in `config`, per call, and per backend
   in the alias file (`max_retries` key), with an environment default for the
   gateway.
3. Exponential backoff with jitter, capped; honor `Retry-After` (seconds or
   HTTP date) up to the cap. The timeout applies to each attempt.
4. A stream retries only before its first event; a mid-stream failure surfaces
   as today.
5. An `on_retry` callback hook (attempt, error, delay).
6. Replace the hand-written retry loop in `docs/advanced.md` "Not Provided";
   API reference and migration guide entries (retries now on by default).

**Definition of done:** grid cells pass for each retryable status and for 400,
401, 404, and a mid-stream failure (no retry); fake-clock tests check backoff,
jitter bounds, the cap, and both `Retry-After` forms; a gateway cell shows a
backend 503 then 200 reaching the client as 200; `max_retries=0` restores 0.3
behavior; offline suite green with no new xfails.

**Dependencies:** none.

**Parallel-ok:** with issues #13–#16. Not with #17 if both touch the alias
backend keys in the same session; sequence them.

### Phase 2: Fallbacks

An ordered list of targets, tried in turn once a target's retries are spent.

**Work:**
1. Library: `fallbacks=["vllm/x", "openrouter/y"]` per call (and on
   `Client` config if the call pattern needs it), across formats through the
   internal format.
2. Gateway: an alias model entry takes `fallbacks: [{backend, model}, ...]`;
   load-time checks cover every entry, and a bad one fails the load (a bad hot
   reload keeps the previous models).
3. Triggers: the retryable set after retries, plus 401, 403, and 404; never
   400.
4. Strict mode checks every target's params before the first call and refuses
   naming the param and target; drop mode lets each target drop its own.
5. An `on_fallback` callback hook; the library reports which target served the
   call; gateway responses and errors still show only the alias.

**Definition of done:** grid cells pass for a cross-format fallback (Ollama to
`anthropic/`, streaming and not), for 400 not falling back, for 404 falling
back, for the strict up-front refusal, and for gateway errors that name only
the alias; an alias file with a bad fallback entry fails to load; offline suite
green with no new xfails.

**Dependencies:** Phase 1. #17's alias-file TLS keys, if done, land first so
the schema changes once per release.

**Parallel-ok:** with #13–#16.

### Phase 3: Cooldown

A target that keeps failing sits out, so fallbacks skip it instead of paying
its retries on every call.

**Work:**
1. After N failures within a window, a target cools down for a set time;
   thresholds per prefix and per alias backend, with defaults.
2. Fallback order skips cooling targets; if all are cooling, try the one that
   recovers soonest rather than failing outright.
3. State is in memory per `Client` (each gateway replica keeps its own), safe
   under concurrent async calls and the sync wrapper's loop thread.
4. An `on_cooldown` callback hook (enter and recover).

**Definition of done:** fake-clock tests cover enter, skip, all-cooling, and
recover; a concurrency test drives parallel calls through a failing target
without races; a gateway test shows a cooled primary bypassed and then used
again after recovery; `Client.close` regression tests in
`tests/library/test_client.py` stay green.

**Dependencies:** Phase 2.

**Parallel-ok:** with #13–#16.

### Phase 4: memory-hub pilot and 0.4.0

Prove it on a consumer that built its own retries.

**Work:**
1. Release 0.4.0 from `main`: bump, release notes calling out retries on by
   default, publish, fresh-venv smoke with pip.
2. On a memory-hub branch: move the `/v1/chat/completions` calls in
   `src/memoryhub_core/services/extraction.py` and `dreaming.py` onto
   `model_calling.Client`; delete the transport retry loops and backoff;
   keep extraction's retry-with-correction (content, not transport) and decide
   with rdwj whether its circuit breaker gives way to cooldown.
3. Run memory-hub's tests; one live smoke within the budget.

**Definition of done:** `model-calling==0.4.0` on PyPI and a fresh 3.11 venv
runs a call that retries against a fake failing server; memory-hub's suite
passes on the branch with its retry loops removed; the PR is open.

**Dependencies:** Phases 1–3. External wait: rdwj's approval before any write
to `redhat-ai-americas/memory-hub` (not an rdwj-owned repo; check
`viewerPermission` first), then its maintainers' review.

**Parallel-ok:** no.

---

## What this covers (and what it doesn't)

**In scope:**
- Retries with backoff and `Retry-After`, on by default
- Fallbacks in the library and in gateway aliases
- Per-target cooldown
- Callback hooks for retry, fallback, and cooldown
- memory-hub pilot and the 0.4.0 release

**Out of scope:**
- Weighted or round-robin load balancing across equivalent targets (wait for a
  user who asks)
- Shared health state across gateway replicas
- Token counting and cost accounting, the Responses store, FIPS validation
  (future epics; see `archive/next-session/NEXT_SESSION-api-shapes-2026-10-30.md`
  "Out of scope")
- Issues #13–#17 (standalone hardening, parallel-ok with every phase)

## What landed last session (2026-10-30)

(No sessions yet for this epic. The api-shapes epic closed with 0.3.1 the same
day; see `retrospectives/2026-10-30_api-shapes-epic/RETRO.md`.)

## Watch out for

- **Clean room.** Names may follow the SDKs (`max_retries`) or LiteLLM's call
  patterns (`fallbacks`), but the design comes from model-calling's code and
  the provider specs. Do not read LiteLLM source or docs.
- **Dependency policy.** Backoff and jitter need no new dependency; if one is
  proposed, it follows "Dependency Policy" in `CONTRIBUTING.md`.
- **Sync API shutdown is delicate** (`model_calling/library/_sync.py`): a call
  sleeping between retries must cancel cleanly on `Client.close`.
- **Tests must not sleep.** Inject the clock and the sleep so retry tests stay
  in the offline suite's ~10 s.
- **Retries change default behavior** for 0.3 users; the migration guide and
  release notes must say so.
- **Live budget:** not yet set for this epic; set it in the first
  `/plan-next-session`. The ledger is `tests/live/spend.jsonl`.

## If blocked

(Populate as phases progress.)
