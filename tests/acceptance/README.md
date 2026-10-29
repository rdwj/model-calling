# Claude Code acceptance test

`test_claude_code.py` proves the gateway's Anthropic Messages ingress end to
end: Claude Code, pointed at the gateway, completes a small tool-using task
(create a file, edit it, run it) on each of three backends.

```bash
uv run pytest -m acceptance              # all three backends
uv run pytest -m acceptance -k ollama    # free; run this first
```

Plain `pytest` and `pytest -m live` never run it.

## Prerequisites

- **Claude Code** (`claude`) on `PATH`. Checked with 2.1.308.
- **Ollama** on `localhost:11434` with `qwen3-coder:30b` pulled
  (`ollama pull qwen3-coder:30b`). Set
  `MODEL_CALLING_ACCEPTANCE_OLLAMA_MODEL` to use another model, such as
  `gpt-oss:20b`. If you use the compose stack, run `podman-compose down`
  first: its Ollama also publishes port 11434 and hides the host's.
- **`OPENROUTER_API_KEY`** in the environment for the two paid backends:
  Claude Haiku 5.5 through the `anthropic` provider on OpenRouter's
  Messages endpoint, and `openai/gpt-oss-120b` through OpenRouter Chat
  Completions. A run of each costs well under a cent, appended to
  `tests/live/spend.jsonl` and counted against the epic budget.

## What a run does

Each run works in a fresh temp directory. It writes an alias file mapping
`main` and `small` to the backend under test, starts the gateway on a free
port (`python -m tests.acceptance.gateway`, which is
`python -m model_calling` plus a callback that logs each call's cost), and
runs `claude --bare -p` from an empty work directory. Claude Code's home
and config directory are the temp directory too, so no user settings,
login, or memory reach the run, and `--bare` makes `ANTHROPIC_API_KEY` its
only credential.

Claude Code's model settings (`ANTHROPIC_MODEL`, the
`ANTHROPIC_DEFAULT_*_MODEL` variables, `ANTHROPIC_SMALL_FAST_MODEL`, and
`CLAUDE_CODE_SUBAGENT_MODEL`) all name the aliases, so no Claude model id
reaches the gateway. Its tools are limited to `Edit`, `Read`, and
`Bash(python3 *)`; anything else is denied without a prompt.

The test passes when `hello.py` contains the edit and `out.txt`, which only
running the script creates, reads `hello, gateway`. A failure prints
Claude Code's JSON result and the tail of the gateway log.
