#!/usr/bin/env bash
# Rebase a stream branch onto the batch branch, verify it, and fast-forward
# the batch branch onto it. Run from the main checkout, not a worktree.
#
#   scripts/merge-stream.sh <stream-branch> [batch-branch]
#
# Verifies: offline suite green with the xfail count unchanged from the
# batch branch, CI's flake8 selection, black --check, gitleaks on the diff.
set -euo pipefail

STREAM="${1:?usage: merge-stream.sh <stream-branch> [batch-branch]}"
BATCH="${2:-batch/0.3.1}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "error: $ROOT has uncommitted changes; commit or stash them first" >&2
    exit 1
fi

xfail_count() {
    uv run pytest -q -p no:cacheprovider 2>&1 | tail -1 \
        | grep -Eo '[0-9]+ xfailed' | grep -Eo '[0-9]+' || echo 0
}

git checkout -q "$BATCH"
BASE_XFAIL="$(xfail_count)"

# A stream checked out in a worktree can't be checked out here; rebase it there.
WT="$(git worktree list --porcelain | awk -v b="refs/heads/$STREAM" '/^worktree /{w=$2} $0=="branch " b {print w}')"
if [[ -n "$WT" ]]; then
    git -C "$WT" rebase "$BATCH"
else
    git rebase "$BATCH" "$STREAM"
    git checkout -q "$STREAM"
fi

cd "${WT:-$ROOT}"
if ! uv run pytest -q -p no:cacheprovider > /tmp/merge-stream-pytest.log 2>&1; then
    tail -40 /tmp/merge-stream-pytest.log >&2
    echo "error: offline suite failed on $STREAM (full log /tmp/merge-stream-pytest.log)" >&2
    exit 1
fi
STREAM_XFAIL="$(tail -1 /tmp/merge-stream-pytest.log | grep -Eo '[0-9]+ xfailed' | grep -Eo '[0-9]+' || echo 0)"
if (( STREAM_XFAIL > BASE_XFAIL )); then
    echo "error: $STREAM adds xfails ($BASE_XFAIL -> $STREAM_XFAIL)" >&2
    exit 1
fi
uv run --frozen --only-group dev flake8 model_calling tests --count --select=E9,F63,F7,F82 --show-source --statistics
uv run --frozen --only-group dev black --check model_calling tests
gitleaks git --no-banner --log-opts="$BATCH..$STREAM" .

cd "$ROOT"
git checkout -q "$BATCH"
git merge --ff-only -q "$STREAM"
echo "merged $STREAM onto $BATCH: $(tail -1 /tmp/merge-stream-pytest.log)"
