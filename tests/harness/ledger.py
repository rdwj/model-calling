"""
The committed OpenRouter spend ledger, tests/live/spend.jsonl.

Every call that costs money appends one JSON line with its cost from
OpenRouter's usage accounting. The guard sums the file and refuses new
spend once the epic budget is reached.
"""

import datetime
import json
from pathlib import Path
from typing import Optional

LEDGER_PATH = Path(__file__).parent.parent / "live" / "spend.jsonl"
EPIC_BUDGET_USD = 5.00


class BudgetExceeded(RuntimeError):
    pass


def total_spend(path: Path = LEDGER_PATH) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            total += float(json.loads(line)["cost_usd"])
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path}:{lineno} is not a valid ledger entry: {line!r}") from e
    return total


def check_budget(budget: float = EPIC_BUDGET_USD, path: Path = LEDGER_PATH) -> float:
    """Return remaining budget, or raise BudgetExceeded when none is left."""
    spent = total_spend(path)
    if spent >= budget:
        raise BudgetExceeded(
            f"{path} records ${spent:.6f} of OpenRouter spend, at or over the "
            f"${budget:.2f} epic budget. Refusing to make more paid calls."
        )
    return budget - spent


def record(
    source: str,
    endpoint: str,
    model: str,
    response_id: Optional[str],
    cost_usd: float,
    path: Path = LEDGER_PATH,
) -> None:
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "endpoint": endpoint,
        "model": model,
        "id": response_id,
        "cost_usd": cost_usd,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
