"""CI-safe smoke check for tests/evals/memory.jsonl.

Calls `run_memory_eval.validate()` directly (no subprocess, no network) so this
runs as part of the ordinary unit suite / `make check` rather than only via the
opt-in `make evals-dry` target.
"""

from tests.evals.run_memory_eval import _JSONL_PATH, validate


def test_memory_jsonl_validates() -> None:
    assert validate(_JSONL_PATH) == 0
