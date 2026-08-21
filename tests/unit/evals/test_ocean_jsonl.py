"""CI-safe smoke check for tests/evals/ocean.jsonl.

Calls `run_ocean_eval.validate()` directly (no subprocess, no network) so this
runs as part of the ordinary unit suite / `make check` rather than only via
the opt-in `make evals-dry`. The offline run itself (`--offline`, no API key
needed) is a heavier in-process integration of `HandleAssessmentTurn` end to
end and stays a `tests/evals` script rather than a unit test — see
`run_ocean_eval.run_offline` and `tests/evals/run_evals.py --suites ocean
--offline`.
"""

from tests.evals.run_ocean_eval import _JSONL_PATH, TRAITS, validate


def test_ocean_jsonl_validates() -> None:
    assert validate(_JSONL_PATH) == 0


def test_ocean_jsonl_has_ten_scripted_runs() -> None:
    import json

    rows = [json.loads(line) for line in _JSONL_PATH.read_text().splitlines() if line.strip()]
    assert len(rows) == 10
    assert len({r["id"] for r in rows}) == 10  # no duplicate ids


def test_ocean_jsonl_every_row_has_twenty_answers_and_phrasings() -> None:
    import json

    rows = [json.loads(line) for line in _JSONL_PATH.read_text().splitlines() if line.strip()]
    for row in rows:
        assert len(row["answers"]) == 20, row["id"]
        assert len(row["phrasings"]) == 20, row["id"]
        assert set(row["expected"]) == set(TRAITS), row["id"]


def test_ocean_jsonl_includes_a_skip_case() -> None:
    """PRD/Task 6 requires at least one row exercising the null-answer (skip)
    path, so the < 3-answered -> null scoring rule is actually covered."""
    import json

    rows = [json.loads(line) for line in _JSONL_PATH.read_text().splitlines() if line.strip()]
    with_nulls = [r for r in rows if any(a is None for a in r["answers"])]
    assert len(with_nulls) >= 2  # oc-03 and at least one more (Task 6 Step 2)
