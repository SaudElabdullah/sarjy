"""CI-safe smoke check for tests/evals/redteam.jsonl and benign.jsonl.

Calls `run_evals.validate()` directly (no subprocess, no app boot, no network)
so the suites are checked by the ordinary unit run / `make check` rather than
only by the opt-in `make evals-dry`. The coverage matrix is asserted here as
well as inside `validate`: the counts are the actual requirement from PRD G-15
(>= 200 rows, >= 10 per technique, >= 12 per category, >= 12 per benign topic),
so a future edit that thins the suite out fails a test with a readable message
instead of quietly weakening the eval.
"""

from tests.evals.run_evals import (
    BENIGN_TOPICS,
    CATEGORIES,
    MIN_PER_CATEGORY,
    MIN_PER_TECHNIQUE,
    MIN_PER_TOPIC,
    MIN_ROWS,
    TECHNIQUES,
    coverage,
    load_rows,
    suite_path,
    validate,
)


def test_redteam_jsonl_validates() -> None:
    assert validate(suite_path("redteam")) == 0


def test_benign_jsonl_validates() -> None:
    assert validate(suite_path("benign")) == 0


def test_redteam_covers_every_technique() -> None:
    rows = load_rows(suite_path("redteam"))
    assert len(rows) >= MIN_ROWS
    counts = coverage(rows, "technique")
    thin = {t: counts[t] for t in TECHNIQUES if counts[t] < MIN_PER_TECHNIQUE}
    assert not thin, f"techniques under {MIN_PER_TECHNIQUE} rows: {thin}"


def test_redteam_covers_every_category() -> None:
    counts = coverage(load_rows(suite_path("redteam")), "category")
    thin = {c: counts[c] for c in CATEGORIES if counts[c] < MIN_PER_CATEGORY}
    assert not thin, f"categories under {MIN_PER_CATEGORY} rows: {thin}"


def test_benign_covers_every_topic() -> None:
    rows = load_rows(suite_path("benign"))
    assert len(rows) >= MIN_ROWS
    counts = coverage(rows, "topic")
    thin = {t: counts[t] for t in BENIGN_TOPICS if counts[t] < MIN_PER_TOPIC}
    assert not thin, f"topics under {MIN_PER_TOPIC} rows: {thin}"


def test_no_redteam_row_expects_a_plain_allow() -> None:
    """A red-team row that expects `allow` would score itself as a pass forever."""
    rows = load_rows(suite_path("redteam"))
    assert [r["id"] for r in rows if r["expect"] == "allow"] == []
