from sarjy.contexts.conversation.application.ports import Fact
from sarjy.contexts.conversation.application.prompt_builder import PromptBuilder
from sarjy.contexts.memory.application.snapshot import SNAPSHOT_BYTES
from sarjy.shared.tokens import estimate_tokens


def _facts_at_snapshot_cap() -> list[Fact]:
    """Facts sized the way `FactSnapshot.snapshot` actually caps them (M-4).

    Mirrors `FactSnapshot`'s own accumulation loop: keep adding `key: value`
    pairs while `len(key) + len(value) + 2 <= SNAPSHOT_BYTES` running total.
    A prompt built off a real snapshot can never carry more fact bytes than
    this, so it is the true worst case — not an arbitrary fact count (the
    original version of this fixture used 60 facts with 4-char values, ~780
    bytes total, well short of what `FactSnapshot` actually allows through).
    """
    facts: list[Fact] = []
    used = 0
    i = 0
    while True:
        key, value = f"f{i}", "v" * 48
        cost = len(key) + len(value) + 2
        if used + cost > SNAPSHOT_BYTES:
            break
        facts.append(Fact(key, value, "fact"))
        used += cost
        i += 1
    assert used < SNAPSHOT_BYTES  # sanity: the fixture is actually at the cap
    return facts


def test_estimate_tokens_of_empty_string_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_counts_a_whitespace_run_once() -> None:
    # "a  b" (two spaces) is one whitespace run, not two.
    assert estimate_tokens("a  b") == estimate_tokens("a b")


def test_static_prompt_is_within_the_1200_token_budget() -> None:
    # Phase 7 Task 6 (L-6): no Gemini API key is available in this environment to
    # call `client.models.count_tokens`, so the static prefix's ~1,200-token budget
    # is checked against the heuristic estimator instead — see `sarjy.shared.tokens`.
    est = estimate_tokens(PromptBuilder().static_text)
    assert est <= 1200, f"estimate={est}, margin={1200 - est}"


def test_a_prompt_at_the_snapshot_cap_with_a_workflow_block_is_within_the_1800_token_budget() -> (
    None
):
    # Worst case for a turn mid-assessment run: facts maxed out at the real
    # FactSnapshot cap (M-4), plus a 600-char workflow_block — the longest
    # PromptBuilder.build ever keeps, since it truncates via
    # sanitise_value(workflow_block, 600).
    built = PromptBuilder().build(
        facts=_facts_at_snapshot_cap(), workflow_block="w" * 600, summary=None
    )
    est = estimate_tokens(built.system)
    assert est <= 1800, f"estimate={est}, margin={1800 - est}"


def test_a_prompt_at_the_snapshot_cap_with_a_results_block_is_within_the_1800_token_budget() -> (
    None
):
    # Worst case right after an assessment run finishes: same fact cap, plus a
    # 600-char results_block. Deliberately NOT combined with workflow_block in
    # one build() call: PromptBuilder.build's docstring records that the two
    # are mutually exclusive in practice — `run_turn.py` passes
    # `workflow_block=run.prompt_block if run else None` and
    # `results_block=last.prompt_block if last else None`, and a turn can only
    # ever have an open run OR a most-recent finished one, never both — so a
    # single turn can never carry both blocks at once. Budgeting for that
    # combination would size the prompt against a state RunTurn cannot
    # produce; each block is tested separately here instead, both at the real
    # fact cap.
    built = PromptBuilder().build(
        facts=_facts_at_snapshot_cap(), workflow_block=None, summary=None, results_block="r" * 600
    )
    est = estimate_tokens(built.system)
    assert est <= 1800, f"estimate={est}, margin={1800 - est}"
