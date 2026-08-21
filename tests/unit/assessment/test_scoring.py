import json
from pathlib import Path

import pytest

from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.scoring import SCALE_MAX_PLUS_ONE, band_for, score

SEED = json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
INS = Instrument.from_definition(SEED)


def test_all_fives_gives_high_on_non_reversed_and_low_on_reversed_traits() -> None:
    # C, E, A, N each have 2 normal + 2 reversed items; 5 → 5 and 6-5=1 → mean 3.0.
    # O (Openness) has the canonical Mini-IPIP-20 key of 1 normal + 3 reversed items
    # (items 5, 10, 15, 20 — see Appendix B), so all-5s gives (5+1+1+1)/4 = 2.0 → low.
    r = score(INS, {n: 5 for n in range(1, 21)})
    for t in r.traits:
        if t.code == "O":
            assert t.score == 2.0 and t.band == "low"
        else:
            assert t.score == 3.0 and t.band == "moderate"


def test_hand_computed_extraversion() -> None:
    # E items: 1 (+), 6 (-), 11 (+), 16 (-). Answers 5,1,4,2 -> 5,5,4,4 -> mean 4.5 high
    answers = {n: 3 for n in range(1, 21)}
    answers.update({1: 5, 6: 1, 11: 4, 16: 2})
    e = next(t for t in score(INS, answers).traits if t.code == "E")
    assert e.score == 4.5 and e.band == "high"


def test_rounding_one_decimal() -> None:
    # A items 2(+),7(-),12(+),17(-): 4,2,4,3 -> 4,4,4,3 -> 3.75 -> 3.8
    answers = {n: 3 for n in range(1, 21)}
    answers.update({2: 4, 7: 2, 12: 4, 17: 3})
    a = next(t for t in score(INS, answers).traits if t.code == "A")
    assert a.score == 3.8


def test_skips_with_three_answered_still_scores() -> None:
    answers = {n: 3 for n in range(1, 21)}
    answers[5] = None  # O item skipped; O has 4 items → 3 answered
    r = score(INS, answers)
    o = next(t for t in r.traits if t.code == "O")
    assert o.score == 3.0 and o.answered == 3 and r.skipped == 1


def test_two_skips_in_one_trait_gives_none() -> None:
    answers = {n: 3 for n in range(1, 21)}
    answers[5] = None
    answers[10] = None
    o = next(t for t in score(INS, answers).traits if t.code == "O")
    assert o.score is None and o.band is None and o.answered == 2


@pytest.mark.parametrize(
    "v,b",
    [
        (1.0, "low"),
        (2.4, "low"),
        (2.5, "moderate"),
        (3.5, "moderate"),
        (3.6, "high"),
        (5.0, "high"),
        (2.45, "moderate"),
    ],
)
def test_bands(v: float, b: str) -> None:
    assert band_for(INS, v) == b


def test_as_dict_shape() -> None:
    d = score(INS, {n: 4 for n in range(1, 21)}).as_dict()
    assert set(d) == {"O", "C", "E", "A", "N", "bands", "answered", "skipped"}


def _o_answers(contributions: list[int]) -> dict[int, int | None]:
    """Raw answers that make Openness's four items contribute these values."""
    out: dict[int, int | None] = {}
    for no, want in zip([i.no for i in INS.items_for_trait("O")], contributions, strict=True):
        out[no] = (SCALE_MAX_PLUS_ONE - want) if INS.item(no).reverse else want
    return out


def test_trait_means_round_half_to_even_not_half_up() -> None:
    """The convention the eval expectations and the reply's figures assume.

    `score()` rounds with Python's `round`, so a mean of exactly 2.25 goes down
    to 2.2 and 2.75 goes up to 2.8 — half to even, either direction. A hand
    calculation that rounds half up disagrees on every such row, which is why
    the scoring docstring says so out loud.
    """
    assert score(INS, _o_answers([3, 2, 2, 2])).trait("O").score == 2.2  # mean 2.25
    assert score(INS, _o_answers([3, 3, 3, 2])).trait("O").score == 2.8  # mean 2.75
