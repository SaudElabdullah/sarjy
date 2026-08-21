from sarjy.contexts.guardrails.domain.grounding import (
    extract_numbers,
    mentions_results,
    ungrounded_numbers,
)
from sarjy.shared.text import to_speech


def test_digits_and_words_grounded() -> None:
    assert (
        ungrounded_numbers(
            "It's twenty-two degrees and 40 percent chance of rain.", [22.0, 71.6, 40.0]
        )
        == []
    )


def test_detects_fabricated() -> None:
    assert ungrounded_numbers("It's 25 degrees in Tokyo.", [22.0, 71.6]) == [25.0]


def test_tolerance_and_negative() -> None:
    assert ungrounded_numbers("Around minus three degrees.", [-3.4]) == []


def test_ignores_time_of_day() -> None:
    assert ungrounded_numbers("Rain is likely around 3 pm.", [60.0]) == []


def test_thousands_separator() -> None:
    assert extract_numbers("There were 1,000 attendees.") == [1000.0]


def test_hundred_and_word_form() -> None:
    assert extract_numbers("The score is one hundred and four.") == [104.0]


def test_ignores_colon_time() -> None:
    assert extract_numbers("Meet me at 8:30.") == []


def test_ignores_ordinal_suffix() -> None:
    assert extract_numbers("Welcome to the 21st century.") == []


def test_ignores_bare_year() -> None:
    # Mid-sentence, not sentence-final, so this genuinely exercises the year
    # filter rather than accidentally passing because a trailing "." blocked
    # the digit match (see C1: the digit lookahead used to swallow "2026." as
    # unmatched instead of extracting-then-filtering it).
    assert extract_numbers("In 2026 it rained a lot.") == []


def test_spoken_decimal() -> None:
    assert extract_numbers("The temperature is twenty two point five degrees.") == [22.5]


def test_item_counter_treated_as_numbers() -> None:
    assert extract_numbers("This is item seven of twenty.") == [7.0, 20.0]


def test_round_trip_via_to_speech() -> None:
    # to_speech() converts digits to spoken words for TTS; extract_numbers
    # should be able to read the resulting words back as the same number.
    assert extract_numbers(to_speech("The count is 104.")) == [104.0]
    assert extract_numbers(to_speech("It's 22.5 degrees.")) == [22.5]
    assert extract_numbers(to_speech("It's 100 degrees.")) == [100.0]


# --- C1: a sentence-final period must not swallow the digit before it -----


def test_sentence_final_digit() -> None:
    assert extract_numbers("It's 25.") == [25.0]
    assert extract_numbers("The high is 30.") == [30.0]
    assert extract_numbers("high of 26.") == [26.0]


def test_sentence_final_decimal() -> None:
    assert extract_numbers("It's 22.4.") == [22.4]


def test_sentence_final_thousands_separator() -> None:
    assert extract_numbers("There were 1,000.") == [1000.0]


# --- C2: spoken ordinals, spoken times, spoken years are not quantities ---


def test_ignores_spoken_ordinal() -> None:
    assert extract_numbers(to_speech("It's the 21st.")) == []


def test_ignores_spoken_time_from_colon() -> None:
    # to_speech has no notion of clock times — it converts "8:30" digit by
    # digit, leaving "eight:thirty pm" — round-tripping through the am/pm
    # frame still has to be recognised as a time, not two quantities.
    assert extract_numbers(to_speech("At 8:30 pm it was warmer.")) == []


def test_ignores_spoken_year() -> None:
    assert extract_numbers(to_speech("In 2026 summers are hotter.")) == []


def test_ignores_spoken_time_with_period_word() -> None:
    assert extract_numbers("Rain is likely around three p.m.") == []


def test_plain_word_number_still_extracted() -> None:
    # Sanity check that the time/ordinal/year stripping above isn't so
    # aggressive it eats ordinary quantities.
    assert extract_numbers("The high is thirty.") == [30.0]


def test_ignores_spoken_time_before_time_of_day_frame() -> None:
    assert extract_numbers("Eight thirty in the morning was chilly.") == []


def test_ignores_spoken_year_with_preposition() -> None:
    assert extract_numbers("The policy has been in place since nineteen ninety-eight.") == []


def test_spoken_year_tail_does_not_swallow_a_real_quantity() -> None:
    # Fix round 2 (2): the old [a-z]+\s+[a-z]+ tail matched any two words
    # after "twenty", so "by twenty five degrees" lost its 25. The tail must
    # be year-shaped (a tens/teen word, optionally with a unit suffix).
    assert extract_numbers("The temperature dropped by twenty five degrees.") == [25.0]
    assert extract_numbers("in twenty twenty-six") == []
    assert extract_numbers("Rain starts in twenty minutes.") == [20.0]


# --- I1: "hundred" without a leading "one" ---------------------------------


def test_a_hundred() -> None:
    assert extract_numbers("It is about a hundred degrees.") == [100.0]


def test_two_hundred() -> None:
    assert extract_numbers("It is about two hundred degrees.") == [200.0]


# --- I3: duplicate quantities are collapsed --------------------------------


def test_dedupes_repeated_numbers() -> None:
    assert extract_numbers("It's 25 degrees, 25 degrees exactly.") == [25.0]


def test_the_extractor_is_the_shared_one_not_a_second_copy() -> None:
    # C1: the narrator pre-screens its own text with the same function, so a
    # narrative it accepts cannot be cut here. Two copies is how they drifted.
    from sarjy.shared.numbers import extract_numbers as shared

    assert extract_numbers is shared


def test_mentions_results_catches_a_follow_ups_own_vocabulary() -> None:
    # A follow-up answers in the words of the QUESTION, not of the instrument.
    # "And the other four?" gets an answer that names no trait at all, and the
    # tightest check in the guard would have sat that sentence out.
    assert mentions_results("The other four are 3.5, 4.0, 3.2 and 2.0, each out of five.")
    assert mentions_results("Your scores were fairly even.")
    assert mentions_results("Your results are on file.")
    # ...and the trait names and scoring verbs it always caught.
    assert mentions_results("You scored 2.8 on openness.")
    assert mentions_results("Your score is 4.")


def test_mentions_results_still_ignores_an_ordinary_sentence() -> None:
    # The point of the narrowness: the score check is decimal-tight, and running
    # it over every sentence would cut the weather the moment a user happened to
    # have finished the test.
    assert not mentions_results("It's 22 degrees in Tokyo.")
    assert not mentions_results("Your train leaves in 15 minutes.")
    assert not mentions_results("There were five of them.")
