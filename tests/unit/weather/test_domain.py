from datetime import UTC, date, datetime

from sarjy.contexts.weather.domain.condition import Condition
from sarjy.contexts.weather.domain.forecast import Forecast
from sarjy.contexts.weather.domain.location import Location
from sarjy.contexts.weather.domain.units import Units
from sarjy.contexts.weather.domain.when import day_label, index_for_when, parse_when

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def _fc() -> Forecast:
    return Forecast.from_metric(
        temp_c=22.4,
        feels_like_c=21.0,
        condition=Condition.from_wmo(2),
        precip_prob=40,
        wind_kph=12.3,
        humidity=55,
        high_c=26.0,
        low_c=17.5,
        day=date(2026, 8, 21),
        observed_at=NOW,
        source="open-meteo",
        fetched_at=NOW,
    )


def _forecast_day_fc() -> Forecast:
    """A future day: high, low, condition and chance of rain — nothing else."""
    return Forecast.from_metric(
        temp_c=None,
        feels_like_c=None,
        condition=Condition.from_wmo(63),
        precip_prob=80,
        wind_kph=None,
        humidity=None,
        high_c=12.0,
        low_c=6.0,
        day=date(2026, 8, 22),
        observed_at=NOW,
        source="open-meteo",
        fetched_at=NOW,
    )


def test_location_label() -> None:
    assert Location("Tokyo", "Japan", 35.68, 139.69).label == "Tokyo, Japan"


def test_condition_from_wmo_table() -> None:
    assert Condition.from_wmo(0).text == "clear sky"
    assert Condition.from_wmo(2).text == "partly cloudy"
    assert Condition.from_wmo(61).text == "light rain"
    assert Condition.from_wmo(95).text == "thunderstorm"
    assert Condition.from_wmo(999).text == "unknown conditions"


def test_forecast_converts_fahrenheit() -> None:
    fc = _fc()
    assert fc.temp_f == 72.3


def test_grounding_numbers_include_c_and_f_and_rounded() -> None:
    nums = _fc().grounding_numbers()
    assert 22.4 in nums and 22.0 in nums and 72.3 in nums and 72.0 in nums
    assert (
        40.0 in nums
        and 12.3 in nums
        and 55.0 in nums
        and 26.0 in nums
        and 17.5 in nums
        and 18.0 in nums
    )


def test_tool_payload_shape_metric() -> None:
    p = _fc().to_tool_payload(Location("Tokyo", "Japan", 35.68, 139.69), Units.METRIC)
    assert p["location"] == {"name": "Tokyo", "country": "Japan"}
    assert p["temp_c"] == 22.4 and p["temp_f"] == 72.3 and p["units"] == "metric"
    assert p["condition_text"] == "partly cloudy"
    assert p["source"] == "open-meteo"
    assert p["day"] == "2026-08-21" and p["day_label"] == "right now"


def test_tool_payload_carries_no_number_the_model_may_not_speak() -> None:
    # Coordinates, the numeric condition code and the timestamps are all
    # ungrounded numbers that read as plausible weather; the guard would reject
    # them if spoken, so they have no business in the payload at all.
    fc = _fc()
    p = fc.to_tool_payload(Location("Tokyo", "Japan", 35.68, 139.69), Units.METRIC)
    for key in ("observed_at", "fetched_at", "condition_code"):
        assert key not in p
    assert "lat" not in p["location"] and "lon" not in p["location"]

    grounded = set(fc.grounding_numbers())
    spoken_numbers = [v for v in p.values() if isinstance(v, int | float)]
    assert spoken_numbers and all(float(v) in grounded for v in spoken_numbers)


def test_tool_payload_labels_a_forecast_day() -> None:
    p = _forecast_day_fc().to_tool_payload(
        Location("Reykjavik", "Iceland", 64.1, -21.9), Units.METRIC, "tomorrow"
    )
    assert p["day"] == "2026-08-22" and p["day_label"] == "tomorrow"


def test_tool_payload_omits_unobserved_fields_entirely() -> None:
    p = _forecast_day_fc().to_tool_payload(
        Location("Reykjavik", "Iceland", 64.1, -21.9), Units.METRIC
    )
    for key in (
        "temp_c",
        "temp_f",
        "feels_like_c",
        "feels_like_f",
        "wind_kph",
        "wind_mph",
        "humidity",
    ):
        assert key not in p, f"{key} should be absent, not null"
    # What was actually forecast is still there.
    assert p["high_c"] == 12.0 and p["low_c"] == 6.0 and p["precip_prob"] == 80
    assert p["condition_text"] == "rain"


def test_grounding_numbers_skip_unobserved_fields() -> None:
    nums = _forecast_day_fc().grounding_numbers()
    assert 12.0 in nums and 6.0 in nums and 80.0 in nums
    assert 53.6 in nums and 42.8 in nums  # high/low in fahrenheit
    # 0.0 would be the tell-tale of a fabricated wind speed or humidity.
    assert 0.0 not in nums


def test_spoken_summary_metric_is_grounded_and_spoken() -> None:
    s = _fc().spoken_summary(Location("Tokyo", "Japan", 35.68, 139.69), Units.METRIC)
    assert s == "It's twenty-two degrees and partly cloudy in Tokyo right now."


def test_spoken_summary_imperial_uses_fahrenheit() -> None:
    s = _fc().spoken_summary(Location("Tokyo", "Japan", 35.68, 139.69), Units.IMPERIAL)
    assert s == "It's seventy-two degrees and partly cloudy in Tokyo right now."


def test_spoken_summary_for_a_forecast_day_speaks_high_and_low() -> None:
    s = _forecast_day_fc().spoken_summary(
        Location("Reykjavik", "Iceland", 64.1, -21.9), Units.METRIC, "tomorrow"
    )
    assert s == "Tomorrow in Reykjavik expect a high of twelve and a low of six with rain."


def test_spoken_summary_uses_a_verb_for_an_adjectival_condition() -> None:
    # "with partly cloudy" is not English; "with rain" is.
    fc = Forecast.from_metric(
        temp_c=None,
        feels_like_c=None,
        condition=Condition.from_wmo(2),
        precip_prob=10,
        wind_kph=None,
        humidity=None,
        high_c=12.0,
        low_c=6.0,
        day=date(2026, 8, 22),
        observed_at=NOW,
        source="open-meteo",
        fetched_at=NOW,
    )
    s = fc.spoken_summary(Location("Reykjavik", "Iceland", 64.1, -21.9), Units.METRIC, "tomorrow")
    assert s == (
        "Tomorrow in Reykjavik expect a high of twelve and a low of six "
        "and it will be partly cloudy."
    )


def test_spoken_summary_for_a_named_day_capitalises_the_label() -> None:
    s = _forecast_day_fc().spoken_summary(
        Location("Reykjavik", "Iceland", 64.1, -21.9), Units.METRIC, "on Monday"
    )
    assert s.startswith("On Monday in Reykjavik expect a high of")


def test_spoken_summary_for_a_forecast_day_imperial_uses_fahrenheit() -> None:
    s = _forecast_day_fc().spoken_summary(
        Location("Reykjavik", "Iceland", 64.1, -21.9), Units.IMPERIAL, "tomorrow"
    )
    assert (
        s == "Tomorrow in Reykjavik expect a high of fifty-four and a low of forty-three with rain."
    )


def test_day_label_is_taken_from_the_request_token() -> None:
    assert day_label("now") == "right now"
    assert day_label("today") == "right now"
    assert day_label("") == "right now"
    assert day_label("tomorrow") == "tomorrow"
    assert day_label("2026-08-24") == "on Monday"
    assert day_label("someday") == "right now"


def test_parse_when() -> None:
    today = date(2026, 8, 21)
    assert parse_when("now", today) == today
    assert parse_when("today", today) == today
    assert parse_when("tomorrow", today) == date(2026, 8, 22)
    assert parse_when("2026-08-27", today) == date(2026, 8, 27)
    assert parse_when("2026-08-29", today) is None  # > 7 days
    assert parse_when("2026-08-20", today) is None  # past
    assert parse_when("someday", today) is None


def test_parse_when_slack_widens_the_window_at_both_ends() -> None:
    # The location's calendar may be a day either side of the server's, so
    # validation has to be looser than the range the provider will enforce.
    today = date(2026, 8, 21)
    assert parse_when("2026-08-20", today, slack_days=1) == date(2026, 8, 20)
    assert parse_when("2026-08-29", today, slack_days=1) == date(2026, 8, 29)
    assert parse_when("2026-08-19", today, slack_days=1) is None
    assert parse_when("2026-08-30", today, slack_days=1) is None


def test_index_for_when_resolves_against_the_providers_own_calendar() -> None:
    # days[0] is the *location's* today, whatever the server's date is.
    days = [date(2026, 8, 22), date(2026, 8, 23), date(2026, 8, 24)]
    assert index_for_when("now", days) == 0
    assert index_for_when("today", days) == 0
    assert index_for_when("", days) == 0
    assert index_for_when("tomorrow", days) == 1
    assert index_for_when("2026-08-24", days) == 2
    assert index_for_when("2026-08-21", days) is None  # behind the location's today
    assert index_for_when("2026-08-25", days) is None
    assert index_for_when("someday", days) is None
    assert index_for_when("now", []) is None
    assert index_for_when("tomorrow", [date(2026, 8, 22)]) is None
