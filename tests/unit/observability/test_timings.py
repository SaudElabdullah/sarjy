from sarjy.observability.timings import Timings


def test_timings_records_stages() -> None:
    t = Timings()
    with t.stage("auth"):
        pass
    t.mark("first_token")
    d = t.as_dict()
    assert "t_auth" in d and "t_first_token" in d
    assert all(isinstance(v, int) for v in d.values())


def test_add_records_a_duration_measured_elsewhere() -> None:
    # The HTTP gate's cost: real, and impossible for this stopwatch to time
    # itself, because it happened before the stopwatch existed.
    t = Timings()
    t.add("auth", 37)
    assert t.as_dict()["t_auth"] == 37


def test_as_dict_sums_the_per_tool_stages() -> None:
    t = Timings()
    # `add` rather than `stage`, so the numbers are chosen rather than measured:
    # what is under test is the roll-up, not the stopwatch.
    t.add("tool_get_weather", 120)
    t.add("tool_recall", 30)
    t.add("guard", 5)
    d = t.as_dict()
    # The per-tool keys survive — "which tool was slow" is the question.
    assert d["t_tool_get_weather"] == 120 and d["t_tool_recall"] == 30
    assert d["t_tool"] == 150
    # And nothing else is swept into the sum.
    assert d["t_guard"] == 5


def test_as_dict_omits_t_tool_when_no_tool_ran() -> None:
    t = Timings()
    t.mark("total")
    assert "t_tool" not in t.as_dict()


def test_as_dict_returns_a_copy() -> None:
    # It is handed to a DoneEvent and stored on a Message; a caller mutating one
    # must not reach back into the stopwatch.
    t = Timings()
    t.add("auth", 1)
    d = t.as_dict()
    d["t_auth"] = 999
    assert t.as_dict()["t_auth"] == 1
