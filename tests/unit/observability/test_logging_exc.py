import json

from sarjy.observability.logging import configure_logging, get_logger


def test_exception_traceback_is_rendered(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging("INFO")
    log = get_logger("t")
    try:
        raise ValueError("boom-marker")
    except ValueError:
        log.exception("turn_failed")
    line = capsys.readouterr().out.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["event"] == "turn_failed"
    assert "boom-marker" in rec["exception"] and "Traceback" in rec["exception"]
