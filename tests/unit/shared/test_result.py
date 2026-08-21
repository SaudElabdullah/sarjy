from sarjy.shared.result import Err, Ok


def test_ok_and_err() -> None:
    r = Ok(3)
    assert r.is_ok and r.unwrap() == 3
    e = Err("bad")
    assert not e.is_ok and e.error == "bad"
