from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.main import create_app


def test_health_live() -> None:
    app = create_app(Settings(), connect_db=False)
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "version": "0.1.0"}
