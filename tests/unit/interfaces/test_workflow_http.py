"""`GET /workflow/latest` — read-only status/results endpoint (Phase 6 T5).

Uses in-memory assessment repos (`c.run_repo`, `c.instrument_repo`) swapped
onto the container the same way other unit tests swap in `use_in_memory_repos()`
pieces — see `Container.run_repo`/`Container.instrument_repo`.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from sarjy.config import Settings
from sarjy.contexts.assessment.domain.instrument import Instrument
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.assessment.infrastructure.memory_repos import MemInstrumentRepo, MemRunRepo
from sarjy.main import create_app
from sarjy.shared.ids import RunId, UserId
from tests.unit.assessment.test_handle_turn import _seed_scoring_run

SECRET = "super-secret-jwt-token-with-at-least-32-characters-long"  # noqa: S105
INS = Instrument.from_definition(
    json.loads((Path(__file__).parents[3] / "supabase" / "mini_ipip.json").read_text())
)


def _tok(u: uuid.UUID) -> str:
    claims = {"sub": str(u), "aud": "authenticated", "exp": int(time.time()) + 60}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def _auth(u: uuid.UUID) -> dict[str, str]:
    return {"Authorization": f"Bearer {_tok(u)}"}


def _app_with_repos() -> tuple[object, MemRunRepo]:
    app = create_app(Settings(), connect_db=False)
    c = app.state.container
    run_repo = MemRunRepo()
    c.run_repo, c.instrument_repo = run_repo, MemInstrumentRepo({INS.id: INS})
    return app, run_repo


def _save(run_repo: MemRunRepo, run: WorkflowRun) -> None:
    asyncio.run(run_repo.save(run))


def test_latest_404_with_no_run() -> None:
    app, _ = _app_with_repos()
    u = uuid.uuid4()
    with TestClient(app) as client:
        r = client.get("/workflow/latest", headers=_auth(u))
    assert r.status_code == 404
    assert r.json()["detail"] == "no_run"


def test_latest_open_run_shape() -> None:
    app, run_repo = _app_with_repos()
    u = uuid.uuid4()
    now = datetime(2026, 8, 21, tzinfo=UTC)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), UserId(u), INS.id, 1, now)
    run.confirm(now)
    run.record_answer(1, 4, "four", 1.0, INS.total_items, now)
    _save(run_repo, run)

    with TestClient(app) as client:
        r = client.get("/workflow/latest", headers=_auth(u))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "active"
    assert body["run_id"] == str(run.id)
    assert body["definition_id"] == INS.id
    assert body["current_item"] == 2
    assert body["total_items"] == 20
    assert body["results"] is None
    assert body["narrative"] is None
    assert body["bands"] is None
    assert "not a clinical or diagnostic tool" in body["disclaimer"]


def test_latest_complete_run_with_results() -> None:
    app, run_repo = _app_with_repos()
    u = uuid.uuid4()
    now = datetime(2026, 8, 21, tzinfo=UTC)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), UserId(u), INS.id, 1, now)
    run.confirm(now)
    for n in range(1, 21):
        run.record_answer(n, 4, "four", 1.0, 20, now)
    run.begin_scoring(now, {n: 4 for n in range(1, 21)}, 20)
    results = {
        "O": 3.0,
        "C": 3.0,
        "E": 3.0,
        "A": 3.0,
        "N": 3.0,
        "bands": {k: "moderate" for k in "OCEAN"},
        "answered": 20,
        "skipped": 0,
    }
    run.finish_scoring(results, "Nice.", now)
    _save(run_repo, run)

    with TestClient(app) as client:
        r = client.get("/workflow/latest", headers=_auth(u))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["run_id"] == str(run.id)
    assert body["current_item"] == 20
    assert body["total_items"] == 20
    assert body["results"]["E"] == 3.0
    assert body["narrative"] == "Nice."
    assert body["bands"] == {k: "moderate" for k in "OCEAN"}


def test_latest_is_scoped_to_the_caller() -> None:
    app, run_repo = _app_with_repos()
    owner, other = uuid.uuid4(), uuid.uuid4()
    now = datetime(2026, 8, 21, tzinfo=UTC)
    run = WorkflowRun.propose(RunId(uuid.uuid4()), UserId(owner), INS.id, 1, now)
    run.confirm(now)
    _save(run_repo, run)

    with TestClient(app) as client:
        r = client.get("/workflow/latest", headers=_auth(other))
    assert r.status_code == 404


def test_latest_reports_a_stranded_scoring_run_as_scoring() -> None:
    # I1/I2: the run exists and its answers are safe, but no results have been
    # computed yet. Saying "complete" would invent them; 404 would deny the run.
    app, run_repo = _app_with_repos()
    u = uuid.uuid4()
    asyncio.run(_seed_scoring_run(run_repo, UserId(u)))

    with TestClient(app) as client:
        r = client.get("/workflow/latest", headers=_auth(u))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "scoring"
    assert body["results"] is None and body["narrative"] is None and body["bands"] is None
    assert body["current_item"] == 20 and body["total_items"] == 20
