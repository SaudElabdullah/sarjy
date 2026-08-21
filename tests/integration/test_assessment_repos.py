from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest

from sarjy.contexts.assessment.application.handle_turn import HandleAssessmentTurn
from sarjy.contexts.assessment.domain.state_machine import Status
from sarjy.contexts.assessment.domain.workflow_run import WorkflowRun
from sarjy.contexts.assessment.infrastructure.pg_instrument_repo import PgInstrumentRepo
from sarjy.contexts.assessment.infrastructure.pg_run_repo import PgRunRepo
from sarjy.infrastructure_shared.db import Database
from sarjy.shared.clock import FakeClock
from sarjy.shared.errors import NotFound
from sarjy.shared.ids import RunId, UserId
from tests.unit.assessment.test_handle_turn import FakeNarrator, ScriptedInterpreter

pytestmark = pytest.mark.integration

DEF_ID = "ocean_mini_ipip"


@pytest.fixture
async def db():  # type: ignore[no-untyped-def]
    d = Database(os.environ["DATABASE_URL_DIRECT"])
    await d.connect()
    yield d
    await d.close()


async def _user(db: Database) -> UserId:
    u = uuid.uuid4()
    await db.execute("insert into auth.users (id,email) values ($1,$2)", u, f"{u}@x.test")
    return UserId(u)


@pytest.fixture
async def user(db: Database) -> UserId:
    return await _user(db)


def _new_run(user: UserId, now: datetime, version: int = 1) -> WorkflowRun:
    return WorkflowRun.propose(RunId(uuid.uuid4()), user, DEF_ID, version, now)


# --- instrument repo -------------------------------------------------------


async def test_the_seeded_definition_loads_as_a_usable_instrument(db: Database) -> None:
    ins = await PgInstrumentRepo(db).get(DEF_ID)
    assert ins.id == DEF_ID and ins.version == 1 and ins.total_items == 20
    assert ins.item(1).text == "I am the life of the party."
    assert ins.item(20).reverse is True
    assert set(ins.traits) == {"O", "C", "E", "A", "N"}
    assert len(ins.scale_labels) == 5


async def test_cached_is_empty_until_get_then_serves_without_the_database(db: Database) -> None:
    repo = PgInstrumentRepo(db)
    assert repo.cached(DEF_ID) is None
    ins = await repo.get(DEF_ID)
    assert repo.cached(DEF_ID) is ins
    await db.close()  # any further query would raise
    assert (await repo.get(DEF_ID)) is ins


async def test_an_expired_cache_entry_is_not_served(db: Database) -> None:
    repo = PgInstrumentRepo(db, ttl_s=-1.0)
    await repo.get(DEF_ID)
    assert repo.cached(DEF_ID) is None


async def test_an_unknown_definition_raises_not_found(db: Database) -> None:
    with pytest.raises(NotFound):
        await PgInstrumentRepo(db).get("no_such_instrument")


# --- run repo --------------------------------------------------------------


async def test_run_roundtrips_through_save_and_get_open(db: Database, user: UserId) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    run = _new_run(user, now)
    await repo.save(run)

    proposed = await repo.get_open(user)
    assert proposed is not None
    assert proposed.id == run.id and proposed.status is Status.PROPOSED
    assert proposed.current_item == 1 and proposed.skips_used == 0
    assert proposed.resume_hint is False and proposed.pending_confirmation is None

    run.confirm(now)
    await repo.save(run)  # the same id updates in place, it does not insert twice
    assert await db.fetchval("select count(*) from workflow_runs where id=$1", run.id) == 1

    loaded = await repo.get(run.id, user)
    assert loaded is not None and loaded.status is Status.ACTIVE


async def test_pending_confirmation_and_resume_hint_survive_a_round_trip(
    db: Database, user: UserId
) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    run = _new_run(user, now)
    run.confirm(now)
    run.set_pending(1, 4, "sort of")
    run.resume_hint = True
    await repo.save(run)

    loaded = await repo.get_open(user)
    assert loaded is not None
    assert loaded.pending_confirmation == {"item_no": 1, "value": 4, "raw_text": "sort of"}
    assert loaded.resume_hint is True

    run.clear_pending()
    run.resume_hint = False
    await repo.save(run)
    cleared = await repo.get_open(user)
    assert cleared is not None
    assert cleared.pending_confirmation is None and cleared.resume_hint is False


async def test_answers_upsert_on_item_number_so_going_back_overwrites(
    db: Database, user: UserId
) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    run = _new_run(user, now)
    run.confirm(now)
    await repo.save(run)

    await repo.save_answer(run.id, 1, "sort of", 4, 0.55)
    await repo.save_answer(run.id, 1, "four", 4, 1.0)  # re-answered after "go back"
    await repo.save_answer(run.id, 2, "skip", None, 1.0)

    assert await repo.answers(run.id) == {1: 4, 2: None}
    row = await db.fetchrow(
        "select raw_text, confidence from workflow_answers where run_id=$1 and item_no=1",
        run.id,
    )
    assert row is not None and row["raw_text"] == "four" and row["confidence"] == pytest.approx(1.0)


async def test_completed_run_keeps_its_results_narrative_and_completed_at(
    db: Database, user: UserId
) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    run = _new_run(user, now)
    run.confirm(now)
    run.begin_scoring(now, range(1, 100), 0)
    run.finish_scoring({"E": 4.5, "bands": {"E": "high"}}, "You are outgoing.", now)
    await repo.save(run)

    assert await repo.get_open(user) is None  # complete is not open
    done = await repo.latest_complete(user)
    assert done is not None
    assert done.results == {"E": 4.5, "bands": {"E": "high"}}
    assert done.narrative == "You are outgoing." and done.completed_at is not None


async def test_get_open_prefers_the_newest_and_ignores_finished_runs(
    db: Database, user: UserId
) -> None:
    repo = PgRunRepo(db)
    old = datetime(2026, 1, 1, tzinfo=UTC)
    abandoned = _new_run(user, old)
    abandoned.quit(old)
    await repo.save(abandoned)

    older_open = _new_run(user, old)
    await repo.save(older_open)
    newer_open = _new_run(user, datetime(2026, 6, 1, tzinfo=UTC))
    newer_open.confirm(datetime(2026, 6, 1, tzinfo=UTC))
    newer_open.pause(datetime(2026, 6, 2, tzinfo=UTC))
    await repo.save(newer_open)

    found = await repo.get_open(user)
    assert found is not None and found.id == newer_open.id and found.status is Status.PAUSED


async def test_reads_that_take_a_user_never_cross_between_users(db: Database) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    alice, bob = await _user(db), await _user(db)

    hers = _new_run(alice, now)
    hers.confirm(now)
    await repo.save(hers)

    assert await repo.get_open(bob) is None
    mine = await repo.get_open(alice)
    assert mine is not None and mine.id == hers.id

    hers.begin_scoring(now, range(1, 100), 0)
    hers.finish_scoring({"E": 3.0}, "steady", now)
    await repo.save(hers)
    assert await repo.latest_complete(bob) is None
    assert (await repo.latest_complete(alice)) is not None


# --- I1/I2: a run stranded in `scoring` ------------------------------------


async def test_get_open_finds_a_run_left_in_scoring(db: Database, user: UserId) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    run = _new_run(user, now)
    run.confirm(now)
    run.begin_scoring(now, range(1, 100), 0)
    await repo.save(run)

    found = await repo.get_open(user)
    assert found is not None and found.id == run.id and found.status is Status.SCORING
    assert found.results is None and found.narrative is None


async def test_the_next_turn_completes_a_run_stranded_in_scoring(
    db: Database, user: UserId
) -> None:
    """The whole recovery, against real tables: twenty answer rows and a run
    row parked in `scoring` — exactly what an interrupted pre-fix turn left
    behind — and one ordinary utterance finishes it."""
    repo = PgRunRepo(db)
    instruments = PgInstrumentRepo(db)
    ins = await instruments.get(DEF_ID)
    now = datetime.now(UTC)

    run = _new_run(user, now, version=ins.version)
    run.confirm(now)
    await repo.save(run)  # the answer rows reference it
    for n in range(1, 21):
        run.record_answer(n, 4, "four", 1.0, ins.total_items, now)
        await repo.save_answer(run.id, n, "four", 4, 1.0)
    run.begin_scoring(now, await repo.answers(run.id), ins.total_items)
    await repo.save(run)
    assert await repo.latest_complete(user) is None

    handler = HandleAssessmentTurn(
        repo, instruments, ScriptedInterpreter(), FakeNarrator(), FakeClock(now)
    )
    reply = await handler.execute(user, "hey, are we done?")

    assert reply is not None and reply.workflow["status"] == "complete"
    assert "Extraversion: 3.0 out of five" in " ".join(reply.sentences)
    done = await repo.latest_complete(user)
    assert done is not None and done.id == run.id
    assert done.results is not None and done.results["E"] == 3.0
    assert done.narrative and done.completed_at is not None
    assert await repo.get_open(user) is None  # and it is closed for good


# --- ownership is part of the query, not a caller's promise ----------------


async def test_get_by_id_hands_nothing_to_a_user_who_does_not_own_the_run(db: Database) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    alice, bob = await _user(db), await _user(db)
    hers = _new_run(alice, now)
    hers.confirm(now)
    await repo.save(hers)

    assert (await repo.get(hers.id, alice)) is not None
    # A run id that leaks — a log line, a stale client — is not a way in.
    assert (await repo.get(hers.id, bob)) is None


async def test_a_save_cannot_take_over_a_run_row_owned_by_someone_else(db: Database) -> None:
    repo = PgRunRepo(db)
    now = datetime.now(UTC)
    alice, bob = await _user(db), await _user(db)
    hers = _new_run(alice, now)
    hers.confirm(now)
    await repo.save(hers)

    # Same id, different owner: the upsert's ownership predicate makes this a
    # no-op rather than a silent takeover of Alice's run.
    stolen = WorkflowRun.propose(hers.id, bob, DEF_ID, 1, now)
    stolen.confirm(now)
    stolen.pause(now)
    await repo.save(stolen)

    row = await db.fetchrow("select user_id, status from workflow_runs where id=$1", hers.id)
    assert row is not None and UserId(row["user_id"]) == alice
    assert row["status"] == "active"
    assert await repo.get_open(bob) is None


async def test_a_stranded_scoring_run_missing_an_answer_is_reopened_not_scored(
    db: Database, user: UserId
) -> None:
    """The dangerous half of the recovery, against real tables: a run parked in
    `scoring` whose twentieth answer row never landed. Scoring it would publish
    a trait mean computed from nineteen answers as if it were twenty."""
    repo = PgRunRepo(db)
    instruments = PgInstrumentRepo(db)
    ins = await instruments.get(DEF_ID)
    now = datetime.now(UTC)

    run = _new_run(user, now, version=ins.version)
    run.confirm(now)
    await repo.save(run)
    for n in range(1, 21):
        run.record_answer(n, 4, "four", 1.0, ins.total_items, now)
        await repo.save_answer(run.id, n, "four", 4, 1.0)
    run.begin_scoring(now, await repo.answers(run.id), ins.total_items)
    await repo.save(run)
    await db.execute("delete from workflow_answers where run_id=$1 and item_no=$2", run.id, 7)
    assert len(await repo.answers(run.id)) == 19

    handler = HandleAssessmentTurn(
        repo, instruments, ScriptedInterpreter(), FakeNarrator(), FakeClock(now)
    )
    reply = await handler.execute(user, "are we done yet?")

    assert reply is not None and reply.workflow["status"] == "active"
    assert reply.workflow["item"] == 7
    assert await repo.latest_complete(user) is None
    reopened = await repo.get_open(user)
    assert reopened is not None and reopened.status is Status.ACTIVE
    assert reopened.current_item == 7 and reopened.results is None

    # Answer the gap and the fourteen items after it, and it completes.
    for _ in range(14):
        reply = await handler.execute(user, "four")
    assert reply is not None and reply.workflow["status"] == "complete"
    done = await repo.latest_complete(user)
    assert done is not None and done.results is not None and done.results["E"] == 3.0
