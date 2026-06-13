# Validator + regression test for FINDING 6.2 — restart double-fires overdue
# scheduled tasks.

# Demonstrates the bug: TaskScheduler.start() aborts stale TaskRun rows but never
# advances ScheduledTask.next_run, so the in-memory _executing guard resets
# across a restart and _check_due_tasks will re-dispatch any task whose
# next_run is still in the past.

# After the fix (start() advances overdue next_run to now + 60s), the regression
# test asserts the opposite: the task fires at most once across two consecutive
# polls.


import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import core.database as cd


def _test_now() -> datetime:
    """Naive UTC 'now' consistent with database.utcnow_naive()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _stub_heavy() -> None:
    """Stub out heavy modules used by the scheduler so tests stay isolated."""
    for name in [
        "src.builtin_actions",
        "src.ai_interaction",
        "src.endpoint_resolver",
        "src.agent_loop",
        "src.session_manager",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))


def _setup_isolated_db():
    """
    Point core.database at an in-memory SQLite engine and recreate the schema.

    This uses the real Base, ScheduledTask, and TaskRun definitions from
    core.database instead of redefining models in the test. That keeps the
    test aligned with schema changes while still isolating state.
    """
    # Create a fresh in-memory engine and bind the global SessionLocal/engine
    eng = create_engine("sqlite:///:memory:")

    # Rebind the global engine and SessionLocal used by application code
    cd.engine = eng
    cd.SessionLocal = sessionmaker(bind=eng, autocommit=False, autoflush=False)

    # Create all tables from the real Base metadata
    cd.Base.metadata.create_all(eng)

    # Use the real models from core.database
    ScheduledTask = cd.ScheduledTask
    TaskRun = cd.TaskRun

    return cd, ScheduledTask, TaskRun


def test_scheduler_utcnow_preserves_naive_utc_contract():
    """
    Regression guard: _now() should return naive UTC, matching utcnow_naive().
    """
    from src.task_scheduler import _now

    now = _now()

    assert now.tzinfo is None
    assert abs((now - _test_now()).total_seconds()) < 2


async def _drive_scheduler(monkeypatch, pre_start_setup=None):
    """
    Helper that wires a TaskScheduler against an isolated in-memory DB,
    runs the startup path + two due-task polls, and captures dispatched coroutines.
    """
    _stub_heavy()
    cd_mod, ScheduledTask, TaskRun = _setup_isolated_db()
    from src.task_scheduler import TaskScheduler

    sch = TaskScheduler.__new__(TaskScheduler)
    sch._executing = set()
    sch._executing_lock = asyncio.Lock()
    sch._concurrency_cap = 1
    sch._run_semaphore = asyncio.Semaphore(1)
    sch._running = True
    sch._task = None
    sch._note_pings_task = None
    sch._known_task_owners = lambda: []
    sch._task_defer_counts = {}

    # Let the caller seed the DB with specific ScheduledTask/TaskRun rows
    if pre_start_setup:
        pre_start_setup(cd_mod, ScheduledTask, TaskRun)

    # No-op loop stub so we don't depend on a real event loop object
    def _never(*args, **kwargs):
        return None

    monkeypatch.setattr(sch, "_loop", _never)
    monkeypatch.setattr(sch, "_note_pings_loop", _never)

    dispatched = []

    def _fake_create_task(coro):
        dispatched.append(coro)

        class _T:
            def cancel(self):
                pass

        return _T()

    monkeypatch.setattr("src.task_scheduler.asyncio.create_task", _fake_create_task)

    await sch.start()
    await sch._check_due_tasks()
    await sch._check_due_tasks()

    # Only count actual task executions, not scheduler background loops
    real_dispatches = []
    for c in dispatched:
        name = getattr(c, "__name__", "")
        code = getattr(c, "cr_code", None)
        code_name = getattr(code, "co_name", "")
        if name == "_execute_task" or code_name == "_execute_task":
            real_dispatches.append(c)

    return cd, ScheduledTask, TaskRun, real_dispatches


@pytest.mark.asyncio
async def test_restart_does_not_re_dispatch_overdue_task(monkeypatch):
    """
    After restart, an overdue active task should fire at most once across
    two consecutive polls (the first poll re-fires it, but next_run is then
    advanced so the second poll does not).
    """

    def _setup(cd_mod, ScheduledTask, TaskRun):
        db = cd_mod.SessionLocal()
        try:
            db.add(
                ScheduledTask(
                    id="t_due_1",
                    owner="alice",
                    name="overdue",
                    task_type="llm",
                    next_run=_test_now() - timedelta(hours=1),
                    status="active",
                )
            )
            db.commit()
        finally:
            db.close()

    cd_mod, ScheduledTask, TaskRun, dispatched = await _drive_scheduler(
        monkeypatch, _setup
    )

    db = cd_mod.SessionLocal()
    try:
        t = db.query(ScheduledTask).filter(ScheduledTask.id == "t_due_1").first()
    finally:
        db.close()

    # After start(), next_run should have been pushed into the near future
    assert t.next_run >= _test_now() - timedelta(seconds=1), (
        f"After start(), next_run should have been pushed into the future; "
        f"got {t.next_run}"
    )

    # Across two consecutive polls, the overdue task should fire at most once
    assert len(dispatched) <= 1, (
        f"Expected at most 1 dispatch across two polls; got {len(dispatched)}. "
        "The startup next_run advance is not preventing the second poll from "
        "re-firing the same overdue task."
    )


@pytest.mark.asyncio
async def test_startup_does_not_advance_fresh_tasks(monkeypatch):
    """
    Tasks whose next_run is in the future must be untouched by the startup
    sweep — only overdue ones get pushed forward.
    """
    future = _test_now() + timedelta(hours=2)

    def _setup(cd_mod, ScheduledTask, TaskRun):
        db = cd_mod.SessionLocal()
        try:
            db.add(
                ScheduledTask(
                    id="t_fresh",
                    owner="alice",
                    name="fresh",
                    task_type="llm",
                    next_run=future,
                    status="active",
                )
            )
            db.commit()
        finally:
            db.close()

    cd_mod, ScheduledTask, TaskRun, dispatched = await _drive_scheduler(
        monkeypatch, _setup
    )

    db = cd_mod.SessionLocal()
    try:
        t = db.query(ScheduledTask).filter(ScheduledTask.id == "t_fresh").first()
    finally:
        db.close()

    assert (
        t.next_run == future
    ), f"Fresh task's next_run was modified: expected {future}, got {t.next_run}"
    assert len(dispatched) == 0


@pytest.mark.asyncio
async def test_startup_does_not_advance_paused_tasks(monkeypatch):
    """
    A paused task with an old next_run is not overdue for execution —
    it should not be advanced by the startup sweep.
    """

    def _setup(cd_mod, ScheduledTask, TaskRun):
        db = cd_mod.SessionLocal()
        try:
            db.add(
                ScheduledTask(
                    id="t_paused",
                    owner="alice",
                    name="paused",
                    task_type="llm",
                    next_run=_test_now() - timedelta(hours=1),
                    status="paused",
                )
            )
            db.commit()
        finally:
            db.close()

    cd_mod, ScheduledTask, TaskRun, dispatched = await _drive_scheduler(
        monkeypatch, _setup
    )

    db = cd_mod.SessionLocal()
    try:
        t = db.query(ScheduledTask).filter(ScheduledTask.id == "t_paused").first()
    finally:
        db.close()

    # The stored next_run should still be ~1h in the past (the startup sweep
    # only advances active overdue tasks; a paused task with an old next_run
    # is left alone). Allow a small delta to absorb the time the sweep took.
    one_hour_ago = _test_now() - timedelta(hours=1)
    assert abs((t.next_run - one_hour_ago).total_seconds()) < 5, (
        f"Paused task's next_run was modified: "
        f"expected ~{one_hour_ago}, got {t.next_run}"
    )
