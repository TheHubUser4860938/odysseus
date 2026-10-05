"""Caller model URLs cannot exceed enabled, owner-visible registration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import core.database as database
import routes.assistant_routes as assistant_routes
import routes.skills_routes as skills_routes
import routes.task.task_routes as task_routes
import src.endpoint_resolver as resolver
from tests.helpers.database import disposable_database


LOCAL = "http://localhost:1234/v1"
LAN = "http://192.168.1.20:8000/v1"
SHARED = "http://127.0.0.1:11434/api"
REJECTED = [
    "https://unregistered.example/v1",
    "http://169.254.169.254/latest/meta-data",
    "http://bob.example/v1",
    "http://disabled.example/v1",
    "http://caller:secret@localhost:1234/v1",
    LOCAL + "?api_key=caller-secret",
    LOCAL + "#fragment",
    LOCAL + "/other-base",
    "not-a-url",
    "   ",
]


def _request(body=None, owner="alice"):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=owner),
        headers={},
        json=AsyncMock(return_value=body or {}),
    )


def _route(router, method, path):
    return next(r.endpoint for r in router.routes if r.path == path and method in r.methods)


@pytest.fixture
def registered_db(tmp_path, monkeypatch):
    with disposable_database(tmp_path) as factory:
        for module in (database, task_routes, assistant_routes, resolver):
            monkeypatch.setattr(module, "SessionLocal", factory)
        with factory() as db:
            for endpoint_id, owner, url, enabled in [
                ("local", "alice", LOCAL + "/", True),
                ("lan", "alice", LAN, True),
                ("shared", None, SHARED, True),
                ("bob", "bob", "http://bob.example/v1", True),
                ("disabled", "alice", "http://disabled.example/v1", False),
            ]:
                db.add(database.ModelEndpoint(
                    id=endpoint_id, name=endpoint_id, owner=owner, base_url=url,
                    is_enabled=enabled, api_key="server-secret",
                ))
            db.add(database.ScheduledTask(
                id="task", owner="alice", name="Existing task", prompt="Work",
                task_type="llm", trigger_type="webhook", status="active",
                endpoint_url=LOCAL,
            ))
            db.add(database.CrewMember(
                id="assistant", owner="alice", name="Assistant",
                is_default_assistant=True, endpoint_url=LOCAL,
            ))
            db.commit()
        yield factory


@pytest.mark.parametrize("url", REJECTED + ["", None, 42])
def test_registered_endpoint_helper_rejects_invalid_or_invisible_url(registered_db, url):
    with registered_db() as db, pytest.raises(ValueError):
        resolver.resolve_owner_registered_endpoint_url(db, url, "alice")


@pytest.mark.parametrize("url, canonical", [
    ("HTTP://LOCALHOST:1234/v1/chat/completions/", LOCAL),
    (LAN + "/models", LAN),
    (SHARED + "/chat", SHARED),
])
def test_registered_endpoint_helper_preserves_local_lan_and_shared(registered_db, url, canonical):
    with registered_db() as db:
        assert resolver.resolve_owner_registered_endpoint_url(db, url, "alice") == canonical


@pytest.mark.asyncio
@pytest.mark.parametrize("url", REJECTED)
async def test_task_create_rejects_unregistered_url_before_persistence(registered_db, url):
    create = _route(task_routes.setup_task_routes(MagicMock()), "POST", "/api/tasks")
    with pytest.raises(HTTPException) as exc:
        await create(_request(), task_routes.TaskCreate(
            name="Rejected", prompt="Work", trigger_type="webhook", endpoint_url=url,
        ))
    assert exc.value.status_code == 400
    with registered_db() as db:
        assert db.query(database.ScheduledTask).count() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("url", REJECTED)
async def test_task_update_rejects_unregistered_url_before_persistence(registered_db, url):
    update = _route(task_routes.setup_task_routes(MagicMock()), "PUT", "/api/tasks/{task_id}")
    with pytest.raises(HTTPException) as exc:
        await update(_request(), "task", task_routes.TaskUpdate(endpoint_url=url, name="Rejected"))
    assert exc.value.status_code == 400
    with registered_db() as db:
        task = db.get(database.ScheduledTask, "task")
        assert task.endpoint_url == LOCAL
        assert task.name == "Existing task"


@pytest.mark.asyncio
async def test_task_create_and_update_store_registered_canonical_url(registered_db):
    router = task_routes.setup_task_routes(MagicMock())
    create = _route(router, "POST", "/api/tasks")
    update = _route(router, "PUT", "/api/tasks/{task_id}")
    result = await create(_request(), task_routes.TaskCreate(
        name="Accepted", prompt="Work", trigger_type="webhook",
        endpoint_url="HTTP://LOCALHOST:1234/v1/chat/completions/",
    ))
    assert result["endpoint_url"] == LOCAL
    assert (await update(_request(), "task", task_routes.TaskUpdate(
        endpoint_url=LOCAL + "/models",
    )))["endpoint_url"] == LOCAL
    with registered_db() as db:
        created = db.get(database.ScheduledTask, result["id"])
        assert created.owner == "alice"
        assert created.endpoint_url == LOCAL
        assert created.request_authority_json
        assert db.get(database.ScheduledTask, "task").endpoint_url == LOCAL


@pytest.mark.asyncio
async def test_task_empty_override_still_restores_default(registered_db):
    router = task_routes.setup_task_routes(MagicMock())
    create = _route(router, "POST", "/api/tasks")
    update = _route(router, "PUT", "/api/tasks/{task_id}")
    result = await create(_request(), task_routes.TaskCreate(
        name="Default", prompt="Work", trigger_type="webhook", endpoint_url="",
    ))
    assert result["endpoint_url"] is None
    assert (await update(_request(), "task", task_routes.TaskUpdate(endpoint_url="")))["endpoint_url"] is None


@pytest.mark.asyncio
async def test_task_update_keeps_task_owner_verification(registered_db):
    update = _route(task_routes.setup_task_routes(MagicMock()), "PUT", "/api/tasks/{task_id}")
    with pytest.raises(HTTPException) as exc:
        await update(_request(owner="bob"), "task", task_routes.TaskUpdate(endpoint_url=LOCAL))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("url", REJECTED)
async def test_assistant_settings_reject_unregistered_url(registered_db, url):
    update = _route(assistant_routes.setup_assistant_routes(MagicMock()), "PATCH", "/api/assistant/settings")
    with pytest.raises(HTTPException) as exc:
        await update(assistant_routes.AssistantSettingsUpdate(endpoint_url=url, name="Rejected"), _request())
    assert exc.value.status_code == 400
    with registered_db() as db:
        crew = db.get(database.CrewMember, "assistant")
        assert crew.endpoint_url == LOCAL
        assert crew.name == "Assistant"


@pytest.mark.asyncio
async def test_assistant_settings_accept_registered_local_endpoint(registered_db):
    update = _route(assistant_routes.setup_assistant_routes(MagicMock()), "PATCH", "/api/assistant/settings")
    await update(assistant_routes.AssistantSettingsUpdate(endpoint_url=LOCAL + "/chat/completions"), _request())
    with registered_db() as db:
        assert db.get(database.CrewMember, "assistant").endpoint_url == LOCAL


@pytest.fixture
def skill_test_route(registered_db, monkeypatch):
    manager = SimpleNamespace(
        load=lambda owner: [{"name": "skill", "owner": "alice"}],
        read_skill_md=lambda name, owner: "# Skill",
    )
    monkeypatch.setattr(resolver, "resolve_endpoint", lambda *a, **kw: (None, None, None))
    probe = MagicMock(return_value=["model"])
    monkeypatch.setattr("src.llm_core.list_model_ids", probe)
    run = AsyncMock()
    monkeypatch.setattr(skills_routes, "_run_skill_test_job", run)
    test = _route(skills_routes.setup_skills_routes(manager), "POST", "/api/skills/{skill_id}/test")
    try:
        yield test, probe, run
    finally:
        skills_routes._skill_test_jobs.pop(("alice", "skill"), None)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", REJECTED + [""])
async def test_skill_test_rejects_raw_fallback_before_network_or_execution(skill_test_route, url):
    test, probe, run = skill_test_route
    with pytest.raises(HTTPException) as exc:
        await test(_request({"endpoint_url": url, "model": "model", "headers": {"Authorization": "attacker"}}), "skill")
    assert exc.value.status_code == 400
    probe.assert_not_called()
    run.assert_not_called()
    assert ("alice", "skill") not in skills_routes._skill_test_jobs


@pytest.mark.asyncio
@pytest.mark.parametrize("server_key", ["server-secret", None])
async def test_skill_test_uses_registered_runtime_credentials(skill_test_route, registered_db, server_key):
    import asyncio

    with registered_db() as db:
        db.get(database.ModelEndpoint, "local").api_key = server_key
        db.commit()
    test, probe, run = skill_test_route
    await test(_request({
        "endpoint_url": "HTTP://LOCALHOST:1234/v1/chat/completions/", "model": "model",
        "headers": {"Authorization": "Bearer attacker", "x-api-key": "attacker", "Host": "169.254.169.254"},
    }), "skill")
    await asyncio.sleep(0)
    headers = {"Authorization": "Bearer server-secret"} if server_key else {}
    probe.assert_called_once_with(LOCAL + "/chat/completions", headers=headers)
    assert run.await_args.args[4:8] == (LOCAL + "/chat/completions", "model", headers, "alice")
    assert skills_routes._skill_test_jobs[("alice", "skill")]["_run"]["headers"] == headers


@pytest.mark.asyncio
async def test_skill_test_prefers_utility_and_ignores_request_headers(skill_test_route, monkeypatch):
    import asyncio

    test, probe, run = skill_test_route
    monkeypatch.setattr(resolver, "resolve_endpoint", lambda *a, **kw: (LAN + "/chat/completions", "model", None))
    await test(_request({"endpoint_url": REJECTED[1], "headers": {"Authorization": "attacker"}}), "skill")
    await asyncio.sleep(0)
    probe.assert_called_once_with(LAN + "/chat/completions", headers=None)
    assert run.await_args.args[6] is None


@pytest.mark.asyncio
async def test_skill_test_keeps_skill_owner_verification(skill_test_route):
    test, probe, run = skill_test_route
    with pytest.raises(HTTPException) as exc:
        await test(_request({"endpoint_url": "http://bob.example/v1", "model": "model"}, owner="bob"), "skill")
    assert exc.value.status_code == 404
    probe.assert_not_called()
    run.assert_not_called()
